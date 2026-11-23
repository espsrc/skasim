"""imaging.py — dirty (OSKAR) and cleaned (WSClean) imaging wrappers."""

from __future__ import annotations

import gc
import os
import re
import shlex
import subprocess
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from loguru import logger

from .config import ImgConfig, resolve_spectral_mode
from .manifest import RunContext
from .runtime import require_karabo_module

SKY_MODEL_CMAP = "viridis_r"

# Longest side, in pixels, of the image handed to APLpy for a PNG preview. A
# preview PNG is ~1200 px wide, so larger images are block-averaged first:
# rendering and contouring 8192^2 images exhausted memory in long runs.
PREVIEW_MAX_PIXELS = 2048

# --------------------------------------------------------------------------- #
# spectral-cube imaging helpers
# --------------------------------------------------------------------------- #


def _resolve_spectral_cube_wsclean_config(
    img_config: ImgConfig,
    n_channels: int,
) -> ImgConfig:
    """Return an imaging config adjusted for per-channel cube output.

    - join_channels forced to False
    - channels_out forced to n_channels if the user did not explicitly set it
    - multiscale disabled (line cubes rarely benefit from multiscale)
    """
    kwargs = img_config.model_dump(exclude_unset=True)
    kwargs["join_channels"] = False
    kwargs["multiscale"] = False
    kwargs["multiscale_scales"] = None
    if img_config.channels_out is None:
        kwargs["channels_out"] = n_channels
    return ImgConfig(**kwargs)


def resolve_wsclean_img_config(
    img_config: ImgConfig, spectral_mode: str, n_channels: int
) -> ImgConfig:
    """Return the WSClean settings actually used for one imaging block.

    - ``cube``: per-channel imaging (see ``_resolve_spectral_cube_wsclean_config``).
    - explicit ``continuum``: a single image of all channels combined
      (``channels_out=1``, no channel joining) unless ``channels_out`` is set.
    - ``auto`` resolving to continuum: the settings are used unchanged.
    """
    if spectral_mode == "cube":
        return _resolve_spectral_cube_wsclean_config(img_config, n_channels)
    if img_config.spectral_mode == "continuum" and img_config.channels_out is None:
        kwargs = img_config.model_dump(exclude_unset=True)
        kwargs["channels_out"] = 1
        kwargs["join_channels"] = False
        return ImgConfig(**kwargs)
    return img_config


def effective_wsclean_channels_out(img_config: ImgConfig, n_channels: int) -> int:
    """Return the ``-channels-out`` value ``build_wsclean_argv`` will use."""
    if img_config.channels_out is not None:
        return int(img_config.channels_out)
    return int(n_channels)


# --------------------------------------------------------------------------- #
# dirty imaging (OSKAR)
# --------------------------------------------------------------------------- #


def run_dirty_imaging(
    ctx: RunContext,
    visibility_path: Path,
    fov: u.Quantity,
    center: SkyCoord,
    img_config: ImgConfig,  # << NEW
    sub_dir: Path,  # << NEW — work_dir/{tag}
) -> None:
    """produce dirty image via OSKAR."""
    imager_module = require_karabo_module("karabo.imaging.imager_oskar")
    visibility_module = require_karabo_module("karabo.simulation.visibility")
    work_dir = sub_dir
    vis = visibility_module.Visibility(str(visibility_path))
    imaging_cellsize = fov / img_config.pixels
    cfg = imager_module.OskarDirtyImagerConfig(
        imaging_npixel=img_config.pixels,
        imaging_cellsize=imaging_cellsize.to(u.rad).value,
        combine_across_frequencies=True,
        imaging_phase_centre=center.icrs,
    )
    imager = imager_module.OskarDirtyImager(config=cfg)
    dirty_image = imager.create_dirty_image(vis)

    dirty_png = work_dir / f"{img_config.tag}_dirty.png"
    dirty_fits = work_dir / f"{img_config.tag}_dirty.fits"
    dirty_image.write_to_file(str(dirty_fits), overwrite=True)
    try:
        write_fits_preview(dirty_fits, dirty_png, "OSKAR Dirty Image")
    except Exception as e:
        logger.warning(f"Failed to generate APLpy dirty image preview: {e}")

    logger.debug(f"Dirty PNG: {dirty_png}")
    logger.debug(f"Dirty FITS: {dirty_fits}")

    image_product_id = f"{img_config.tag}_dirty"
    ctx.manifest.add_output(
        "image_product",
        str(dirty_png.relative_to(ctx.work_dir)),
        image_product_id=image_product_id,
        imager="oskar-dirty",
        role="preview",
        metadata={"tag": img_config.tag},
    )
    ctx.manifest.add_output(
        "image_product",
        str(dirty_fits.relative_to(ctx.work_dir)),
        image_product_id=image_product_id,
        imager="oskar-dirty",
        role="dirty",
        metadata={"tag": img_config.tag},
    )


# --------------------------------------------------------------------------- #
# cleaned imaging (WSClean)
# --------------------------------------------------------------------------- #


def build_wsclean_argv(
    img_config: ImgConfig,
    visibility_path: Path,
    fov: u.Quantity,
    output_prefix: str,
    n_channels: int = 1,
) -> list[str]:
    """Build a shell-free WSClean argv list from the resolved imaging config."""
    imaging_cellsize = fov / img_config.pixels
    channels_out = effective_wsclean_channels_out(img_config, n_channels)

    argv = shlex.split(img_config.wsclean_command) + [
        "-weight",
        "briggs",
        str(img_config.robust),
        "-size",
        str(img_config.pixels),
        str(img_config.pixels),
        "-scale",
        f"{imaging_cellsize.to(u.arcsec).value:.6f}asec",
        "-niter",
        str(img_config.clean_iterations),
        "-mgain",
        str(img_config.mgain if img_config.mgain is not None else 0.8),
        "-auto-threshold",
        str(
            img_config.auto_threshold if img_config.auto_threshold is not None else 0.3
        ),
        "-auto-mask",
        str(img_config.auto_mask if img_config.auto_mask is not None else 3.0),
        "-channels-out",
        str(channels_out),
        "-name",
        output_prefix,
    ]

    if img_config.multiscale is not False:
        argv.append("-multiscale")
        if img_config.multiscale_scales:
            argv += [
                "-multiscale-scales",
                ",".join(str(s) for s in img_config.multiscale_scales),
            ]
    if img_config.local_rms is not False:
        argv.append("-local-rms")
    if img_config.join_channels is not False:
        argv.append("-join-channels")
    if img_config.padding is not None:
        argv += ["-padding", str(img_config.padding)]
    if img_config.threads is not None:
        argv += ["-j", str(img_config.threads)]

    argv.append(str(visibility_path))
    return argv


def run_wsclean_command(argv: list[str], work_dir: Path):
    """Run WSClean with argv and an explicit working directory.

    Streams stdout/stderr line-by-line through loguru so that WSClean
    progress appears in the skasim logs in real time.
    """
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    lines: list[str] = []
    with subprocess.Popen(
        argv,
        shell=False,
        cwd=str(work_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ) as proc:
        assert proc.stdout is not None  # guaranteed by PIPE
        for line in proc.stdout:
            stripped = line.rstrip("\n")
            logger.info("[wsclean] {}", stripped)
            lines.append(line)
        proc.wait()
    combined = "".join(lines)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode,
            argv,
            output=combined,
            stderr=None,
        )
    return subprocess.CompletedProcess(
        argv, proc.returncode, stdout=combined, stderr=""
    )


def wsclean_output_prefix(ctx: RunContext) -> str:
    """Return the stable WSClean output prefix for this run."""
    return f"{ctx.work_dir.name}_wsclean"


def collect_wsclean_outputs(work_dir: Path, output_prefix: str) -> list[Path]:
    """Collect WSClean FITS outputs for one configured output prefix."""
    return sorted(work_dir.glob(f"{output_prefix}*.fits"))


_WSCLEAN_CHANNEL_INDEX = re.compile(r"^\d{4,}-")
_WSCLEAN_TITLES = {
    "image": "Cleaned image (WSClean)",
    "model": "Component model (WSClean)",
    "residual": "Residual (WSClean)",
    "dirty": "Dirty image (WSClean)",
    "psf": "Point spread function (WSClean)",
}


def wsclean_product_name(
    filename: str, output_prefix: str | None
) -> tuple[str | None, bool]:
    """Split a WSClean output name into ``(product, per_channel)``.

    WSClean names outputs ``<prefix>-<product>`` with ``-channels-out 1``, and
    ``<prefix>-NNNN-<product>`` per channel plus ``<prefix>-MFS-<product>``
    otherwise. ``product`` is e.g. ``image`` or ``psf`` (extension removed);
    it is ``None`` when the name does not start with ``<prefix>-``, in which
    case ``per_channel`` falls back to spotting a ``-NNNN-`` index.
    """
    name = Path(filename).name
    if not output_prefix or not name.startswith(f"{output_prefix}-"):
        return None, re.search(r"-\d{4,}-[^.]+\.\w+$", name) is not None
    rest = name[len(output_prefix) + 1 :].rsplit(".", 1)[0]
    per_channel = _WSCLEAN_CHANNEL_INDEX.match(rest) is not None
    rest = _WSCLEAN_CHANNEL_INDEX.sub("", rest)
    if rest.startswith("MFS-"):
        rest = rest[len("MFS-") :]
    return rest, per_channel


def run_wsclean_imaging(
    ctx: RunContext,
    visibility_path: Path,
    fov: u.Quantity,
    img_config: ImgConfig,  # << NEW
    sub_dir: Path,  # << NEW — work_dir/{tag}
    n_channels: int = 1,
) -> None:
    """produce cleaned image via external WSClean binary."""
    wsclean_module = require_karabo_module("karabo.imaging.imager_wsclean")
    file_handler_module = require_karabo_module("karabo.util.file_handler")
    work_dir = sub_dir

    output_prefix = f"{img_config.tag}_wsclean"

    # cube mode forces per-channel imaging; explicit continuum mode collapses
    # all channels into one image (see resolve_wsclean_img_config)
    spectral_mode = resolve_spectral_mode(img_config, ctx.config)
    img_config = resolve_wsclean_img_config(img_config, spectral_mode, n_channels)

    argv = build_wsclean_argv(
        img_config,
        visibility_path,
        fov,
        output_prefix=output_prefix,
        n_channels=n_channels,
    )
    logger.info(f"WSClean command: {argv}")

    file_handler_module.FileHandler().get_tmp_dir(
        prefix=wsclean_module.TMP_PREFIX_CUSTOM,
        purpose=wsclean_module.TMP_PURPOSE_CUSTOM,
    )
    run_wsclean_command(argv, work_dir)

    # remove the temporary files created by WSClean
    for tmp in work_dir.glob("wsclean-00*.fits"):
        if tmp.name.startswith(output_prefix):
            continue
        try:
            tmp.unlink()
        except Exception:
            logger.exception("Failed to clean up temp file %s", tmp)

    wsclean_outputs = collect_wsclean_outputs(work_dir, output_prefix)
    # combined products: <prefix>-MFS-<product>, or <prefix>-<product> when
    # WSClean writes a single output channel
    combined_files = [
        p.name
        for p in wsclean_outputs
        if not wsclean_product_name(p.name, output_prefix)[1]
    ]
    logger.info(f"Combined-channel files: {combined_files}")

    for img_path in wsclean_outputs:
        png_name = img_path.with_suffix(".png").name
        png_path = work_dir / png_name

        # infer image type from filename for correct plot title
        product, per_channel = wsclean_product_name(img_path.name, output_prefix)
        title = "Imaging output (WSClean)"
        if not per_channel:
            title = _WSCLEAN_TITLES.get(product or "", title)
        is_psf = not per_channel and product == "psf"

        if is_psf:
            try:
                write_psf_profile_preview(img_path, png_path, title)
            except Exception as exc:
                logger.warning(f"Failed to generate PSF profile preview: {exc}")
                write_fits_preview(img_path, png_path, title)
        else:
            write_fits_preview(img_path, png_path, title)
        role = "image"
        # classify by the product part only, so a tag such as "model_test"
        # does not turn every output into a model
        lower_name = (product or img_path.name).lower()
        if "model" in lower_name:
            role = "model"
        elif "residual" in lower_name:
            role = "residual"
        elif "dirty" in lower_name:
            role = "dirty"
        elif "psf" in lower_name:
            role = "psf"
        ctx.manifest.add_output(
            "image_product",
            str(png_path.relative_to(ctx.work_dir)),
            image_product_id=output_prefix,
            imager="wsclean",
            role=f"{role}_preview",
            metadata={"tag": img_config.tag},
        )
        ctx.manifest.add_output(
            "image_product",
            str(img_path.relative_to(ctx.work_dir)),
            image_product_id=output_prefix,
            imager="wsclean",
            role=role,
            metadata={"tag": img_config.tag},
        )

    # collect per-channel FITS clean images and stack them into a single 3D cube.
    # WSClean writes per-channel products as <prefix>-<dddd>-<product>.fits.
    per_channel_images = sorted(work_dir.glob(f"{output_prefix}-????-image.fits"))
    if per_channel_images:
        cube_fits = work_dir / f"{output_prefix}-cube-image.fits"
        if cube_fits.exists():
            cube_fits.unlink()
        try:
            stack_channels(per_channel_images, cube_fits)
        except Exception as exc:
            logger.warning(f"Failed to stack clean cube: {exc}")
        else:
            png_name = cube_fits.with_suffix(".png").name
            png_path = work_dir / png_name
            write_fits_preview(cube_fits, png_path, "Spectral cube (clean)")
            ctx.manifest.add_output(
                "image_product",
                str(png_path.relative_to(ctx.work_dir)),
                image_product_id=output_prefix,
                imager="wsclean",
                role="cube_image_preview",
                metadata={"tag": img_config.tag},
            )
            ctx.manifest.add_output(
                "image_product",
                str(cube_fits.relative_to(ctx.work_dir)),
                image_product_id=output_prefix,
                imager="wsclean",
                role="cube_image",
                metadata={"tag": img_config.tag},
            )

    # produce only moment 8 for spectral-cube clean outputs
    if spectral_mode == "cube":
        from .loaders.image_models import run_moment8_for_spectral_cube

        run_moment8_for_spectral_cube(ctx, work_dir, output_prefix, img_config.tag)


def stack_channels(channel_paths: list[Path], output_path: Path) -> None:
    """Stack WSClean per-channel FITS images into a single 3D spectral cube.

    Uses the same logic as ``scripts/wsclean_channels_to_cube.py`` but inlined
    here so it works without putting ``scripts/`` on ``PYTHONPATH``.
    """
    import re

    channels: list[tuple[float, int, np.ndarray, fits.Header]] = []
    for path in channel_paths:
        match = re.search(r"-([0-9]{4,})-[a-zA-Z0-9_]+\.fits$", path.name)
        if not match:
            continue
        ch = int(match.group(1))
        with fits.open(path) as hdul:
            hdu = hdul[0]
            data = np.squeeze(np.asarray(hdu.data, dtype=np.float32))
            if data.ndim != 2:
                raise ValueError(
                    f"{path}: expected 2D image after squeezing, got shape {hdu.data.shape}"
                )
            header = hdu.header.copy()
        crpix3 = header.get("CRPIX3", 1.0)
        crval3 = header.get("CRVAL3")
        cdelt3 = header.get("CDELT3")
        cunit3 = (header.get("CUNIT3") or "").strip().lower()
        if crval3 is None or cdelt3 is None:
            raise ValueError(f"{path}: missing CRVAL3/CDELT3")
        # Each WSClean channel image carries its own one-pixel FREQ axis: the
        # channel centre is the world value at pixel 1 (CRVAL3 when CRPIX3 is
        # 1) and CDELT3 is that channel's width, not a cube-wide spacing.
        freq_hz = crval3 + (1.0 - crpix3) * cdelt3
        if cunit3 == "mhz":
            freq_hz *= 1e6
        channels.append((float(freq_hz), ch, data, header))

    if not channels:
        raise ValueError("no valid WSClean per-channel images found")

    channels.sort(key=lambda x: (x[0], x[1]))
    first_data = channels[0][2]
    cube = np.empty((len(channels), *first_data.shape), dtype=np.float32)
    for i, (_, ch, data, _) in enumerate(channels):
        if data.shape != first_data.shape:
            raise ValueError(
                f"channel {ch} has shape {data.shape}, expected {first_data.shape}"
            )
        cube[i] = data

    out_header = channels[0][3].copy()
    # force 3D header
    for key in list(out_header.keys()):
        if re.match(r"NAXIS\d+", key):
            out_header.remove(key, ignore_missing=True)
    out_header["NAXIS"] = 3
    out_header["NAXIS1"] = channels[0][3].get("NAXIS1")
    out_header["NAXIS2"] = channels[0][3].get("NAXIS2")
    out_header["NAXIS3"] = len(channels)
    freqs = np.array([c[0] for c in channels])
    if len(freqs) > 1:
        df = float(freqs[1] - freqs[0])
        if not np.allclose(np.diff(freqs), df, rtol=1e-6, atol=0.0):
            logger.warning(
                "stack_channels: channel frequencies are not evenly spaced; "
                "the cube's linear FREQ axis (CDELT3={} Hz) is approximate.",
                df,
            )
    else:
        # one channel: use its width
        header0 = channels[0][3]
        df = float(header0.get("CDELT3") or 1.0)
        if (header0.get("CUNIT3") or "").strip().lower() == "mhz":
            df *= 1e6
    out_header["CRPIX3"] = 1.0
    out_header["CRVAL3"] = float(freqs[0])
    out_header["CDELT3"] = float(df)
    out_header["CTYPE3"] = "FREQ"
    out_header["CUNIT3"] = "Hz"
    out_header["HISTORY"] = "stacked by skasim.imaging.stack_channels"

    fits.writeto(output_path, cube, out_header, overwrite=True)


# --------------------------------------------------------------------------- #
# image previews (FITS -> PNG via APLpy)
# --------------------------------------------------------------------------- #


def write_fits_preview(
    img_path: Path,
    png_path: Path,
    title: str,
    recenter: tuple[float, float, float] | None = None,
    scale_factor: float = 1000.0,
    bunit: str = "mJy/beam",
    colorbar_label: str = "mJy/beam",
) -> None:
    """Write a publication-style PNG preview for a WSClean FITS image, optionally recentered.

    Only the first 2D plane is read. Images larger than ``PREVIEW_MAX_PIXELS``
    are block-averaged before rendering; the colour scale and contour levels
    come from the full-resolution plane.
    """
    try:
        _render_fits_preview(
            img_path, png_path, recenter, scale_factor, bunit, colorbar_label
        )
    finally:
        # APLpy figures hold reference cycles; collect them now so a run that
        # writes dozens of previews does not keep every image in memory.
        gc.collect()


def _render_fits_preview(
    img_path: Path,
    png_path: Path,
    recenter: tuple[float, float, float] | None,
    scale_factor: float,
    bunit: str,
    colorbar_label: str,
) -> None:
    """Render one FITS preview; see :func:`write_fits_preview`."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import aplpy
    import cmasher as cmr
    import matplotlib.pyplot as plt

    data, header = _read_first_plane(img_path)
    data *= scale_factor
    data[~np.isfinite(data)] = np.nan
    has_finite = not np.isnan(data).all()
    if has_finite:
        rms = float(np.nanstd(data))
        vmin = float(np.nanmin(data))
        vmax = float(np.nanmax(data))
        if rms > 0:
            vmin = max(vmin, -2.0 * rms)
            vmax = min(vmax, 20.0 * rms)
    else:
        rms = 0.0
        vmin = vmax = None

    hdu = _make_2d_preview_hdu(data, header, bunit=bunit, max_pixels=PREVIEW_MAX_PIXELS)
    del data
    hdul = fits.HDUList([hdu])
    fig = plt.figure(figsize=(8, 7))
    try:
        ffig = aplpy.FITSFigure(hdul, figure=fig)

        if recenter:
            ra_deg, dec_deg, fov_deg = recenter
            ffig.recenter(ra_deg, dec_deg, width=fov_deg, height=fov_deg)

        cmap = cmr.get_sub_cmap("cmr.rainforest", 0.30, 0.85)
        ffig.show_colorscale(cmap=cmap, vmin=vmin, vmax=vmax)
        if rms > 0 and has_finite:
            levels = 5.0 * rms * np.sqrt(3.0) ** np.arange(1, 25)
            # contours are drawn on the (possibly block-averaged) preview data
            drawable_levels = levels[levels <= np.nanmax(hdu.data)]
            if drawable_levels.size:
                try:
                    ffig.show_contour(
                        hdul,
                        levels=drawable_levels,
                        colors="white",
                        linewidths=0.45,
                    )
                except AttributeError as exc:
                    logger.warning(f"Skipping FITS preview contours: {exc}")
        if "BMAJ" in hdu.header and "BMIN" in hdu.header:
            ffig.add_beam()
            ffig.beam.set_color("white")
            ffig.beam.set_edgecolor("black")
        ffig.axis_labels.set_xtext("RA")
        ffig.axis_labels.set_ytext("Dec")
        ffig.add_colorbar()
        ffig.colorbar.set_axis_label_text(colorbar_label)
        ffig.savefig(str(png_path), dpi=150)
    finally:
        plt.close(fig)


def _read_first_plane(img_path: Path) -> tuple[np.ndarray, fits.Header]:
    """Return a float copy of the first 2D plane of a FITS image, and its header.

    The file is memory-mapped, so only that plane is read from a cube.
    """
    with fits.open(img_path, memmap=True) as hdul:
        hdu = hdul[0]
        header = hdu.header.copy()
        plane = hdu.data
        if plane is None:
            raise ValueError(f"{img_path} has no image data")
        plane = plane.squeeze()
        while plane.ndim > 2:
            plane = plane[0]
        data = np.array(plane, dtype=np.result_type(plane.dtype, np.float32))
        del plane
    return data, header


def _make_2d_preview_hdu(
    data: np.ndarray,
    header: fits.Header,
    bunit: str | None = None,
    max_pixels: int | None = None,
) -> fits.PrimaryHDU:
    """Build an in-memory 2D celestial HDU suitable for APLpy rendering.

    With ``max_pixels``, an image whose longest side exceeds it is
    block-averaged by an integer factor, and the celestial WCS is rescaled so
    each preview pixel sits on the sky position of its block.
    """
    factor = 1
    if max_pixels is not None:
        factor = max(1, int(np.ceil(max(data.shape) / max_pixels)))
    try:
        wcs = WCS(header).celestial
        if factor > 1:
            _downsample_celestial_wcs(wcs, factor)
        preview_header = wcs.to_header()
    except Exception:
        preview_header = fits.Header()
    if factor > 1:
        data = _block_mean(data, factor)
    for key in ("BMAJ", "BMIN", "BPA", "BUNIT", "OBJECT", "TELESCOP", "INSTRUME"):
        if key in header:
            preview_header[key] = header[key]
    if bunit is not None:
        preview_header["BUNIT"] = bunit
    return fits.PrimaryHDU(data=data, header=preview_header)


def _downsample_celestial_wcs(wcs: WCS, factor: int) -> None:
    """Rescale a 2D WCS in place for an image block-averaged by ``factor``.

    Preview pixel p' (1-based) covers full-resolution pixels
    ``factor * (p' - 1) + 1 .. factor * p'``, centred on
    ``factor * (p' - 1) + (factor + 1) / 2``; hence
    ``CRPIX' = (CRPIX - 0.5) / factor + 0.5``.
    """
    if wcs.wcs.has_cd():
        wcs.wcs.cd = wcs.wcs.cd * factor
    else:
        wcs.wcs.cdelt = wcs.wcs.cdelt * factor
    wcs.wcs.crpix = (wcs.wcs.crpix - 0.5) / factor + 0.5
    wcs.wcs.set()


def _block_mean(data: np.ndarray, factor: int) -> np.ndarray:
    """Average ``factor x factor`` blocks of a 2D image, ignoring non-finite pixels.

    The image is padded at its high ends to a multiple of ``factor``; a block
    with no finite pixels becomes NaN.
    """
    ny, nx = data.shape
    pad_y, pad_x = -ny % factor, -nx % factor
    if pad_y or pad_x:
        data = np.pad(data, ((0, pad_y), (0, pad_x)), constant_values=np.nan)
    blocks = data.reshape(
        data.shape[0] // factor, factor, data.shape[1] // factor, factor
    )
    valid = np.isfinite(blocks)
    total = np.where(valid, blocks, 0.0).sum(axis=(1, 3), dtype=np.float64)
    count = valid.sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
    return mean.astype(np.float32)


def _gaussian_fwhm(x: np.ndarray, fwhm: float, amplitude: float = 1.0) -> np.ndarray:
    """Return a Gaussian curve with the given FWHM, evaluated at ``x`` around zero."""
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return amplitude * np.exp(-0.5 * (x / sigma) ** 2)


def write_psf_profile_preview(
    psf_path: Path,
    png_path: Path,
    title: str = "Point spread function",
) -> None:
    """Write a PSF preview with 1D x/y cuts through the peak, to judge gaussianity.

    The 2D panel shows the PSF core; the two line panels are slices through the
    peak along x and y so an asymmetric or non-Gaussian main lobe (a common
    sign of poor UV coverage or excessive weighting) is visible by eye.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    with fits.open(psf_path) as hdul:
        header = hdul[0].header
        data = np.asarray(hdul[0].data, dtype=float).squeeze()
    while data.ndim > 2:
        data = data[0]
    if data.ndim != 2 or data.size == 0:
        raise ValueError(f"{psf_path}: expected a 2D PSF image")

    finite = np.isfinite(data)
    if not finite.any():
        raise ValueError(f"{psf_path}: PSF has no finite pixels")
    data = np.where(finite, data, 0.0)
    ny, nx = data.shape
    py, px = np.unravel_index(np.argmax(data), data.shape)

    pixel_scale_deg = abs(header.get("CDELT1") or header.get("CD1_1") or 0.0)
    bmaj_deg = header.get("BMAJ")
    bmin_deg = header.get("BMIN")
    bpa_deg = header.get("BPA")
    if bmaj_deg and pixel_scale_deg:
        half_window = max(8, int(round(4.0 * bmaj_deg / pixel_scale_deg)))
    else:
        half_window = max(8, min(nx, ny) // 8)
    half_window = min(half_window, px, py, nx - 1 - px, ny - 1 - py)
    if half_window <= 0:
        half_window = min(nx, ny) // 2

    x_slice = slice(px - half_window, px + half_window + 1)
    y_slice = slice(py - half_window, py + half_window + 1)
    x_profile = data[py, x_slice]
    y_profile = data[y_slice, px]

    if pixel_scale_deg:
        unit = "arcsec"
        x_axis = (
            (np.arange(x_slice.start, x_slice.stop) - px) * pixel_scale_deg * 3600.0
        )
        y_axis = (
            (np.arange(y_slice.start, y_slice.stop) - py) * pixel_scale_deg * 3600.0
        )
    else:
        unit = "pixels"
        x_axis = np.arange(x_slice.start, x_slice.stop) - px
        y_axis = np.arange(y_slice.start, y_slice.stop) - py

    peak_value = float(data[py, px])

    # Reference Gaussians at the CLEAN restoring beam's major/minor FWHM, so a
    # non-Gaussian or mismatched main lobe is visible against the nominal beam.
    # These are overlaid on both panels since a cut along x or y generally
    # isn't aligned with the beam's major/minor axes when BPA != 0/90.
    beam_curves: list[tuple[float, str, str]] = []
    x_dense = y_dense = None
    if unit == "arcsec":
        x_dense = np.linspace(x_axis.min(), x_axis.max(), 200)
        y_dense = np.linspace(y_axis.min(), y_axis.max(), 200)
        if bmaj_deg:
            bmaj_arcsec = bmaj_deg * 3600.0
            pa_label = f", PA {bpa_deg:.0f}°" if bpa_deg is not None else ""
            beam_curves.append(
                (bmaj_arcsec, f'BMAJ {bmaj_arcsec:.2f}"{pa_label}', "#f0883e")
            )
        if bmin_deg:
            bmin_arcsec = bmin_deg * 3600.0
            beam_curves.append((bmin_arcsec, f'BMIN {bmin_arcsec:.2f}"', "#8250df"))

    fig, (ax_img, ax_x, ax_y) = plt.subplots(1, 3, figsize=(10.5, 3.2))
    fig.suptitle(title)

    cutout = data[y_slice, x_slice]
    ax_img.imshow(cutout, origin="lower", cmap="viridis")
    ax_img.axhline(half_window, color="white", lw=0.6, ls="--")
    ax_img.axvline(half_window, color="white", lw=0.6, ls="--")
    ax_img.set_title("PSF core")
    ax_img.set_xticks([])
    ax_img.set_yticks([])

    ax_x.plot(x_axis, x_profile, color="#0969da", label="PSF")
    ax_x.axhline(0.0, color="#888", lw=0.5)
    ax_x.set_title("X profile")
    ax_x.set_xlabel(f"offset from peak ({unit})")
    ax_x.set_ylabel("normalized amplitude")

    ax_y.plot(y_axis, y_profile, color="#cf222e", label="PSF")
    ax_y.axhline(0.0, color="#888", lw=0.5)
    ax_y.set_title("Y profile")
    ax_y.set_xlabel(f"offset from peak ({unit})")

    for fwhm, label, color in beam_curves:
        ax_x.plot(
            x_dense,
            _gaussian_fwhm(x_dense, fwhm, peak_value),
            color=color,
            ls="--",
            lw=1.1,
            label=label,
        )
        ax_y.plot(
            y_dense,
            _gaussian_fwhm(y_dense, fwhm, peak_value),
            color=color,
            ls="--",
            lw=1.1,
            label=label,
        )
    if beam_curves:
        ax_x.legend(fontsize=6.5, frameon=False, loc="upper right")
        ax_y.legend(fontsize=6.5, frameon=False, loc="upper right")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(str(png_path), dpi=140)
    plt.close(fig)


def write_sky_model_previews(
    sky_model,
    center: SkyCoord,
    fov: u.Quantity,
    work_dir: Path,
    run_id: str,
) -> list[tuple[str, str]]:
    """Write full and FoV sky-model source previews."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib.colors import LogNorm

    sources = sky_model.to_json()
    if not sources:
        return []

    ra = np.asarray([src["ra"] for src in sources], dtype=float)
    dec = np.asarray([src["dec"] for src in sources], dtype=float)
    flux = np.asarray([src["I"] for src in sources], dtype=float)
    major_axis = np.asarray(
        [src.get("major_axis", 0.0) or 0.0 for src in sources],
        dtype=float,
    )
    minor_axis = np.asarray(
        [src.get("minor_axis", 0.0) or 0.0 for src in sources],
        dtype=float,
    )
    position_angle = np.asarray(
        [src.get("pa", 0.0) or 0.0 for src in sources], dtype=float
    )
    positive_flux = flux[flux > 0]
    norm = None
    if positive_flux.size:
        norm = LogNorm(
            vmin=float(np.nanmin(positive_flux)), vmax=float(np.nanmax(positive_flux))
        )

    full_name = f"{run_id}_sky_model.png"
    fov_name = f"{run_id}_sky_model_fov.png"
    _plot_sky_model_sources(
        work_dir / full_name,
        ra,
        dec,
        flux,
        major_axis,
        minor_axis,
        position_angle,
        norm,
        title=f"Sky model ({len(sources)} sources)",
    )
    half_fov = fov.to(u.deg).value / 2.0
    _plot_sky_model_sources(
        work_dir / fov_name,
        ra,
        dec,
        flux,
        major_axis,
        minor_axis,
        position_angle,
        norm,
        title=f"Sky model FoV ({fov.to(u.deg).value:.2f} deg)",
        xlim=(center.ra.deg + half_fov, center.ra.deg - half_fov),
        ylim=(center.dec.deg - half_fov, center.dec.deg + half_fov),
        fov_circle=(center.ra.deg, center.dec.deg, half_fov),
    )
    return [(full_name, "sky_model"), (fov_name, "sky_model_fov")]


def _plot_sky_model_sources(
    png_path: Path,
    ra: np.ndarray,
    dec: np.ndarray,
    flux: np.ndarray,
    major_axis: np.ndarray,
    minor_axis: np.ndarray,
    position_angle: np.ndarray,
    norm,
    title: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    fov_circle: tuple[float, float, float] | None = None,
) -> None:
    """Plot source positions as ellipses with astronomical RA orientation."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import PatchCollection

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")
    ax.set_title(title)
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.grid(True, color="0.85", linestyle=":", linewidth=0.8)
    if xlim is not None:
        ax.set_xlim(*xlim)
        plot_width_deg = abs(xlim[1] - xlim[0])
    else:
        ra_min, ra_max = _padded_limits(ra)
        ax.set_xlim(ra_max, ra_min)
        plot_width_deg = ra_max - ra_min
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(*_padded_limits(dec))
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_box_aspect(1)

    compact = _compact_source_mask(major_axis, plot_width_deg)
    resolved = ~compact
    if np.any(resolved):
        ellipses = _sky_model_ellipses(
            ra[resolved],
            dec[resolved],
            major_axis[resolved],
            minor_axis[resolved],
            position_angle[resolved],
        )
        ellipse_collection = PatchCollection(
            ellipses,
            cmap=SKY_MODEL_CMAP,
            norm=norm,
            alpha=0.82,
            edgecolor="black",
            linewidth=0.35,
        )
        ellipse_collection.set_array(flux[resolved])
        ax.add_collection(ellipse_collection)
    if np.any(compact):
        ax.scatter(
            ra[compact],
            dec[compact],
            s=_flux_marker_sizes(flux[compact]),
            c=flux[compact],
            cmap=SKY_MODEL_CMAP,
            norm=norm,
            marker="+",
            linewidths=1.2,
            alpha=0.9,
        )
    if fov_circle is not None:
        from matplotlib.patches import Circle

        ax.add_patch(
            Circle(
                (fov_circle[0], fov_circle[1]),
                fov_circle[2],
                fill=False,
                color="tab:red",
                linestyle="--",
                linewidth=1.2,
            )
        )
    scalar = ScalarMappable(norm=norm, cmap=SKY_MODEL_CMAP)
    scalar.set_array(flux)
    cbar = fig.colorbar(scalar, ax=ax)
    cbar.set_label("Stokes I (Jy)")
    fig.tight_layout()
    fig.savefig(png_path, dpi=140)
    plt.close(fig)


def _padded_limits(
    values: np.ndarray, pad_fraction: float = 0.05
) -> tuple[float, float]:
    """Return finite min/max limits with a small visual padding."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (0.0, 1.0)
    lower = float(np.nanmin(finite))
    upper = float(np.nanmax(finite))
    span = upper - lower
    if span <= 0:
        span = max(abs(lower) * 0.01, 1.0 / 3600.0)
    pad = span * pad_fraction
    return (lower - pad, upper + pad)


def _compact_source_mask(
    major_axis_arcsec: np.ndarray,
    plot_width_deg: float,
) -> np.ndarray:
    """Return sources too small to read as ellipses at the plotted FoV."""
    threshold_arcsec = max(3.0, abs(plot_width_deg) * 3600.0 * 0.01)
    return major_axis_arcsec < threshold_arcsec


def _flux_marker_sizes(flux: np.ndarray) -> np.ndarray:
    """Map source flux densities to visible cross marker areas."""
    positive = flux[np.isfinite(flux) & (flux > 0)]
    if positive.size == 0:
        return np.full(flux.shape, 45.0)
    lo = float(np.nanmin(positive))
    hi = float(np.nanmax(positive))
    safe_flux = np.clip(flux, lo, hi)
    if hi <= lo:
        scaled = np.ones_like(safe_flux)
    else:
        scaled = (np.log10(safe_flux) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
    return 35.0 + scaled * 140.0


def _sky_model_position_angle(pa_deg: float) -> float:
    """Convert astronomical PA east of north to Matplotlib angle from +x."""
    return 90.0 - pa_deg


def _sky_model_ellipses(
    ra: np.ndarray,
    dec: np.ndarray,
    major_axis_arcsec: np.ndarray,
    minor_axis_arcsec: np.ndarray,
    position_angle_deg: np.ndarray,
) -> list:
    """Convert source shape metadata to Matplotlib ellipses in degree units."""
    from matplotlib.patches import Ellipse

    ellipses = []
    fallback_arcsec = 8.0
    for ra_deg, dec_deg, major, minor, pa in zip(
        ra,
        dec,
        major_axis_arcsec,
        minor_axis_arcsec,
        position_angle_deg,
    ):
        major = float(major) if np.isfinite(major) and major > 0 else fallback_arcsec
        minor = float(minor) if np.isfinite(minor) and minor > 0 else major
        ellipses.append(
            Ellipse(
                (float(ra_deg), float(dec_deg)),
                width=major / 3600.0,
                height=minor / 3600.0,
                angle=_sky_model_position_angle(float(pa)) if np.isfinite(pa) else 90.0,
            )
        )
    return ellipses
