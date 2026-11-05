"""Public façade for image-model loading, validation and injection."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from loguru import logger

from ...config import (
    CasaTaylorTermsModelEntry,
    ContinuumIAlphaModelEntry,
    ModelEntry,
    SpectralCubeModelEntry,
    StaticStokesMapsModelEntry,
    has_spectral_cube_model,
    spectral_cube_model_entries,
)
from ...config import (
    ComponentSkyModelEntry as ComponentSkyModelEntry,
)
from ...config import (
    ImgConfig as ImgConfig,
)
from ...config import (
    SimConfig as SimConfig,
)
from ...manifest import RunContext
from .casa_interop import (
    CasaModelProduct,
    adjust_spectral_reference,
    merge_model_data_into_data,
    prepare_casa_taylor_terms,
    prepare_continuum_i_alpha_for_casa,
    prepare_spectral_cube_for_casa,
    require_casa_executable,
    run_casa_exportfits,
    run_casa_ft,
    run_casa_importfits,
    run_casa_set_spectral_coordinate,
    validate_casa_taylor_terms,
)
from .casa_interop import (
    _resample_spectral_axis_to_ms_channels as _resample_spectral_axis_to_ms_channels,
)
from .casa_interop import (
    _set_crval4_via_script as _set_crval4_via_script,
)
from .fits_io import (
    FitsCubeInfo,
    FitsImageInfo,
    _select_wsclean_img_config,
    component_model_entries,
    image_model_center,
    image_model_entries,
    image_model_footprints,
    primary_model_fits_path,
    read_fits_cube_info,
    read_fits_image_info,
    validate_continuum_i_alpha,
    validate_spectral_cube,
    validate_static_stokes_maps,
)
from .previews import (
    run_moment8_for_spectral_cube,
    write_image_model_previews,
    write_sky_composition_preview,
    write_spectral_cube_input_preview,
)

if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord

    from ...primary_beam import GaussianStationBeam

# avoid circular imports: wsclean_predict is imported locally inside
# inject_image_models when a wsclean_predict backend is selected.


def inject_image_models(
    ctx: RunContext,
    visibility_path: Path,
    imaging_configs: Optional[List[ImgConfig]] = None,
    phase_center: Optional["SkyCoord"] = None,
    beam: Optional["GaussianStationBeam"] = None,
) -> None:
    """Inject configured image models into an existing Measurement Set.

    ``beam`` and ``phase_center`` describe the shared Gaussian primary beam the
    pipeline also applies to catalogue sources; WSClean-predicted entries are
    attenuated by it. Each entry is merged into ``DATA`` straight after its
    prediction because both backends overwrite ``MODEL_DATA``.
    """
    beam_kwargs = {"phase_center": phase_center, "beam": beam}
    entries = image_model_entries(ctx.config)
    if not entries:
        return
    effective_imaging = (
        imaging_configs if imaging_configs is not None else ctx.config.imaging
    )

    backends = {"casa_ft", "wsclean_predict"}
    ctx.add_milestone(
        "image_injection_started",
        "started",
        details={"n_model_entries": len(entries), "backends": sorted(backends)},
    )

    for index, entry in enumerate(entries):
        if isinstance(entry, StaticStokesMapsModelEntry):
            img_config = _select_wsclean_img_config(effective_imaging)
            report = validate_static_stokes_maps(
                entry, ctx.config.observation, img_config
            )
            from ..wsclean_predict import inject_static_stokes_i_with_wsclean_predict

            report_predict = inject_static_stokes_i_with_wsclean_predict(
                ctx,
                entry,
                index,
                visibility_path,
                img_config,
                **beam_kwargs,
            )
            _record_injection(
                ctx,
                index,
                entry,
                backend="wsclean_predict",
                report={**report, **report_predict},
                model_paths=[Path(p) for p in report_predict.get("model_paths", [])],
            )
        elif isinstance(entry, ContinuumIAlphaModelEntry):
            img_config = _select_wsclean_img_config(effective_imaging)
            report = validate_continuum_i_alpha(entry)
            product = prepare_continuum_i_alpha_for_casa(ctx, entry, index)

            if entry.injection_backend == "casa_ft":
                logger.warning(
                    "continuum_i_alpha injection_backend='casa_ft' is deprecated; "
                    "prefer 'wsclean_predict'."
                )
                run_casa_ft(
                    visibility_path=visibility_path,
                    model_paths=product.model_paths,
                    nterms=product.nterms,
                    reffreq=product.reffreq,
                    incremental=False,
                )
                backend = "casa_ft"
            else:
                from ..wsclean_predict import (
                    inject_continuum_i_alpha_with_wsclean_predict,
                )

                report_predict = inject_continuum_i_alpha_with_wsclean_predict(
                    ctx,
                    entry,
                    index,
                    visibility_path,
                    img_config,
                    product,
                    **beam_kwargs,
                )
                backend = "wsclean_predict"
                report = {**report, **report_predict}
            _record_injection(
                ctx,
                index,
                entry,
                backend=backend,
                report=report,
                model_paths=[Path(p) for p in product.model_paths],
                nterms=product.nterms,
                reffreq=product.reffreq,
            )
        elif isinstance(entry, CasaTaylorTermsModelEntry):
            report = validate_casa_taylor_terms(entry)
            product = prepare_casa_taylor_terms(ctx, entry, index)
            logger.warning(
                "casa_taylor_terms uses the deprecated CASA ft backend; "
                "consider migrating to continuum_i_alpha with wsclean_predict."
            )
            run_casa_ft(
                visibility_path=visibility_path,
                model_paths=product.model_paths,
                nterms=product.nterms,
                reffreq=product.reffreq,
                incremental=False,
            )
            _record_injection(
                ctx,
                index,
                entry,
                backend="casa_ft",
                report=report,
                model_paths=[Path(p) for p in product.model_paths],
                nterms=product.nterms,
                reffreq=product.reffreq,
            )
        elif isinstance(entry, SpectralCubeModelEntry):
            img_config = _select_wsclean_img_config(effective_imaging)
            report = validate_spectral_cube(entry, ctx.config.observation, img_config)
            product = prepare_spectral_cube_for_casa(ctx, entry, index, report)
            from ..wsclean_predict import inject_spectral_cube_with_wsclean_predict

            assert product.cube_data is not None
            assert product.header is not None
            assert product.freq_axis is not None
            report_predict = inject_spectral_cube_with_wsclean_predict(
                ctx,
                entry,
                index,
                visibility_path,
                img_config,
                product.cube_data,
                product.header,
                product.freq_axis,
            )
            _record_injection(
                ctx,
                index,
                entry,
                backend="wsclean_predict",
                report={**report, **report_predict},
                model_paths=[Path(p) for p in product.model_paths],
                nterms=product.nterms,
                reffreq=product.reffreq,
            )
        else:
            continue

        # Both backends overwrite MODEL_DATA, so each entry is added into DATA
        # before the next prediction. This is an in-place MS change: record it
        # as a milestone, not an output.
        merge_model_data_into_data(visibility_path)
        ctx.add_milestone(
            "image_model_merged",
            "completed",
            details={"model_entry_index": index, "model_type": entry.type},
        )

    ctx.add_milestone(
        "image_injection_completed",
        "completed",
        details={
            "visibility_path": str(visibility_path),
            "model_data_merged": True,
            "n_merges": len(entries),
            # MODEL_DATA is left in place and holds only the final entry's
            # prediction; DATA holds the sum of every component.
            "model_data_contents": "last_entry_only",
        },
    )


def _record_injection(
    ctx: RunContext,
    index: int,
    entry: ModelEntry,
    backend: str,
    report: dict,
    model_paths: list[Path],
    nterms: int = 1,
    reffreq: str = "",
) -> None:
    """Record a sky_model output and image_model_injected milestone."""
    ctx.manifest.add_output(
        "sky_model",
        model_paths[0].name if model_paths else report.get("prefix", "model"),
        role="casa_model_image",
        metadata={
            "model_entry_index": index,
            "model_type": entry.type,
            "nterms": nterms,
            "reffreq": reffreq,
            "all_model_paths": [path.name for path in model_paths],
        },
    )
    ctx.add_milestone(
        "image_model_injected",
        "completed",
        details={
            "model_entry_index": index,
            "model_type": entry.type,
            "backend": backend,
            **report,
        },
    )


__all__ = [
    "CasaModelProduct",
    "FitsCubeInfo",
    "FitsImageInfo",
    "adjust_spectral_reference",
    "component_model_entries",
    "has_spectral_cube_model",
    "image_model_center",
    "image_model_entries",
    "image_model_footprints",
    "inject_image_models",
    "merge_model_data_into_data",
    "prepare_casa_taylor_terms",
    "prepare_continuum_i_alpha_for_casa",
    "prepare_spectral_cube_for_casa",
    "primary_model_fits_path",
    "read_fits_cube_info",
    "read_fits_image_info",
    "require_casa_executable",
    "run_casa_exportfits",
    "run_casa_ft",
    "run_casa_importfits",
    "run_casa_set_spectral_coordinate",
    "run_moment8_for_spectral_cube",
    "spectral_cube_model_entries",
    "validate_casa_taylor_terms",
    "validate_continuum_i_alpha",
    "validate_spectral_cube",
    "validate_static_stokes_maps",
    "write_image_model_previews",
    "write_sky_composition_preview",
    "write_spectral_cube_input_preview",
]
