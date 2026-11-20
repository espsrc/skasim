User Guide
==========

This guide explains the main configuration language for a ``skasim`` run.

The Configuration Model
-----------------------

Every simulation is defined by a single :class:`~skasim.config.SimConfig` object.
It groups the telescope, observation, imaging, sky-model, noise, UV-coverage,
and output settings in one validated document.

.. code-block:: json

   {
     "telescope": "SKA1MID",
     "telescope_version": "AA4",
     "observation": {
       "frequency_mhz": 1300,
       "bandwidth_mhz": 770,
       "n_channels": 64,
       "observation_time_s": 600,
       "phase_center_ra_deg": 0.0,
       "phase_center_dec_deg": -30.0
     },
     "imaging": [
       {"tag": "dirty", "imager": "oskar-dirty", "pixels": 1024},
       {
         "tag": "wsclean",
         "imager": "wsclean",
         "pixels": 1024,
         "clean_iterations": 10000,
         "robust": 0.0,
         "multiscale": true
       }
     ],
     "models": [
       {"type": "component_sky_model", "catalog": "MIGHTEE"}
     ],
     "output_dir": "my_run",
     "overwrite": true,
     "uv_coverage": true
   }

The next sections describe each block.

``SimConfig`` (top-level)
~~~~~~~~~~~~~~~~~~~~~~~~~

Top-level run settings:

- ``telescope`` (str): Telescope name, e.g. ``SKA1MID``, ``MeerKAT``, ``SKA1LOW``.
- ``telescope_version`` (Optional[str]): Array-assembly version for telescopes that
  require one (e.g. ``AA0.5``, ``AA1``, ``AA4``, ``AA*`` for SKA1-MID). Use
  ``skasim --show-telescopes`` to list accepted versions.
- ``observation`` (:class:`~skasim.config.ObsConfig`): Observing setup.
- ``imaging`` (list of :class:`~skasim.config.ImgConfig`): One or more imaging
  passes, each with a unique ``tag``.
- ``models`` (list of :class:`~skasim.config.ModelEntry`): Typed sky-model entries.
- ``sky_file`` / ``catalog`` / ``fits_image`` (legacy): Single sky-model source
  used when ``models`` is empty.
- ``source_flux_jy`` / ``stokes_q_jy`` / ``stokes_u_jy`` / ``stokes_v_jy``:
  Generated point-source mode (used when no explicit sky model is supplied).
- ``center`` (Optional[str]): Field centre as a sexagesimal string
  (e.g. ``"10h01m35.1s 2d41m41s"``). Overrides the sky-model-derived centre.
- ``rms`` / ``rms_value`` / ``rms_sigma`` (legacy noise toggle): Simple noise flag
  and level in Jy.
- ``noise_rms_start`` / ``noise_rms_end`` (Optional[float]): OSKAR station-noise
  override in Jy. When set, the simulation uses a noise ``Range`` with these
  values instead of the telescope model noise.
- ``noise_injection`` (:class:`~skasim.config.NoiseInjectionConfig`,
  experimental): Per-baseline thermal noise from per-antenna SEFDs, added after
  the simulation. See :ref:`sefd-noise-injection`.
- ``uv_coverage`` (bool): Whether to run ``shadeMS`` and produce a UV-coverage plot.
- ``shadems_command`` (str): Command used for ``shadeMS`` (default: ``shadems``).
- ``uv_coverage_canvas_size`` (int): Square canvas size for the UV-coverage plot.
- ``output_dir`` (Optional[str]): Exact output directory. Defaults to
  ``YYYYMMDD_HHMMSS_<telescope>``.
- ``overwrite`` (bool): Remove an existing output directory before running.
- ``title`` / ``description`` (Optional[str]): Human metadata shown in the weblog
  and manifest (config-file only, not CLI flags).
- ``sky_format`` / ``column_mapping`` / ``flux_scale``: Options shared with the
  legacy component-sky-model path.

``ObsConfig``
~~~~~~~~~~~~~

Observation parameters passed to Karabo:

- ``frequency_mhz`` (float, default 700.0): Central observing frequency.
- ``bandwidth_mhz`` / ``n_channels`` / ``channel_width_mhz`` (Optional): Spectral
  grid. If all three are omitted, defaults of 100 MHz, 8 channels, and 12.5 MHz
  are applied. If two are provided, the third is derived. If all three are
  provided, they must be consistent.
- ``observation_time_s`` (int, default 600): Total observing duration in seconds.
- ``phase_center_ra_deg`` / ``phase_center_dec_deg`` (Optional[float]): Phase
  centre in degrees. When omitted, the sky model centre is used.
- ``start_time`` (Optional[datetime]): Observation start time. Defaults to the
  source culmination time computed by the pipeline.

``ImgConfig``
~~~~~~~~~~~~~

Imaging parameters for one pass:

- ``tag`` (str, default ``"default"``): Unique label for the pass. Used as the
  output subdirectory name and in the weblog.
- ``imager`` (Literal): ``oskar-dirty`` or ``wsclean``.
- ``pixels`` (int, default 512): Image dimensions ``pixels × pixels`` (must be
  ≥ 64).
- ``fov_deg`` (Optional[float]): Field of view in degrees. If omitted, the
  pipeline uses the diffraction-limited FoV for the telescope and frequency.
- ``robust`` (float, default 0.0): Briggs robustness parameter for WSClean.
- ``wsclean_command`` (str, default ``"wsclean"``): Command prefix for WSClean.
  This can include extra native flags such as ``"wsclean -pol V"``.
- ``clean_iterations`` (int, default 5000): CLEAN iteration limit for WSClean.

WSClean-only flags (ignored for ``oskar-dirty``):

- ``mgain`` (Optional[float]): Major-cycle gain. Default effective value 0.8.
- ``multiscale`` (Optional[bool]): Enable multiscale CLEAN. ``None`` and ``True``
  both enable it; ``False`` disables it.
- ``multiscale_scales`` (Optional[List[int]]): Pixel scales for multiscale CLEAN.
- ``auto_threshold`` (Optional[float]): Auto threshold sigma. Default 0.3.
- ``auto_mask`` (Optional[float]): Auto mask sigma. Default 3.0.
- ``local_rms`` (Optional[bool]): Enable ``-local-rms``. Default off.
- ``join_channels`` (Optional[bool]): Enable ``-join-channels``. Default on.
- ``channels_out`` (Optional[int]): Number of WSClean output channels. Defaults
  to the observation channel count (1 in explicit ``continuum`` mode).
- ``spectral_mode`` (``"auto"``, ``"cube"`` or ``"continuum"``, default
  ``"auto"``): ``cube`` images every channel separately (no channel joining,
  no multiscale) and adds moment-8 maps; ``continuum`` images all channels
  combined (``channels_out=1`` unless set). ``auto`` uses ``cube`` when the run
  has a ``spectral_cube`` model and the block uses WSClean, otherwise
  ``continuum``. ``cube`` requires ``imager: "wsclean"``; the OSKAR dirty
  imager always combines channels.
- ``padding`` (Optional[float]): WSClean padding factor.
- ``threads`` (Optional[int]): WSClean parallel thread count (``-j``).

Sky Ingestion Models
--------------------

Version 0.2 introduced a typed ``models`` API. A run can include one
``component_sky_model`` plus any number of image-model entries; see
:ref:`combining-models` for how the components are added together.

Component Models
~~~~~~~~~~~~~~~~

**Type**: ``component_sky_model``

Point-source / Gaussian component catalogs. Exactly one of ``path`` or
``catalog`` must be provided.

- ``path``: Path to a FITS table, JSON array/JSONL, pickle, or Karabo model file.
- ``catalog``: Named built-in catalog (``MIGHTEE``, ``GLEAM``, ``SKAMid``).
- ``sky_format`` (Literal): ``auto``, ``fits``, ``json``, ``pickle``, ``random``.
- ``column_mapping`` (str): Comma-separated FITS column indices for
  ``id,ra,dec,I,Q,U,V,spec_index,ref_freq,rm,major,minor,pa``.
  Default: ``"0,1,2,3,4,5,6,7,8,9,10,11,12"``.
- ``flux_scale`` (float): Global multiplier applied to Stokes I fluxes.

Example (JSON config):

.. code-block:: json

   "models": [
     {
       "type": "component_sky_model",
       "catalog": "MIGHTEE"
     }
   ]

Image Models
~~~~~~~~~~~~

Image-based models are predicted into the ``MODEL_DATA`` column with
``wsclean -predict`` (the default) or, for the deprecated backends, CASA
``ft``. Each entry is added into ``DATA`` straight after its prediction.

**1. Continuum I + Alpha** (``continuum_i_alpha``)

Injects a spatially extended source with a varying spectral index.

- ``stokes_i``: FITS image in Jy/pixel.
- ``alpha``: FITS spectral-index map on the same grid.
- ``reference_frequency_hz``: Frequency at which ``stokes_i`` is defined.

Semantics: :math:`I(\nu) = I_0 \cdot (\nu/\nu_0)^\alpha`.

**2. CASA Taylor Terms** (``casa_taylor_terms``)

Direct usage of existing CASA Taylor-term images (e.g. from a previous WSClean
run).

- ``tt0``: Path to the CASA image table for Taylor term 0.
- ``tt1``: (Optional) Path to Taylor term 1.
- ``reference_frequency_hz``: Taylor expansion reference frequency.

**3. Static Stokes Maps** (``static_stokes_maps``)

Injects a spectrally flat, spatially extended Stokes-I model through
``wsclean -predict``. ``skasim`` creates one FITS model plane per observation
channel, using the observation spectral grid, then WSClean writes the predicted
visibilities to ``MODEL_DATA``.

- ``stokes_i``: Required 2D spatial FITS image in Jy/pixel. It is copied without
  flux scaling into every observation channel.
- The FITS header must provide valid ``CDELT1`` and ``CDELT2`` pixel scales.
- In pipeline runs the image is placed by its FITS WCS (see
  :ref:`combining-models`); non-square images and non-square pixels are
  regridded onto a square WSClean grid.
- The model frequency reference is the observation channel grid; there is no
  model-level reference-frequency parameter for this spectrally flat type.

Example (JSON config):

.. code-block:: json

   "models": [
     {
       "type": "static_stokes_maps",
       "stokes_i": "/path/to/model_i.fits"
     }
   ]

.. warning::
   The schema also accepts ``stokes_q``, ``stokes_u`` and ``stokes_v`` paths,
   but this implementation injects **Stokes I only**. Q/U/V planes are not yet
   validated or predicted; do not rely on them for polarized simulations.

.. _combining-models:

Combining Models (Multimodal Runs)
----------------------------------

A run may combine one ``component_sky_model`` with any number of
``continuum_i_alpha``, ``static_stokes_maps`` and ``spectral_cube`` entries,
e.g. a background catalogue plus one or more extended targets, or an HI line
source in a field of continuum sources:

.. code-block:: json

   "models": [
     {"type": "component_sky_model", "catalog": "MIGHTEE"},
     {"type": "continuum_i_alpha", "stokes_i": "target_i.fits",
      "alpha": "target_alpha.fits", "reference_frequency_hz": 1.4e9},
     {"type": "static_stokes_maps", "stokes_i": "halo_i.fits"}
   ]

``configs/fits_model/vla_c_catalog_plus_two_images_quick.json`` (two
continuum images) and ``configs/fits_model/vla_c_catalog_image_hi_cube_quick.json``
(continuum image plus HI cube) are small worked examples. Generate their models
with ``scripts/make_multimodal_demo_models.py`` and
``scripts/make_multimodal_cube_demo_models.py``, and check a finished run with
``scripts/check_multimodal_run.py``.

How the components are added
   OSKAR simulates the catalogue into ``DATA``. Each image model is then
   predicted into ``MODEL_DATA`` and added into ``DATA`` before the next one
   is predicted, because WSClean (and CASA ``ft`` without ``incremental``)
   overwrite ``MODEL_DATA``. After the run ``DATA`` holds every component and
   ``MODEL_DATA`` holds only the **last** image model's prediction.

Primary beam
   OSKAR attenuates catalogue sources with a Gaussian station beam whose field
   FWHM is the simulation field of view at the band centre, scaling as
   :math:`\nu_{\rm ref}/\nu`. Visibilities see the power pattern (product of
   two station beams), whose FWHM is about the field FWHM divided by
   :math:`\sqrt{2}`. ``skasim`` multiplies every per-channel image-model plane
   by the same power pattern, centred on the phase centre, so catalogue and
   image components share one flux scale. Image models are therefore
   expected to be intrinsic (primary-beam corrected) skies. This also applies
   to image-only runs.

Placement on the sky
   WSClean predict ignores a model's ``CRVAL`` and assumes a SIN projection
   tangent at the phase centre. ``skasim`` places each image at the position
   given by its own FITS WCS: when the image already matches WSClean's grid to
   within 0.1 pixel it is passed with ``-shift`` (the world position of its
   centre pixel); otherwise (other projections, a tangent point away from the
   phase centre, rotated or mirrored axes, non-square pixels or images) it is
   regridded with ``reproject_exact`` onto a WSClean-native grid tangent at the
   phase centre, conserving flux. Predict grids are at least 32 pixels wide
   (WSClean crashes or hangs on smaller ones), so small models and compact
   cube emission are zero-padded by regridding. The record of each entry
   states which path was taken, the shift, and the flux before and after
   regridding.

Phase centre
   The phase centre is ``center`` if given, otherwise the catalogue centre,
   otherwise the centre of the first image model (spectral cubes included) in
   ``models`` order. The deciding rule (and, for ``image_model``, the deciding
   entry) is recorded,
   and a warning is raised when an image model reaches beyond the beam
   half-power radius or the simulation field.

Restrictions
   The deprecated CASA ``ft`` backends (``casa_taylor_terms`` and
   ``continuum_i_alpha`` with ``injection_backend: "casa_ft"``) receive
   neither the shared beam nor WCS placement, so they are only accepted as the
   sole model entry. A run takes at most one ``component_sky_model``.

Spectral cubes
   Cubes follow the same placement and beam rules as images, channel by
   channel: each channel is attenuated by the power beam at its own frequency.
   When a cube needs regridding, only channels with emission are regridded,
   cropped to the region holding the emission (identical result, much faster
   for a line source). The cube's size is independent of the imaging blocks.
   ``DATA`` then holds continuum and line together; ``skasim`` does not
   subtract the continuum. To get both a line cube and a continuum image, use
   two imaging blocks:

   .. code-block:: json

      "imaging": [
        {"tag": "cont", "imager": "oskar-dirty", "spectral_mode": "continuum",
         "pixels": 512, "cell_size_arcsec": 3.0},
        {"tag": "line", "imager": "wsclean", "spectral_mode": "cube",
         "pixels": 512, "cell_size_arcsec": 3.0}
      ]

   A run that combines cubes with continuum models but has no continuum block
   records a note. Each cube's line channels (MS channel indices and
   frequencies) and integrated line flux (Jy km/s when the cube has
   ``RESTFRQ``, otherwise Jy MHz) are recorded, so continuum subtraction can
   be done downstream. All components share the observation's single channel
   grid; a warning is recorded when MS channels are wider than a cube's
   channels (the line is smoothed, flux is conserved). Cube frequencies must be
   a ``FREQ`` axis and are used as observed frequencies; ``SPECSYS`` is
   recorded but not converted (velocity axes and frame conversion:
   `issue #50 <https://github.com/espsrc/skasim/issues/50>`_).

WSClean predict settings
   Threads and the WSClean executable for prediction come from the first
   ``wsclean`` imaging block (or the first imaging block when none uses
   WSClean); ``wsclean_predict_command`` overrides the executable.

Spectral Reference Policy
-------------------------

When using image models, ``skasim`` automatically adjusts the model's spectral
reference to the centre of the observed band.

- **Idempotency**: The original model files are never modified. The pipeline
  works on local copies in the ``work_dir``.
- **Correction**: If a spectral index is provided (via ``alpha`` map or ``tt1``),
  pixel values are scaled using the power-law convention:
  :math:`tt0' = tt0 \cdot (\nu_{new} / \nu_{old})^\alpha`.
- **Metadata**: The single-channel spectral coordinate (CRVAL4) of the local
  CASA images is updated to the observation's centre frequency.

Generated Source Fluxes (Legacy Mode)
--------------------------------------

If no typed models are provided, ``skasim`` can generate point sources using
legacy CLI flags::

    skasim --flux-density 1.0 0.5 --observation-time 120

.. warning::
   Generated source flags are ignored if the ``models`` list is used.

.. _sefd-noise-injection:

SEFD-Based Noise Injection (Experimental)
-----------------------------------------

.. warning::
   SEFD-based noise injection is a temporary, experimental feature. It will be
   moved to Karabo and is not meant to stay in ``skasim``. Expect the
   ``noise_injection`` block and the SEFD tables in ``data/sefd/`` to change or
   be removed once Karabo provides this feature.

Instead of OSKAR's station-noise model (``rms`` / ``noise_rms_start``), the
pipeline can simulate noiselessly and then add thermal noise to the ``DATA``
column of the measurement set. The noise level of each baseline follows from
the SEFDs of its two antennas through the radiometer equation (Radcliffe et
al. 2024, MNRAS 527, 942):

.. math::

   \sigma_{pq} = \frac{1}{\eta_c}
   \sqrt{\frac{\mathrm{SEFD}_p\,\mathrm{SEFD}_q}{2\,\Delta t\,\Delta\nu}}

where :math:`\Delta t` is the visibility integration time (read per row from
the MS ``INTERVAL`` column) and :math:`\Delta\nu` is
``observation.channel_width_mhz``. :math:`\sigma_{pq}` is the noise in the real
*or* imaginary part of one visibility: each part gets independent Gaussian
noise with standard deviation :math:`\sigma_{pq}`, in every channel and
correlation.

For a homogeneous array of :math:`N` antennas, total bandwidth :math:`B`,
observing time :math:`t` and :math:`n_\mathrm{pol}` polarizations, the
natural-weight image noise is then

.. math::

   \sigma_\mathrm{image} = \frac{\mathrm{SEFD}}
   {\eta_c\sqrt{n_\mathrm{pol}\,N(N-1)\,B\,t}}

Noise is injected after all sky models have been predicted into the
visibilities and before imaging. The ``WEIGHT`` and ``SIGMA`` columns, and
``WEIGHT_SPECTRUM`` if present, are rewritten as :math:`1/\sigma_{pq}^2` and
:math:`\sigma_{pq}`, so imagers that use the MS weights (such as WSClean)
weight each baseline by its real noise level. The noise is not seeded, so
every run draws a different realization. The
``noise_injection_completed`` milestone in the run manifest records the
minimum, maximum and mean :math:`\sigma_{pq}`.

``noise_injection`` block:

- ``enabled`` (bool, default ``false``): Turn noise injection on. It cannot be
  combined with OSKAR noise (``rms`` or ``noise_rms_start``).
- ``sefd_file`` (Optional[str]): Path to an SEFD table. Required when
  ``enabled`` is ``true``. Relative paths are resolved from the directory
  ``skasim`` runs in.
- ``eta_c`` (float, default 0.88): System (correlator) efficiency
  :math:`\eta_c`.

.. code-block:: json

   {
     "telescope": "SKA1MID",
     "telescope_version": "AA*",
     "observation": {
       "frequency_mhz": 1400,
       "bandwidth_mhz": 100,
       "n_channels": 100,
       "observation_time_s": 3600
     },
     "imaging": [{"tag": "wsclean", "imager": "wsclean", "pixels": 1024}],
     "models": [{"type": "component_sky_model", "catalog": "MIGHTEE"}],
     "noise_injection": {
       "enabled": true,
       "sefd_file": "data/sefd/ska-mid-aastar-band2-corrected-sefd.json",
       "eta_c": 0.88
     }
   }

**SEFD file format.** A JSON object with an ``antennas`` list. Each entry has a
``name`` and a positive ``sefd_jy``:

.. code-block:: json

   {
     "antennas": [
       {"name": "Defford", "sefd_jy": 350},
       {"name": "Cambridge", "sefd_jy": 175}
     ]
   }

The list is matched to the measurement set **by position**: entry *i* is
antenna index *i*, so the order must match the station order of the
telescope model. Names are not checked against the MS ``ANTENNA`` table. The
run fails if the MS refers to more antennas than the file lists.

**Bundled tables** (in ``data/sefd/`` of the repository; the Docker image does
not include them, so mount the file into the container):

- ``ska-mid-aastar-band1-sefd.json``, ``ska-mid-aastar-band2-corrected-sefd.json``,
  ``ska-mid-aastar-band5a-sefd.json``, ``ska-mid-aastar-band5b-sefd.json``:
  SKA-Mid AA*, 144 dishes.
- ``ska-low-aastar-200mhz-300mhzbw-sefd.json``: SKA-Low AA*, 307 stations.
- ``meerkat-sefd-Lband.json``: MeerKAT L-band, 64 dishes.
- ``vla-sefd-Lband.json``: VLA L-band, 27 antennas.
- ``emerlin-sefd.json``: e-MERLIN, 7 antennas.

.. note::
   In the SKA-Mid band 5a/5b tables, the 64 MeerKAT dishes, which have no
   band-5 receivers, have an SEFD of :math:`10^{12}` Jy. Their baselines get
   near-zero weight but very large noise in ``DATA``, so image these runs with
   an imager that uses the MS weights, such as WSClean.

Multi-Imaging Runs
------------------

``imaging`` is a list, so several passes can be produced from a single
visibility set::

    "imaging": [
      {"tag": "dirty", "imager": "oskar-dirty"},
      {"tag": "clean", "imager": "wsclean", "clean_iterations": 1000}
    ]

The CLI ``--imager`` flag overrides only the first entry; for full control over
multiple passes, use a JSON config file.

WSClean Command
---------------

The WSClean command defaults to ``wsclean``. For containerized environments,
pass the command explicitly::

    skasim --imager wsclean \
      --wsclean-command "singularity exec wsclean-3.10.sif wsclean"

The ``wsclean_command`` field also accepts native WSClean flags that do not
have a first-class ``ImgConfig`` field, for example polarization selection::

    {"wsclean_command": "wsclean -pol V"}

Run Records
-----------

Each run writes a **Manifest** and an **Always-on Weblog**.

.. note::
   See the :doc:`outputs` section for detailed information on run records.
