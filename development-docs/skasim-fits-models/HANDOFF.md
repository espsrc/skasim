# skasim FITS / Image-Model Injection Handoff

## Current State

The FITS/image-model implementation is in a usable v1 state.

Implemented model entry types:

- `component_sky_model`
  - Existing catalog/component path.
  - May be combined with image-model entries (see "Multimodal runs" below).

- `continuum_i_alpha`
  - Inputs: Stokes-I FITS image, alpha FITS image, `reference_frequency_hz`.
  - Contract: `I(nu) = I0 * (nu / nu0)^alpha`.
  - Implementation creates CASA Taylor-term images in the run directory:
    - `tt0 = I`
    - `tt1 = I * alpha`
  - `wsclean -predict` (default `injection_backend`) predicts per-channel
    planes into `MODEL_DATA`; CASA `ft` remains as a deprecated backend.
  - `MODEL_DATA` is merged into final `DATA`.
  - FITS preview is shown in the weblog Sky Model section.

- `casa_taylor_terms`
  - Inputs: existing CASA image-table `tt0`, optional `tt1`, `reference_frequency_hz`.
  - Implementation copies CASA image tables into the run directory.
  - The run-local copies have their single-channel spectral coordinate aligned to `reference_frequency_hz`.
  - CASA `ft` predicts into `MODEL_DATA`.
  - The `tt0` term is exported to FITS for the weblog model preview.

- `static_stokes_maps`
  - Stokes I is injected through `wsclean -predict` (spectrally flat).
  - Q/U/V paths are accepted by the schema but not injected.

CASA execution:

- If `casatasks` is importable, in-process CASA tasks are used.
- Otherwise, the installed `casa` executable is used in batch mode.
- Batch logs are written as `skasim_casa_*.log` in the run directory.

Demo configs:

- Quick FITS demos live in `configs/fits_model/`.
- Long CASA Taylor-term demos live in `configs/fits_model/casa_taylor_terms/`.
- Long demo runner:
  - `configs/fits_model/run_casa_taylor_model_demos.sh`

Important note about the original FITS demos:

- They use `CY4223_L_004_20180322_avg_target_3C320_vla_loop_01-0003-model.fits`.
- That is a numbered WSClean per-channel model FITS.
- It is not the `...-MFS-model.fits` product.

## Verified Behavior

Recent verified checks:

- Unit/integration suite passed after the last implementation slice:
  - `229 passed`
- Verified earlier full quick demo configs:
  - VLA C image-only FITS
  - MeerKAT image-only FITS
  - SKA1-Mid image-only FITS
  - VLA C catalog plus FITS image
  - VLA C FITS plus WSClean imaging

Known runtime caveat:

- Existing `demo_output/` products may have been generated before the latest Taylor-term fix.
- Older Taylor-term LOFAR outputs can be blank if CASA logged:
  - `No overlap in frequency between image channels and selected data`
- Rerun those configs with the current code to regenerate outputs with run-local spectral-coordinate alignment.

## Design Decisions Preserved

- `catalog` is the canonical user-facing word. Numeric catalog IDs are removed.
- Typed `models` entries are the canonical configuration shape.
- Model preparation and injection are separate from catalog loading.
- CASA `ft` was the v1 default image-model injection backend; `wsclean -predict`
  is now the default and CASA `ft` is deprecated.
- `MODEL_DATA` is retained and merged into `DATA` after every image entry; it
  holds only the last entry's prediction at the end of the run.

## Multimodal runs

Combining one catalogue with several image models is specified in
`.hermes/PRD-multimodal-input.md`: per-entry merge, a shared OSKAR-equivalent
Gaussian beam applied to image models, WCS placement through WSClean `-shift`
or `reproject_exact` regridding, CASA `ft` entries accepted only alone, and
composition reporting in the manifest and weblog.
- Image-only runs create a base MS and may fall back to a zero-flux placeholder source.
- Weblog should show model provenance and visual previews where possible.

## Open Questions / Caveats

### `I + alpha` versus `tt0/tt1`

Conceptually, `I + alpha` and `tt0/tt1` can encode the same continuum model, but only if the Taylor convention is explicit.

Current state:

- `continuum_i_alpha` has explicit power-law semantics.
- `casa_taylor_terms` treats CASA image tables as prepared Taylor products and uses CASA `ft`.

Potential future improvement:

- Add an explicit semantic mode such as `continuum_taylor_terms`.
- Contract:
  - `tt0 = I0`
  - `tt1 = I0 * alpha`
  - therefore `alpha = tt1 / tt0`
- This would make `tt0/tt1` exchangeable with `I + alpha` when the user declares the convention.
- It should not be silently assumed for every CASA `.tt0/.tt1` product.

### Frequency extrapolation

`continuum_i_alpha` can extrapolate by definition.

Raw CASA image tables do not automatically imply valid extrapolation outside their spectral coordinate. The implementation now aligns single-channel Taylor-term metadata, but the physical validity of large extrapolations remains the user's responsibility.

## Recommended Future Implementations

### 1. Semantic Taylor-Term Continuum Mode

Add `continuum_taylor_terms` or similar.

Purpose:

- Treat `tt0/tt1` as a scientific continuum model, not merely CASA backend files.
- Convert to `I + alpha` semantics internally.
- Allow interpolation/extrapolation by explicit power-law convention.

Likely fields:

```json
{
  "type": "continuum_taylor_terms",
  "tt0": "model.tt0",
  "tt1": "model.tt1",
  "reference_frequency_hz": 1575010361.7577474,
  "convention": "tt1_equals_i_alpha"
}
```

Implementation paths:

- Derive alpha as `tt1 / tt0` with masking around zero-valued `tt0`.
- Reuse the `continuum_i_alpha` backend path.
- Preserve original CASA terms as provenance.

### 2. Static Stokes Map Injection

Complete backend support for `static_stokes_maps`.

Inputs:

- FITS Stokes I/Q/U/V maps.
- A subset may be allowed, but supplied planes must be explicit.

Key questions:

- Whether CASA `ft` should receive separate Stokes images or a multi-Stokes CASA image.
- How to validate consistent WCS and units across all supplied Stokes planes.
- How to represent polarization previews in the weblog.

### 3. Polarization Cubes

Extend static Stokes maps to spectral polarization cubes.

Inputs:

- RA-Dec-frequency cubes for I/Q/U/V.

Use cases:

- Frequency-dependent polarization structure.
- Rotation-measure-like behavior if encoded directly in Q/U cubes.

Likely challenges:

- Cube size and performance.
- Frequency grid compatibility with the simulated MS.
- Preview/reporting beyond a single image plane.

### 4. Spectral Line Cubes

Add `spectral_cube`.

Inputs:

- FITS cube with RA, Dec, frequency/velocity axes.

Use cases:

- HI or molecular line simulations.
- Continuum plus line in the same run.

Design direction:

- Prefer explicit frequency-axis handling.
- Fail clearly if cube channels do not overlap the observation unless an interpolation policy is configured.
- Consider whether to regrid model cubes to the MS spectral grid before CASA `ft`.

### 5. Interpolation / Extrapolation Policy

Introduce explicit spectral behavior controls.

Possible fields:

```json
{
  "spectral_policy": "strict_overlap"
}
```

or:

```json
{
  "spectral_policy": "power_law_extrapolate",
  "allow_extrapolation": true
}
```

Useful policies:

- `strict_overlap`: fail if model and observation do not overlap.
- `nearest`: use nearest model plane.
- `linear_interpolate`: interpolate within model spectral range only.
- `power_law_extrapolate`: allow continuum extrapolation using alpha.
- `regrid_to_ms`: create backend-ready images/cubes on the MS spectral grid.

This is especially important for LOFAR-style runs from GHz-reference models.

### 6. WSClean / DP3 Prediction Backends

CASA `ft` is functional but can be slow.

Future alternatives:

- WSClean predict:
  - useful for some image-model workflows,
  - append/additive semantics need careful handling.

- DP3/WGridderPredict:
  - promising for performance,
  - needs investigation and stable installation assumptions.

Recommendation:

- Keep CASA `ft` as the correctness backend.
- Add alternative backends only behind explicit config and comparison tests.

### 7. Better Base-MS Creation

Current behavior:

- Image-only runs try to create an empty/base MS.
- Runtime may fall back to a zero-flux placeholder source.

Future improvement:

- Implement a cleaner explicit base-MS creation path.
- Record fewer warnings in normal image-only runs.
- Add test coverage for different telescopes and long spectral grids.

### 8. Weblog Improvements

Current state:

- FITS and CASA Taylor-term previews are shown as FITS-derived PNGs.
- Science images are shown when dirty/WSClean products exist.

Future work:

- Add a dedicated model-injection section.
- Show model type, paths, reference frequency, backend, and spectral policy together.
- For cubes, show selected planes or moment maps.
- For polarization, show I/Q/U/V panels or derived polarized intensity and angle.

### 9. Demo Matrix Cleanup

Current long demos are useful but heavy.

Future improvements:

- Split demos into:
  - quick smoke configs,
  - long science-like configs,
  - local-fixture examples that require `models_for_testing/`.
- Add README warnings for local-only model assets.
- Add optional runner flags to run one model pair or one telescope family.

### 10. Validation and Numerical Checks

Current tests mostly verify workflow, metadata, and presence of outputs.

Future tests should add:

- Nonzero dirty-image checks for representative image-model runs.
- Comparison of `continuum_i_alpha` and declared-equivalent `continuum_taylor_terms`.
- Spectral extrapolation sanity checks.
- Additive multi-model visibility checks.
- Failure tests for spectral non-overlap when policy is strict.

## Suggested Next Implementation Order

1. Add semantic `continuum_taylor_terms` mode.
2. Add strict/interpolate/extrapolate spectral policy handling.
3. Complete `static_stokes_maps` backend injection.
4. Add spectral cube support.
5. Add polarization cube / RM-oriented support.
6. Investigate DP3/WGridderPredict as an optional performance backend.
7. Improve weblog model-injection reporting.

## Files to Read First Next Time

- `src/skasim/config.py`
- `src/skasim/image_models.py`
- `src/skasim/pipeline.py`
- `src/skasim/imaging.py`
- `src/skasim/weblog.py`
- `configs/fits_model/README.md`
- `development/skasim-fits-models/skasim_docs/MODEL_INPUT_CONTRACTS.md`
- `development/skasim-fits-models/skasim_docs/PRD.md`
- `development/skasim-fits-models/skasim_docs/VALIDATION_TEST_MATRIX.md`
