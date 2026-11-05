Run Outputs and Records
=======================

Every ``skasim`` execution produces a structured set of outputs designed for 
both automated parsing and human review.

Output Directory Structure
--------------------------

By default, the output is stored in a directory named:
``YYYYMMDD_HHMMSS_<telescope>`` (e.g., ``20260604_143005_SKA1MID``).

Within this directory, you will find:

- ``manifest.json``: The root structured record of the run.
- ``weblog.html``: A self-contained HTML report for visual review.
- ``work_dir/``: Intermediate files, including run-local CASA image copies and MS files.
- ``skasim_casa_*.log``: Detailed logs from CASA batch executions.

The Manifest (manifest.json)
----------------------------

The manifest is the canonical "truth" of a simulation run. It is a JSON file 
containing:

- **Config**: A copy of the ``SimConfig`` used for the run.
- **Milestones**: A chronological list of high-level pipeline events.
- **Outputs**: Paths to all generated products (images, MS, logs).

Milestones
~~~~~~~~~~

Milestones provide an audit trail. For example, the ``adjusted_spectral_reference`` 
milestone records how an input image model was scaled:

.. code-block:: json

   {
     "name": "adjusted_spectral_reference",
     "status": "completed",
     "timestamp": "2026-06-04T12:00:00.000000",
     "details": {
       "model_type": "casa_taylor_terms",
       "old_reference_frequency_hz": 5991892258.0,
       "new_reference_frequency_hz": 1284000000.0,
       "nterms": 2
     }
   }

Multimodal runs record, as milestones:

- ``phase_centre_resolved``: the phase centre and the rule that chose it
  (``explicit``, ``catalogue``, ``image_model``, ``default_field`` or
  ``generated_sources``).
- ``primary_beam_configured``: the shared Gaussian beam (field FWHM,
  reference frequency, power FWHM).
- ``catalogue_beam_flux``: catalogue Stokes I at the band centre with and
  without the beam.
- ``image_model_footprints``: each image model's extent relative to the beam
  half-power radius and the simulation field.
- ``image_model_injected`` (one per image model): backend, WCS placement
  (``-shift`` position, whether the image was regridded, flux before and
  after), and intrinsic and beam-attenuated flux at the band centre.
- ``image_model_merged`` (one per image model): ``MODEL_DATA`` added into
  ``DATA``. These are in-place changes to the Measurement Set, recorded as
  milestones rather than outputs.
- ``image_injection_completed``: number of merges and
  ``model_data_contents: "last_entry_only"``.

The Weblog
----------

The Weblog is a self-contained HTML report generated after each run. 
It is a single file: ``weblog.html`` located in the run's output directory. 
Open it in any modern browser to view the simulation summary.

Key Sections
~~~~~~~~~~~~

1. **Sky Model**:
   - Lists all ingested models.
   - Shows provenance (original file paths).
   - Provides visual previews. For CASA Taylor-term models, the ``tt0`` term is automatically exported to FITS/PNG for display.
   - For runs with image models, a **Sky composition** table (position,
     placement, intrinsic and in-beam flux for every component) and a
     **Combined view** of catalogue sources, image footprints, the beam
     half-power circle and the simulation field around the phase centre.
     
2. **Observation Details**:
   - Spectral grid setup.
   - Phase center and telescope configuration.

3. **Injected Visibility Data**:
   - Identifies the backend used for injection and model provenance.

4. **Imaging Results**:
   - One dedicated section **per imaging tag** for multi-imaging runs.
   - Image previews for each available product (e.g., ``oskar-dirty`` and ``wsclean``).
   - If enabled, UV Coverage plots (produced via ``shadeMS``).

5. **Logs**:
   - Filterable view of the pipeline logs.
   - Access to raw CASA logs (``skasim_casa_*.log`` files).

HTML Report
~~~~~~~~~~~

The ``weblog.html`` file is a single, self-contained HTML file. It does not 
require a server; all CSS and images are embedded in the final report. This 
makes it trivial to archive or share.

Visibility Data (Measurement Set)
---------------------------------

The resulting Measurement Set (``.ms``) is located within the output 
directory. Its ``DATA`` column contains the combined visibilities from all sky
models, ready for further processing with standard radio-interferometry tools.

.. note::
   When image models are injected, ``MODEL_DATA`` is left in the Measurement
   Set but holds **only the last image model's prediction**, not the full sky
   model. Image models are added into ``DATA`` one at a time because each
   prediction overwrites ``MODEL_DATA``. Use ``DATA`` for the simulated sky;
   any later WSClean or CASA deconvolution rewrites ``MODEL_DATA``.
