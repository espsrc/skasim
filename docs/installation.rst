Installation
============

``skasim`` supports three installation paths:

- a **container image** with the complete runtime (Karabo, OSKAR, WSClean,
  CASA, shadeMS) — the quickest way to run simulations;
- a **pip-only** path for lightweight use, including imports, configuration
  validation, CLI help, documentation builds, and lightweight tests;
- a **conda full-runtime** path for executing simulations with Karabo, OSKAR,
  and the supported runtime stack.

Container image (Docker / Apptainer)
------------------------------------

Released versions of ``skasim`` are published as a container image on the
GitHub Container Registry::

    docker pull ghcr.io/espsrc/skasim:0.5.0      # or :latest

The image contains ``skasim`` with the validated runtime stack (Python 3.12,
Karabo 0.36, OSKAR 2.12, WSClean 3.6 with EveryBeam and Dysco, CASA 6.7
``casatools``/``casatasks`` with its runtime data, and shadeMS). It contains
no configurations, sky models or data: mount your working directory and
reference files relative to it. The image's command is ``skasim``, and its
working directory is ``/work``::

    docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work" \
      ghcr.io/espsrc/skasim:0.5.0 --config myconfig.json

``--user`` makes the outputs belong to you rather than to ``root``. Any
command that does not start with ``-`` is run inside the activated
environment instead of ``skasim``, for example ``wsclean --version``,
``python`` or ``bash``::

    docker run --rm -it -v "$PWD:/work" ghcr.io/espsrc/skasim:0.5.0 bash

**Platforms and GPUs.** The image is built for ``linux/amd64`` only; on Apple
Silicon it runs under emulation at best. It ships the CUDA 12 runtime used by
Karabo and OSKAR, and falls back to the CPU when no GPU is visible. It has been
validated on CPU only; GPU execution (``docker run --gpus all ...``, which
needs the NVIDIA Container Toolkit and a driver supporting CUDA 12) is not yet
tested.

**HPC clusters.** Where Docker is not available, Apptainer (Singularity) can
run the same image::

    apptainer pull skasim_0.5.0.sif docker://ghcr.io/espsrc/skasim:0.5.0
    apptainer run --bind "$PWD:/work" --pwd /work skasim_0.5.0.sif --config myconfig.json

Apptainer already runs as your own user, so ``--user`` is not needed.

**Licences.** ``skasim`` is MIT-licensed, but the image also redistributes
GPL-licensed software (WSClean, EveryBeam) and CASA under their own licences,
so the image as a whole is subject to those terms. The full list is in
``/usr/share/doc/skasim/THIRD_PARTY_LICENSES.md`` inside the image and in
``docker/THIRD_PARTY_LICENSES.md`` in the repository.

**Maintainers.** The image recipe lives in ``docker/``. Pushing a release tag
``vX.Y.Z`` builds, smoke-tests and publishes ``:X.Y.Z`` and ``:latest``; the
*Docker image* workflow can also be run manually for an existing tag. After
changing ``environment.yml`` or the dependencies in ``pyproject.toml``,
regenerate the lockfiles with ``docker/update-lock.sh`` (requires Docker only).

Pip-only lightweight install
----------------------------

Use this path when you want to inspect the CLI, validate configuration, build
the documentation, or run tests that do not execute Karabo simulations::

    git clone git@github.com:espsrc/espsrc_ska_simulator.git
    cd espsrc_ska_simulator
    python -m pip install -e .

This install intentionally does not provide the full Karabo simulation runtime.
Commands such as ``skasim --help`` should work, but running a simulation requires
the conda environment below.

Conda full-runtime install
--------------------------

Use this procedure for simulation runs outside the container. The checked-in
``environment.yml`` declares the validated cluster runtime: Python 3.12,
``karabo-pipeline`` 0.36.0 (which no longer depends on RASCIL), OSKAR 2.12.2
from the ``i4ds/label/dev`` channel, and the CUDA 12 runtime. The editable pip
install also installs Python dependencies from ``pyproject.toml``, including
``shadems`` for UV-coverage plots::

    conda env create -f environment.yml
    conda activate skasim

Use flexible channel priority (``conda config --set channel_priority
flexible``) if the solve fails: Karabo comes from ``i4ds`` while OSKAR comes
from ``i4ds/label/dev``, and strict channel priority can also block MPI/FFTW
dependencies required by the Karabo runtime stack.

The exact package versions used by the container image are pinned in
``docker/conda-linux-64.lock`` and ``docker/requirements-pip.lock``.

WSClean
-------

The conda full-runtime install above installs WSClean through the
``karabo-pipeline`` dependency set: Karabo 0.36 depends on ``wsclean`` and
Conda installs WSClean 3.6, with EveryBeam and Dysco support, from the
``i4ds`` channel.

WSClean checks whether OpenBLAS multi-threading is enabled and aborts if
``OPENBLAS_NUM_THREADS`` is not set to ``1``. The ``skasim`` pipeline sets this
environment variable automatically before launching WSClean, so pipeline runs
can succeed even when a direct shell command such as ``wsclean --version``
fails. To inspect WSClean directly, use::

    conda run -n skasim env OPENBLAS_NUM_THREADS=1 wsclean --version

The WSClean command defaults to ``wsclean``. If WSClean is available through a
wrapper or container instead, pass it explicitly with ``--wsclean-command``.

On this machine, the Singularity container can be used as an example::

    skasim --imager wsclean \
      --wsclean-command "singularity exec /mnt/software/containers/wsclean-3.10-dysco.sif wsclean"

This command is an environment-specific example, not the default.

shadeMS
-------

``skasim`` uses ``shadems`` to generate the UV-coverage plot displayed beside
the telescope layout in the weblog. ``shadems`` is installed from pip through
the package dependencies. To inspect it directly, use::

    conda run -n skasim shadems --help

The pipeline runs shadeMS with U and V axes, a square canvas, and writable
Matplotlib/Numba cache directories inside the run output directory.

Verifying the installation
--------------------------

For the pip-only path, confirm that the CLI can be inspected::

    skasim --help

For the conda full-runtime path, verify the installed runtime before running
simulations::

    conda run -n skasim python -c "import karabo, oskar; print('karabo', karabo.__version__); print('oskar ok')"
    conda run -n skasim which oskar_sim_interferometer
    conda run -n skasim which wsclean
    conda run -n skasim which shadems
    conda run -n skasim skasim --help

``oskarpy`` is not required for the supported execution path; the successful
``import oskar`` check and a completed visibility simulation verify the OSKAR
backend used by Karabo.

A small WSClean smoke run with the named MIGHTEE catalog is::

    conda run -n skasim skasim \
      --output-dir smoke_mightee_wsclean_quick \
      --telescope MeerKAT \
      --observation-time 30 \
      --frequency-mhz 1300 \
      --bandwidth-mhz 25 \
      --n-channels 2 \
      --pixels 256 \
      --catalog MIGHTEE \
      --imager wsclean \
      --clean-iterations 20 \
      --overwrite

The verified smoke run completed with exit code 0 and wrote
``visibilities.MS``, a shadeMS UV-coverage plot, WSClean FITS products, PNG previews, ``run_manifest.json``, and ``weblog.html`` under
``smoke_mightee_wsclean_quick/``.
