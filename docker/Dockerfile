# syntax=docker/dockerfile:1.7
#
# skasim distribution image: skasim + Karabo/OSKAR + WSClean (all from the
# pinned conda environment) + CASA (casatools/casatasks wheels), on the
# validated cluster stack (Python 3.12, Karabo 0.36, CUDA 12).
#
# Build from the repository root:
#   docker build -f docker/Dockerfile -t skasim:dev .
#
# Build the skasim source of another git ref with this recipe (used by the
# release workflow to back-build tags that predate the Docker files):
#   git worktree add /tmp/skasim-v0.5.0 v0.5.0
#   docker build -f docker/Dockerfile --build-context skasim-src=/tmp/skasim-v0.5.0 .

ARG UBUNTU_VERSION=24.04
ARG MICROMAMBA_VERSION=2.0.5

# skasim source tree. Overridden with --build-context skasim-src=<dir>.
FROM scratch AS skasim-src
COPY pyproject.toml README.md LICENSE /
COPY src /src

FROM mambaorg/micromamba:${MICROMAMBA_VERSION} AS micromamba

FROM ubuntu:${UBUNTU_VERSION}

ARG SKASIM_VERSION=dev
ARG CASA_DATA_VERSION=casarundata-2026.10.27-1.tar.gz

LABEL org.opencontainers.image.title="skasim" \
      org.opencontainers.image.description="SKA synthetic observation simulator (skasim) with Karabo, OSKAR, CASA and WSClean" \
      org.opencontainers.image.source="https://github.com/espsrc/skasim" \
      org.opencontainers.image.version="${SKASIM_VERSION}" \
      org.opencontainers.image.licenses="MIT AND GPL-3.0-or-later AND LGPL-2.0-or-later AND BSD-3-Clause"

RUN apt-get update -qq \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=micromamba /bin/micromamba /usr/local/bin/micromamba

ENV MAMBA_ROOT_PREFIX=/opt/conda
COPY docker/conda-linux-64.lock docker/requirements-pip.lock /tmp/lock/
RUN micromamba install -y -q -n base -f /tmp/lock/conda-linux-64.lock \
    && micromamba clean -a -f -y \
    && micromamba run -n base python -m pip install -q --no-deps --no-cache-dir \
        --root-user-action=ignore -r /tmp/lock/requirements-pip.lock \
    && rm -rf /tmp/lock

# CASA runtime data (measures, ephemerides, ...). CASA reads its location from
# casasiteconfig.py; auto-updates are off so runs never download at start-up
# and work for any --user uid. casacore (python-casacore, WSClean) is pointed
# at the same measures tables through /etc/casarc.
RUN site="$(micromamba run -n base python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
    && printf '%s\n' \
        'measurespath = "/opt/casa-data"' \
        'measures_auto_update = False' \
        'data_auto_update = False' > "${site}/casasiteconfig.py" \
    && echo 'measures.directory: /opt/casa-data' > /etc/casarc \
    && mkdir -p /opt/casa-data \
    && cd /tmp && micromamba run -n base python -c \
        "from casaconfig import pull_data; pull_data('/opt/casa-data', version='${CASA_DATA_VERSION}')" \
    && rm -f /tmp/casa-*.log

# Package data (e.g. the weblog templates) is copied explicitly as well:
# releases up to v0.5.0 do not declare it in pyproject.toml, so a regular
# (non-editable) install would leave it out.
COPY --from=skasim-src /pyproject.toml /README.md /LICENSE /tmp/skasim/
COPY --from=skasim-src /src /tmp/skasim/src
RUN micromamba run -n base python -m pip install -q --no-deps --no-build-isolation \
        --no-cache-dir --root-user-action=ignore /tmp/skasim \
    && site="$(micromamba run -n base python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
    && cd /tmp/skasim/src \
    && find skasim -type f ! -name '*.py' ! -path '*/__pycache__/*' \
        -exec install -D -m 0644 {} "${site}/{}" \; \
    && cd / \
    && install -D -m 0644 /tmp/skasim/LICENSE /usr/share/doc/skasim/LICENSE \
    && rm -rf /tmp/skasim
COPY docker/THIRD_PARTY_LICENSES.md /usr/share/doc/skasim/THIRD_PARTY_LICENSES.md
COPY --chmod=0755 docker/entrypoint.sh /usr/local/bin/skasim-entrypoint

# Runs are expected with --user "$(id -u):$(id -g)", so HOME must be
# writable by any uid (matplotlib, astropy and CASA keep caches there).
# WSClean aborts when OpenBLAS multi-threading is enabled; skasim sets this
# itself before calling WSClean, the ENV covers direct `wsclean` calls.
RUN mkdir -p /home/skasim /work && chmod 1777 /home/skasim /work
ENV HOME=/home/skasim \
    PATH=/opt/conda/bin:${PATH} \
    CASARCFILES=/etc/casarc \
    MPLBACKEND=Agg \
    OPENBLAS_NUM_THREADS=1

RUN cd /tmp && skasim-entrypoint wsclean --version \
    && skasim-entrypoint python -c \
        "import skasim, karabo, oskar, casatools, casatasks; from casatasks import ft, importfits; print('imports ok')" \
    && skasim-entrypoint --help > /dev/null \
    && rm -rf /tmp/casa-*.log /home/skasim/.[!.]* /home/skasim/*

WORKDIR /work
ENTRYPOINT ["/usr/local/bin/skasim-entrypoint"]
CMD ["--help"]
