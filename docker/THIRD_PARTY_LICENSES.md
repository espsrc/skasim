# Third-party components in the skasim container image

skasim itself is MIT-licensed (`/usr/share/doc/skasim/LICENSE`). The container
image also redistributes the third-party software below, each under its own
licence. Because the image includes GPL-licensed programs (notably WSClean and
EveryBeam), **the image as a whole is subject to the GPL terms of those
components**; the corresponding source code is available from the URLs listed.

All versions are pinned in `docker/conda-linux-64.lock` (conda packages) and
`docker/requirements-pip.lock` (pip packages); the CASA data version is set by
the `CASA_DATA_VERSION` argument in `docker/Dockerfile`.

| Component | How it is included | Licence | Source |
|---|---|---|---|
| WSClean | conda (`i4ds` channel, Karabo dependency) | GPL-3.0-or-later | https://gitlab.com/aroffringa/wsclean |
| EveryBeam | conda (`i4ds` channel, Karabo dependency) | GPL-3.0-or-later | https://git.astron.nl/RD/EveryBeam |
| casacore and python-casacore (with Dysco) | conda-forge | LGPL-2.0-or-later | https://github.com/casacore/casacore |
| CASA `casatools`, `casatasks`, `casaconfig` | PyPI wheels | GPL-2.0 / LGPL, as declared by the CASA project | https://casa.nrao.edu, https://open-bitbucket.nrao.edu/projects/CASA |
| CASA runtime data (`casarundata`) | Downloaded at build time (`/opt/casa-data`) | See CASA documentation | https://casadocs.readthedocs.io/en/stable/notebooks/external-data.html |
| Karabo pipeline | conda (`i4ds` channel) | MIT | https://github.com/i4Ds/Karabo-Pipeline |
| OSKAR | conda (`i4ds` channel) | BSD-3-Clause | https://github.com/OxfordSKA/OSKAR |
| NVIDIA CUDA runtime libraries | conda-forge | NVIDIA CUDA EULA | https://docs.nvidia.com/cuda/eula/ |
| micromamba | Binary | BSD-3-Clause | https://github.com/mamba-org/mamba |
| Ubuntu 24.04 base packages | apt | Various (see `/usr/share/doc/*/copyright`) | https://ubuntu.com |
| Other Python and conda packages | conda-forge / PyPI | Various (see each package's metadata in `/opt/conda`) | See lockfiles |
