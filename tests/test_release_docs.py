"""Release documentation verification."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_environment_file_defines_skasim_conda_runtime():
    """environment.yml declares the skasim name and verified channels."""
    text = (REPO_ROOT / "environment.yml").read_text(encoding="utf-8")
    assert "name: skasim" in text
    assert "- i4ds/label/dev" in text
    assert "- i4ds" in text
    assert "- conda-forge" in text
    assert "nvidia/label/cuda-11.7.0" not in text
    assert "- python=3.12" in text
    assert "- karabo-pipeline==0.36.0" in text
    assert "- cuda-version=12.9" in text


def test_installation_docs_describe_pip_and_conda_paths():
    """Installation docs separate lightweight pip use from full simulations."""
    text = (REPO_ROOT / "docs/installation.rst").read_text(encoding="utf-8")

    assert "pip-only" in text
    assert "conda env create -f environment.yml" in text
    assert "python=3.10" not in text
    assert "cuda-version=11.7" not in text
    assert "import karabo, oskar" in text
    assert "oskar_sim_interferometer" in text
    assert "Karabo extra" not in text
    assert "--wsclean-command" in text
    assert (
        "singularity exec /mnt/software/containers/wsclean-3.10-dysco.sif wsclean"
        in text
    )
    assert "shadeMS" in text
    assert "shadems" in text.lower()
    assert "smoke_mightee_wsclean_quick" in text
    assert "example" in text.lower()


def test_release_examples_use_0_2_cli_language():
    """Examples in docs use the unified skasim CLI instead of internal helpers."""
    text = (REPO_ROOT / "docs/installation.rst").read_text(encoding="utf-8")
    assert "skasim \\\n      --output-dir" in text
    assert "skasim --help" in text
    assert "skasim-pipeline" not in text
    assert "python -m skasim" not in text


def test_release_docs_document_removed_0_1_surface():
    """Documentation reflects that numeric catalog names are removed."""
    text = (REPO_ROOT / "docs/installation.rst").read_text(encoding="utf-8")
    assert "named mightee catalog" in text.lower()
    assert "--catalog MIGHTEE" in text


def test_smoke_docs_include_named_catalog_and_fits_shapes():
    """The smoke run verified for 0.2 is documented."""
    text = (REPO_ROOT / "docs/installation.rst").read_text(encoding="utf-8")
    assert "smoke_mightee_wsclean_quick" in text
    assert "visibilities.MS" in text
    assert "shadeMS UV-coverage" in text
    assert "WSClean FITS products" in text
    assert "PNG previews" in text
    assert "run_manifest.json" in text
    assert "weblog.html" in text


def test_installation_docs_describe_container_image():
    """The container distribution channel is documented for Docker and Apptainer."""
    text = (REPO_ROOT / "docs/installation.rst").read_text(encoding="utf-8")
    assert "ghcr.io/espsrc/skasim" in text
    assert '--user "$(id -u):$(id -g)"' in text
    assert "apptainer pull" in text
    assert "linux/amd64" in text
    assert "THIRD_PARTY_LICENSES.md" in text
    assert "docker/update-lock.sh" in text


def test_container_recipe_pins_runtime_and_excludes_data():
    """The image installs from lockfiles and keeps repository data out."""
    docker_dir = REPO_ROOT / "docker"
    dockerfile = (docker_dir / "Dockerfile").read_text(encoding="utf-8")
    assert "conda-linux-64.lock" in dockerfile
    assert "requirements-pip.lock" in dockerfile
    conda_lock = (docker_dir / "conda-linux-64.lock").read_text(encoding="utf-8")
    assert "@EXPLICIT" in conda_lock
    assert "/karabo-pipeline-0.36.0-" in conda_lock
    pip_lock = (docker_dir / "requirements-pip.lock").read_text(encoding="utf-8")
    assert (
        "casatools @ " in pip_lock and "manylinux_2_28_x86_64.whl#sha256=" in pip_lock
    )
    ignore = (docker_dir / "Dockerfile.dockerignore").read_text(encoding="utf-8")
    assert ignore.splitlines()[2:4] == ["*", "!docker/"]


def test_package_data_includes_weblog_template():
    """Non-editable installs ship the weblog template."""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.setuptools.package-data]" in text
    assert 'skasim = ["templates/*.j2"]' in text
    assert (REPO_ROOT / "src/skasim/templates/weblog.html.j2").is_file()
