import os
import sys
import warnings

# Check if JuliaCall is already loaded, and if so, warn the user
# about the relevant environment variables. If not loaded,
# set up sensible defaults.
if "juliacall" in sys.modules:
    #warnings.warn(
    #    "juliacall module already imported. "
    #    "Make sure that you have set the environment variable `PYTHON_JULIACALL_HANDLE_SIGNALS=yes` to avoid segfaults. "
    #    "Also note that truncatedgaussianmixtures will not be able to configure `PYTHON_JULIACALL_THREADS` or `PYTHON_JULIACALL_OPTLEVEL` for you."
    #)
    pass
else:
    # Required to avoid segfaults (https://juliapy.github.io/PythonCall.jl/dev/faq/)
    if os.environ.get("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes") != "yes":
        warnings.warn(
            "PYTHON_JULIACALL_HANDLE_SIGNALS environment variable is set to something other than 'yes' or ''. "
            + "You will experience segfaults if running with multithreading."
        )

    if os.environ.get("PYTHON_JULIACALL_THREADS", "auto") != "auto":
        warnings.warn(
            "PYTHON_JULIACALL_THREADS environment variable is set to something other than 'auto', "
            "so truncatedgaussianmixtures was not able to set it. You may wish to set it to `'auto'` for full use "
            "of your CPU."
        )

    # TODO: Remove these when juliapkg lets you specify this
    for k, default in (
        ("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes"),
        ("PYTHON_JULIACALL_THREADS", "auto"),
        ("PYTHON_JULIACALL_OPTLEVEL", "3"),
    ):
        os.environ[k] = os.environ.get(k, default)


# Register TruncatedGaussianMixtures.jl with juliapkg *before* juliacall starts,
# so that it is part of the resolved Julia environment (juliapkg deletes the
# Manifest on every re-resolve, so an ad-hoc `Pkg.add`/`Pkg.develop` at runtime
# does not survive).
#
# * TGMM_JULIA_DEV_PATH=/path/to/TruncatedGaussianMixtures.jl  -> `Pkg.develop` that
#   checkout (the choice is remembered in this Python environment's juliapkg
#   deps file, so it only needs to be set once).
# * otherwise the GitHub version at TGMM_JULIA_REV is used (unless a dev
#   checkout was recorded before).
TGMM_JULIA_UUID = "d6a30cd5-d352-42fd-9418-e34000539a5a"
TGMM_JULIA_URL = "https://github.com/Potatoasad/TruncatedGaussianMixtures.jl"
# the JAX backend needs the matching Julia branch; switch back to "main" once it is merged there
TGMM_JULIA_REV = "jax-backend"


def _register_tgmm_julia_package():
    import juliapkg

    dev_path = os.environ.get("TGMM_JULIA_DEV_PATH")
    current = juliapkg.deps.load_cur_deps().get("packages", {}).get("TruncatedGaussianMixtures")
    if dev_path:
        wanted = {"uuid": TGMM_JULIA_UUID, "dev": True, "path": os.path.abspath(os.path.expanduser(dev_path))}
    elif current is None or not current.get("dev"):
        wanted = {"uuid": TGMM_JULIA_UUID, "url": TGMM_JULIA_URL, "rev": TGMM_JULIA_REV}
    else:
        return
    if current != wanted:
        spec = {k: v for k, v in wanted.items() if k != "uuid"}
        juliapkg.add("TruncatedGaussianMixtures", TGMM_JULIA_UUID, **spec)


REVISE_UUID = "295af30f-e4ad-537b-8983-00126c2a3abe"
# TGMM_JULIA_REVISE=1: load Revise before the package so that edits to a
# developed TruncatedGaussianMixtures.jl checkout take effect in a running
# Python/Jupyter session.
_USE_REVISE = os.environ.get("TGMM_JULIA_REVISE", "") not in ("", "0", "no", "false")


def _register_revise():
    import juliapkg
    if "Revise" not in juliapkg.deps.load_cur_deps().get("packages", {}):
        juliapkg.add("Revise", REVISE_UUID)


if "juliacall" not in sys.modules:
    _register_tgmm_julia_package()
    if _USE_REVISE:
        _register_revise()

from juliacall import Main as jl  # type: ignore
from .julia_helpers import *

jl_version = (jl.VERSION.major, jl.VERSION.minor, jl.VERSION.patch)

# Next, automatically load the juliacall extension if we're in a Jupyter notebook
autoload_extensions = os.environ.get("PYSR_AUTOLOAD_EXTENSIONS", "yes")
if autoload_extensions in {"yes", ""} and jl_version >= (1, 9, 0):
    try:
        get_ipython = sys.modules["IPython"].get_ipython

        if "IPKernelApp" not in get_ipython().config:
            raise ImportError("console")

        print(
            "Detected Jupyter notebook. Loading juliacall extension. Set `PYSR_AUTOLOAD_EXTENSIONS=no` to disable."
        )

        # TODO: Turn this off if juliacall does this automatically
        get_ipython().run_line_magic("load_ext", "juliacall")
    except Exception:
        pass
elif autoload_extensions not in {"no", "yes", ""}:
    warnings.warn(
        "PYSR_AUTOLOAD_EXTENSIONS environment variable is set to something other than 'yes' or 'no' or ''."
    )


jl.seval("using Pkg: Pkg")
Pkg = jl.Pkg


def force_install_tgmm(force=True, install_pkgs=False, needed_pkgs=None):
    """Re-resolve the Julia environment (TGMM is managed through juliapkg now)."""
    import juliapkg
    juliapkg.resolve(force=True)


if _USE_REVISE:
    jl.seval("using Revise")
jl.seval("using TruncatedGaussianMixtures: TruncatedGaussianMixtures")
jl.seval("using TruncatedGaussianMixtures")
# The legacy TGMM object loaded these into Julia's Main on construction, and user
# code (e.g. custom `jl.seval` helpers using MvNormal / diag) relies on them.
jl.seval("using LinearAlgebra, Distributions")
TruncatedGaussianMixtures = jl.TruncatedGaussianMixtures


def revise():
    """Apply pending edits of a developed TruncatedGaussianMixtures.jl (needs TGMM_JULIA_REVISE=1).

    Revise's file watchers are Julia tasks, which only run when Julia yields,
    so give them a moment before revising.
    """
    if not _USE_REVISE:
        raise RuntimeError("set TGMM_JULIA_REVISE=1 before importing truncatedgaussianmixtures")
    jl.seval("sleep(0.05); Revise.revise(; throw=true)")
