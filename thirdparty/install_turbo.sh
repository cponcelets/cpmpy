#!/usr/bin/env bash
#
# install_turbo.sh - install the two direct dependencies of CPM_turbo:
#   1. MiniZinc (used only as a MiniZinc -> FlatZinc translator, see cpmpy/solvers/turbo.py)
#   2. turbo_python (the GPU solver itself, built from https://github.com/cponcelets/turbo/tree/turbo_python)
#
# Linux x86_64 only (matches the turbo_python build requirements: CUDA + CMake).
# Under WSL, thirdparty/wsl.patch is auto-applied to turbo (disables
# concurrent managed memory, which WSL2's CUDA does not support).
#
# Usage:
#   ./thirdparty/install_turbo.sh [options]
#
# Options:
#   --lattice-land-dir DIR   Parent dir for the lattice-land sibling repos (default: thirdparty/lattice-land)
#   --preset NAME            CMake preset: gpu-release-local | cpu-release-local (default: auto-detected from nvcc)
#   --venv PATH              Python venv to install turbo_python into (default: $VIRTUAL_ENV, else <repo>/.venv, else ./turbo-venv)
#   --skip-minizinc          Skip the MiniZinc install/check step
#   --skip-clone             Skip cloning/updating the lattice-land repos and patch step (for re-runs)
#   -h, --help                Show this help
#
# What this script deliberately does NOT do:
#   - It never runs `sudo`/apt-get. Build tools (cmake, nvcc, doxygen, libxml2) are checked, and if
#     missing you get the exact command to install them yourself - guessing at CUDA/driver setup is
#     the kind of thing that fails silently in confusing ways, so we stop instead.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LATTICE_LAND_DIR="$REPO_ROOT/thirdparty/lattice-land"
PRESET=""
VENV_PATH=""
SKIP_MINIZINC=0
SKIP_CLONE=0

TURBO_REPO="git@github.com:cponcelets/turbo.git"
TURBO_BRANCH="turbo_python"
# url|ref pairs. These are NOT tags or `main` - they're the exact commits
# that actually compiled a working turbo_python.so against turbo_python@4f68ca5b
# (verified from a real prior build with the .so artifacts still on disk), not
# an inference from any repo's current HEAD. `main` for these repos moves
# fast and has broken this build twice already (see git history of this
# file): once by not having the SimplifierStats API turbo's source expects,
# once by having split cartesian_product.hpp out into a separate lala-interval
# repo. This exact combination has both without needing lala-interval at all.
SIBLING_REPOS=(
    "https://github.com/xcsp3team/XCSP3-CPP-Parser.git|179670dd97cd2b87cdeb93aa7268b57bba719490"
    "https://github.com/ptal/cpp-peglib.git|245b59446d424b499b316ee77ec59de18443de23"
    "https://github.com/lattice-land/cuda-battery.git|a7d99f0b2100b86aba8b0490cabe08da6c045c28"
    "https://github.com/lattice-land/lala-core.git|91a2356e63c883bdd81d387f0355ed44fddc9089"
    "https://github.com/lattice-land/lala-pc.git|ad3abba0ff4161422be60603ca790030c9bff339"
    "https://github.com/lattice-land/lala-power.git|da55aaf01a7476e9e36eca2dfdedb32ef8276aa3"
    "https://github.com/lattice-land/lala-parsing.git|c49de4c5e511eaa205ea55a7134122095d7fd2cf"
)

log()  { echo -e "\033[1;34m[install_turbo]\033[0m $*"; }
die()  { echo -e "\033[1;31m[install_turbo] ERROR:\033[0m $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lattice-land-dir) LATTICE_LAND_DIR="$2"; shift 2 ;;
        --preset) PRESET="$2"; shift 2 ;;
        --venv) VENV_PATH="$2"; shift 2 ;;
        --skip-minizinc) SKIP_MINIZINC=1; shift ;;
        --skip-clone) SKIP_CLONE=1; shift ;;
        -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) die "Unknown option: $1 (see --help)" ;;
    esac
done

[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] \
    || die "This script only supports Linux x86_64 (turbo_python's CUDA/CMake build isn't portable to other platforms)."

# ---------------------------------------------------------------------------
# 1. Check build prerequisites turbo_python needs (never auto-installed: these
#    are system-wide/hardware-sensitive, so we stop rather than guess).
# ---------------------------------------------------------------------------
have_libxml2() {
    # Checked three ways since installs vary (apt registers it with ldconfig
    # and ships a .pc file, but conda/manual installs may only satisfy one).
    ldconfig -p 2>/dev/null | grep -q libxml2 && return 0
    pkg-config --exists libxml-2.0 2>/dev/null && return 0
    compgen -G "/usr/lib/*/libxml2.so*" >/dev/null 2>&1 || compgen -G "/usr/lib/libxml2.so*" >/dev/null 2>&1
}

check_build_tools() {
    log "Checking build prerequisites..."
    local missing=()

    command -v cmake >/dev/null || missing+=("cmake (>=3.27): sudo apt-get install cmake")
    if command -v cmake >/dev/null; then
        local cmv; cmv="$(cmake --version | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')"
        [[ "$(printf '%s\n' "3.27.0" "$cmv" | sort -V | head -1)" == "3.27.0" ]] \
            || missing+=("cmake >=3.27 required, found $cmv: upgrade via https://apt.kitware.com or pip install cmake")
    fi

    command -v doxygen >/dev/null || missing+=("doxygen: sudo apt-get install doxygen")
    have_libxml2 || missing+=("libxml2: sudo apt-get install libxml2-dev")

    if [[ "$PRESET" == gpu-* || -z "$PRESET" ]]; then
        if command -v nvcc >/dev/null; then
            local nvv; nvv="$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+')"
            [[ "$(printf '%s\n' "12.0" "$nvv" | sort -V | head -1)" == "12.0" ]] \
                || die "nvcc $nvv found, but turbo_python requires CUDA 12.0+. Install a newer CUDA toolkit and retry."
            [[ -z "$PRESET" ]] && PRESET="gpu-release-local"
        elif [[ "$PRESET" == gpu-* ]]; then
            die "nvcc not found on PATH, but --preset $PRESET was requested. Install the CUDA toolkit (https://developer.nvidia.com/cuda-downloads) and retry."
        else
            log "No CUDA compiler (nvcc) found - falling back to --preset cpu-release-local."
            PRESET="cpu-release-local"
        fi
    fi

    python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
        || missing+=("python3 >=3.10 required, found $(python3 --version)")

    if [[ ${#missing[@]} -gt 0 ]]; then
        log "Missing prerequisites:"
        printf '  - %s\n' "${missing[@]}"
        die "install the above and re-run this script."
    fi
    log "Build prerequisites OK (preset: $PRESET)."
}

# ---------------------------------------------------------------------------
# 2. MiniZinc: pip package (python bindings) + the binary bundle (the actual
#    `minizinc` executable, downloaded from the official GitHub releases).
# ---------------------------------------------------------------------------
install_minizinc() {
    [[ $SKIP_MINIZINC -eq 1 ]] && { log "Skipping MiniZinc (--skip-minizinc)."; return; }

    log "Installing MiniZinc python bindings into $VENV_PATH..."
    pip_install minizinc

    if command -v minizinc >/dev/null; then
        log "MiniZinc binary already on PATH: $(minizinc --version | head -1)"
        return
    fi

    log "MiniZinc binary not found on PATH, downloading the official bundle..."
    local dest="$HOME/.local/opt/minizinc"
    local tmp; tmp="$(mktemp -d)"
    local asset_url
    asset_url="$(curl -sL https://api.github.com/repos/MiniZinc/MiniZincIDE/releases/latest \
        | python3 -c 'import sys,json; d=json.load(sys.stdin); print(next(a["browser_download_url"] for a in d["assets"] if a["name"].endswith("x86_64-linux-gnu.tgz")))')"
    [[ -n "$asset_url" ]] || die "Could not find a linux x86_64 MiniZinc bundle in the latest release. Download it manually from https://www.minizinc.org/software.html"

    curl -sL "$asset_url" -o "$tmp/minizinc.tgz"
    mkdir -p "$dest"
    tar -xzf "$tmp/minizinc.tgz" -C "$dest" --strip-components=1
    rm -rf "$tmp"

    [[ -x "$dest/bin/minizinc" ]] || die "Extracted MiniZinc bundle to $dest but bin/minizinc is missing - check $dest manually."
    log "MiniZinc bundle installed at $dest."
    export PATH="$dest/bin:$PATH"

    echo "export PATH=\"$dest/bin:\$PATH\"" > "$REPO_ROOT/thirdparty/env.sh"
    log "Wrote $REPO_ROOT/thirdparty/env.sh - source it in every new shell before using CPM_turbo:"
    log "    source thirdparty/env.sh"
}

# ---------------------------------------------------------------------------
# 3. Clone the lattice-land sibling repos + turbo (turbo_python branch), and
#    apply the two required patches (idempotent: skips already-applied ones).
# ---------------------------------------------------------------------------
is_real_clone() {
    git -C "$1" rev-parse --git-dir >/dev/null 2>&1
}

clone_and_patch() {
    [[ $SKIP_CLONE -eq 1 ]] && { log "Skipping clone/patch step (--skip-clone)."; return; }

    mkdir -p "$LATTICE_LAND_DIR"
    log "Cloning lattice-land dependencies into $LATTICE_LAND_DIR..."
    for entry in "${SIBLING_REPOS[@]}"; do
        local repo="${entry%%|*}" ref="${entry##*|}"
        local name; name="$(basename "$repo" .git)"
        if is_real_clone "$LATTICE_LAND_DIR/$name"; then
            log "  $name already present."
        else
            # CMake's FetchContent(SOURCE_DIR=...) pre-creates an empty placeholder
            # directory when a local dep is missing, so an existing-but-empty dir
            # here means a previous build failed partway - remove it and re-clone.
            rm -rf "${LATTICE_LAND_DIR:?}/$name"
            git clone --quiet "$repo" "$LATTICE_LAND_DIR/$name"
        fi
        # Always pin to the exact ref turbo_python's CMake chain expects, even on
        # an already-cloned repo: these repos move fast, and a sibling sitting on
        # `main` (or any other ref) is a real source of "missing header" build
        # failures that have nothing to do with cmake itself.
        git -C "$LATTICE_LAND_DIR/$name" fetch --quiet --tags origin "$ref" 2>/dev/null || true
        git -C "$LATTICE_LAND_DIR/$name" checkout --quiet "$ref"
        log "  $name @ $ref"
    done

    if is_real_clone "$LATTICE_LAND_DIR/turbo"; then
        log "  turbo already present, pulling latest $TURBO_BRANCH..."
        git -C "$LATTICE_LAND_DIR/turbo" checkout --quiet "$TURBO_BRANCH"
        git -C "$LATTICE_LAND_DIR/turbo" pull --quiet
    else
        rm -rf "${LATTICE_LAND_DIR:?}/turbo"
        git clone --quiet --recursive "$TURBO_REPO" "$LATTICE_LAND_DIR/turbo"
        git -C "$LATTICE_LAND_DIR/turbo" checkout --quiet "$TURBO_BRANCH"
    fi

    log "Applying required patches..."
    apply_patch_once "lala-core" "$LATTICE_LAND_DIR/turbo/patches/lala-core_inline_get_value_of.patch"
    apply_patch_once "lala-parsing" "$LATTICE_LAND_DIR/turbo/patches/lala-parsing_outputs.patch"

    if is_wsl; then
        log "WSL detected, applying WSL-specific patch (disables concurrent managed memory, unsupported under WSL2's CUDA)..."
        apply_patch_once "turbo" "$REPO_ROOT/thirdparty/wsl.patch"
    fi
}

is_wsl() {
    [[ -n "${WSL_DISTRO_NAME:-}" ]] && return 0
    grep -qiE "microsoft|wsl" /proc/sys/kernel/osrelease 2>/dev/null
}

apply_patch_once() {
    local target_repo="$1" patch_path="$2"
    local target_dir="$LATTICE_LAND_DIR/$target_repo"
    local patch_name; patch_name="$(basename "$patch_path")"
    [[ -f "$patch_path" ]] || die "Expected patch not found: $patch_path"

    if git -C "$target_dir" apply --check "$patch_path" 2>/dev/null; then
        git -C "$target_dir" apply "$patch_path"
        log "  applied $patch_name to $target_repo"
    else
        log "  $patch_name already applied (or doesn't apply cleanly) to $target_repo, skipping"
    fi
}

# ---------------------------------------------------------------------------
# 4. venv + build turbo_python via its CMake preset, then drop the compiled
#    extension module into the venv's site-packages (CMAKE_PRESET / `pip
#    install .` do the CMake build but don't install the .so themselves,
#    see the turbo_python README).
# ---------------------------------------------------------------------------
setup_venv() {
    if [[ -z "$VENV_PATH" ]]; then
        if [[ -n "${VIRTUAL_ENV:-}" ]]; then
            VENV_PATH="$VIRTUAL_ENV"
        elif [[ -d "$REPO_ROOT/.venv" ]]; then
            VENV_PATH="$REPO_ROOT/.venv"
        else
            VENV_PATH="$REPO_ROOT/turbo-venv"
        fi
    fi
    if [[ ! -d "$VENV_PATH" ]]; then
        log "Creating venv at $VENV_PATH..."
        python3 -m venv "$VENV_PATH"
    fi
    log "Using venv: $VENV_PATH"

    # venvs created by `uv` ship no `pip` binary/module at all (uv installs
    # packages itself via `uv pip`); fall back to bootstrapping pip via
    # ensurepip only when uv isn't available.
    if command -v uv >/dev/null; then
        log "  uv detected, using it to install into the venv."
    elif [[ ! -x "$VENV_PATH/bin/pip" ]]; then
        "$VENV_PATH/bin/python" -m ensurepip --upgrade
    fi
    if ! command -v uv >/dev/null; then
        pip_install --upgrade pip setuptools wheel
    fi
    pip_install pybind11
}

# Installs into $VENV_PATH regardless of whether it has its own `pip`.
pip_install() {
    if command -v uv >/dev/null; then
        uv pip install --python "$VENV_PATH/bin/python" "$@"
    else
        "$VENV_PATH/bin/python" -m pip install "$@"
    fi
}

build_turbo_python() {
    log "Building turbo_python (preset: $PRESET, this runs nvcc/cmake and can take a while)..."
    (
        cd "$LATTICE_LAND_DIR/turbo"
        CMAKE_PRESET="$PRESET" pip_install .
    )

    local so_file
    so_file="$(find "$LATTICE_LAND_DIR/turbo/build/$PRESET" -maxdepth 1 -name 'turbo_python*.so' | head -1)"
    [[ -n "$so_file" ]] || die "Build finished but no turbo_python*.so found under $LATTICE_LAND_DIR/turbo/build/$PRESET"

    local site_packages
    site_packages="$("$VENV_PATH/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
    cp "$so_file" "$site_packages/"
    log "Installed $(basename "$so_file") into $site_packages"
}

# ---------------------------------------------------------------------------
# 5. Register the turbo.<flavor>.release.msc MiniZinc solver configuration.
#    The one shipped in the repo hard-codes the original author's absolute
#    paths, so we rewrite them to point at this machine's clone before
#    installing it into MiniZinc's user solver directory.
# ---------------------------------------------------------------------------
install_msc() {
    local flavor="${PRESET%%-*}"  # gpu-release-local -> gpu, cpu-release-local -> cpu
    local msc_src="$LATTICE_LAND_DIR/turbo/benchmarks/minizinc/turbo.$flavor.release.msc"
    [[ -f "$msc_src" ]] || die "Expected solver config not found: $msc_src"

    local turbo_dir; turbo_dir="$(cd "$LATTICE_LAND_DIR/turbo" && pwd)"
    local dest_dir="$HOME/.minizinc/solvers"
    mkdir -p "$dest_dir"

    python3 - "$msc_src" "$turbo_dir" "$dest_dir/turbo.$flavor.release.msc" <<'PYEOF'
import json, re, sys
src, turbo_dir, dest = sys.argv[1:4]
with open(src) as f:
    cfg = json.load(f)
old_prefix = re.match(r"(.*)/build/", cfg["executable"]).group(1)
cfg["executable"] = cfg["executable"].replace(old_prefix, turbo_dir)
cfg["mznlib"] = cfg["mznlib"].replace(old_prefix, turbo_dir)
with open(dest, "w") as f:
    json.dump(cfg, f, indent=4)
PYEOF

    log "Installed MiniZinc solver config: $dest_dir/turbo.$flavor.release.msc"
}

# ---------------------------------------------------------------------------
main() {
    check_build_tools
    setup_venv
    install_minizinc
    clone_and_patch
    build_turbo_python
    install_msc

    log "Done."
    [[ -f "$REPO_ROOT/thirdparty/env.sh" ]] && log "Run 'source thirdparty/env.sh' in every new shell (puts MiniZinc's bundle on PATH)."
    log "Verify with:"
    log "    $VENV_PATH/bin/python -c 'import turbo_python; print(turbo_python.__file__)'"
    log "    minizinc --solvers | grep -i turbo"
}

# Guard against `main` running when this file is sourced instead of executed.
[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main "$@"
