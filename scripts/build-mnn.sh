#!/usr/bin/env bash
set -euo pipefail

readonly MNN_VERSION=3.6.1
readonly MNN_COMMIT=d407447ed56c4121a11ccbd266dc184ca1ead0c2
readonly MNN_REPOSITORY=https://github.com/alibaba/MNN.git
readonly BUILD_JOBS="${BUILD_JOBS:-8}"

fail() {
    echo "$*" >&2
    exit 1
}

# 1. 입력과 도구를 확인하고 경로를 확정한다.
check_inputs() {
    build_python=false
    if [[ $# -gt 1 || ($# -eq 1 && "$1" != --python) ]]; then
        fail "Usage: bash scripts/build-mnn.sh [--python]"
    fi
    if [[ $# -eq 1 ]]; then
        build_python=true
    fi
    if [[ ! "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]]; then
        fail "BUILD_JOBS must be a positive integer."
    fi

    local platform tool script_dir
    platform=$(uname -s)
    case "$platform" in
        MINGW*|MSYS*) ;;
        *) fail "Run this script in Git Bash on Windows." ;;
    esac
    for tool in git cmake ninja cygpath cmd.exe mkdir tee dirname; do
        command -v "$tool" >/dev/null 2>&1 || fail "Required tool not found: $tool"
    done

    # MSVC 환경을 불러온 뒤에도 같은 도구를 사용한다.
    CMAKE_EXE=$(command -v cmake)
    GIT_EXE=$(command -v git)
    MKDIR_EXE=$(command -v mkdir)
    TEE_EXE=$(command -v tee)
    CYGPATH_EXE=$(command -v cygpath)
    NINJA_EXE=$(command -v ninja)
    NINJA_EXE=$("$CYGPATH_EXE" -m "$NINJA_EXE")
    script_dir=$(dirname -- "${BASH_SOURCE[0]}")
    PROJECT_DIR=$(cd -- "$script_dir/.." && pwd)
    PROJECT_WINDOWS=$("$CYGPATH_EXE" -m "$PROJECT_DIR")
    RUNTIME_DIR="$PROJECT_DIR/MNNRuntime"
    RUNTIME_WINDOWS="$PROJECT_WINDOWS/MNNRuntime"
    readonly CMAKE_EXE GIT_EXE MKDIR_EXE TEE_EXE CYGPATH_EXE NINJA_EXE
    readonly PROJECT_DIR PROJECT_WINDOWS RUNTIME_DIR RUNTIME_WINDOWS

    if "$build_python"; then
        PYTHON_EXE=$("$CYGPATH_EXE" -m "${PYTHON_EXECUTABLE:-$PROJECT_DIR/.venv/Scripts/python.exe}")
        readonly PYTHON_EXE
        if [[ ! -f "$PYTHON_EXE" ]]; then
            fail "Run uv sync --locked --group runtime first, or set PYTHON_EXECUTABLE."
        fi
        "$PYTHON_EXE" -c 'import sys; import numpy; sys.exit("Python 3.12 is required.") if sys.version_info[:2] != (3, 12) else None'
    fi
}

# 2. 설치된 vcvars64.bat에서 MSVC x64 환경을 가져온다.
prepare_msvc() {
    local vswhere vs_install vcvars msvc_env entry name value msvc_path
    vswhere=$("$CYGPATH_EXE" -u "${SYSTEMDRIVE:-C:}/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe")
    if [[ ! -f "$vswhere" ]]; then
        fail "Install Visual Studio Build Tools with Desktop development with C++."
    fi
    vs_install=$("$vswhere" -latest -products '*' \
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 \
        -property installationPath)
    vs_install=${vs_install//$'\r'/}
    if [[ -z "$vs_install" ]]; then
        fail "MSVC x64 build tools were not found."
    fi
    vcvars="${vs_install}\\VC\\Auxiliary\\Build\\vcvars64.bat"
    msvc_env=$(printf 'call "%s" >nul\r\nif errorlevel 1 exit 1\r\nset\r\nexit\r\n' "$vcvars" | MSYS_NO_PATHCONV=1 cmd.exe /d /q)
    msvc_path=''
    while IFS= read -r entry; do
        entry=${entry%$'\r'}
        name=${entry%%=*}
        value=${entry#*=}
        case "$name" in
            PATH|Path|path) msvc_path=$("$CYGPATH_EXE" -u -p "$value") ;;
            INCLUDE|LIB|LIBPATH) export "$name=$value" ;;
        esac
    done <<< "$msvc_env"
    if [[ -z "$msvc_path" || -z "${INCLUDE:-}" || -z "${LIB:-}" ]]; then
        fail "MSVC environment is incomplete."
    fi
    export PATH="$msvc_path"
    command -v cl.exe >/dev/null 2>&1 || fail "MSVC compiler not found."
    command -v rc.exe >/dev/null 2>&1 || fail "Windows SDK resource compiler not found."
}

# 3. 지정한 버전의 수정되지 않은 소스를 사용한다.
prepare_source() {
    local source_commit source_changes
    cd -- "$PROJECT_DIR"
    "$MKDIR_EXE" -p MNNRuntime
    if [[ ! -d "$RUNTIME_DIR/MNN" ]]; then
        "$GIT_EXE" clone --depth 1 --branch "$MNN_VERSION" "$MNN_REPOSITORY" "$RUNTIME_WINDOWS/MNN"
    fi
    source_commit=$("$GIT_EXE" -C "$RUNTIME_WINDOWS/MNN" rev-parse HEAD)
    if [[ "$source_commit" != "$MNN_COMMIT" ]]; then
        fail "MNN source must be at $MNN_COMMIT (version $MNN_VERSION)."
    fi
    source_changes=$("$GIT_EXE" -C "$RUNTIME_WINDOWS/MNN" status --porcelain)
    if [[ -n "$source_changes" ]]; then
        fail "MNN source has local changes. Resolve them before building: $RUNTIME_DIR/MNN"
    fi
}

# 4. OpenCL Runtime과 검증 도구를 빌드한다.
build_runtime() {
    "$CMAKE_EXE" -S "$RUNTIME_WINDOWS/MNN" -B "$RUNTIME_WINDOWS/build" -G Ninja \
        "-DCMAKE_MAKE_PROGRAM=$NINJA_EXE" \
        -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl \
        -DCMAKE_BUILD_TYPE=Release \
        -DMNN_OPENCL=ON -DMNN_USE_SYSTEM_LIB=OFF -DMNN_SEP_BUILD=OFF \
        -DMNN_BUILD_SHARED_LIBS=ON -DMNN_WIN_RUNTIME_MT=OFF \
        -DMNN_BUILD_TEST=ON -DMNN_BUILD_TOOLS=ON -DMNN_BUILD_CONVERTER=OFF \
        2>&1 | "$TEE_EXE" "$RUNTIME_DIR/configure.log"
    "$CMAKE_EXE" --build "$RUNTIME_WINDOWS/build" \
        --target MNN run_test.out MNNV2Basic.out --parallel "$BUILD_JOBS" \
        2>&1 | "$TEE_EXE" "$RUNTIME_DIR/build.log"
    echo "MNN $MNN_VERSION OpenCL Runtime built: $RUNTIME_DIR/build/MNN.dll"
}

# 5. --python이면 동일한 Runtime에 연결하는 Python 바인딩도 빌드한다.
build_python_binding() {
    "$CMAKE_EXE" -S "$PROJECT_WINDOWS/cpp/pymnn" -B "$RUNTIME_WINDOWS/python" -G Ninja \
        "-DCMAKE_MAKE_PROGRAM=$NINJA_EXE" -DCMAKE_CXX_COMPILER=cl \
        -DCMAKE_BUILD_TYPE=Release "-DPython3_EXECUTABLE=$PYTHON_EXE" \
        "-DMNN_SOURCE_DIR=$RUNTIME_WINDOWS/MNN" "-DMNN_RUNTIME_DIR=$RUNTIME_WINDOWS/build" \
        2>&1 | "$TEE_EXE" "$RUNTIME_DIR/python-configure.log"
    "$CMAKE_EXE" --build "$RUNTIME_WINDOWS/python" --parallel "$BUILD_JOBS" \
        2>&1 | "$TEE_EXE" "$RUNTIME_DIR/python-build.log"
    echo "Python binding built: $RUNTIME_DIR/python"
}

check_inputs "$@"
prepare_msvc
prepare_source
build_runtime
if "$build_python"; then
    build_python_binding
fi
