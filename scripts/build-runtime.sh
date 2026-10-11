#!/usr/bin/env bash
set -euo pipefail

readonly BUILD_JOBS="${BUILD_JOBS:-8}"

fail() {
    echo "$*" >&2
    exit 1
}

if [[ $# -ne 0 ]]; then
    fail "Usage: bash scripts/build-runtime.sh"
fi
if [[ ! "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    fail "BUILD_JOBS must be a positive integer."
fi
case "$(uname -s)" in
    MINGW*|MSYS*) ;;
    *) fail "Run this script in Git Bash on Windows." ;;
esac
for tool in cmake ninja cygpath cmd.exe dirname; do
    command -v "$tool" >/dev/null 2>&1 || fail "Required tool not found: $tool"
done

# MSVC 환경을 불러오기 전에 개발 도구 경로를 보관한다.
CMAKE_EXE=$(command -v cmake)
CTEST_EXE="$(dirname -- "$CMAKE_EXE")/ctest.exe"
CYGPATH_EXE=$(command -v cygpath)
NINJA_EXE=$("$CYGPATH_EXE" -m "$(command -v ninja)")
PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PROJECT_WINDOWS=$("$CYGPATH_EXE" -m "$PROJECT_DIR")
BUILD_DIR="$PROJECT_WINDOWS/build/windows-runtime"
readonly CMAKE_EXE CTEST_EXE CYGPATH_EXE NINJA_EXE PROJECT_DIR PROJECT_WINDOWS BUILD_DIR
[[ -f "$CTEST_EXE" ]] || fail "CTest not found next to CMake."

# 기존 MNN 빌드와 동일한 MSVC x64 환경을 사용한다.
vswhere=$("$CYGPATH_EXE" -u "${SYSTEMDRIVE:-C:}/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe")
[[ -f "$vswhere" ]] || fail "Install Visual Studio Build Tools with Desktop development with C++."
vs_install=$("$vswhere" -latest -products '*' \
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 \
    -property installationPath)
vs_install=${vs_install//$'\r'/}
[[ -n "$vs_install" ]] || fail "MSVC x64 build tools were not found."
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
export VSLANG=1033
command -v cl.exe >/dev/null 2>&1 || fail "MSVC compiler not found."
command -v rc.exe >/dev/null 2>&1 || fail "Windows SDK resource compiler not found."

"$CMAKE_EXE" -S "$PROJECT_WINDOWS/cpp" -B "$BUILD_DIR" -G Ninja \
    "-DCMAKE_MAKE_PROGRAM=$NINJA_EXE" -DCMAKE_CXX_COMPILER=cl \
    -DCMAKE_BUILD_TYPE=Release -DSD15_BUILD_CONVERTER=OFF \
    -DSD15_BUILD_RUNTIME=ON -DSD15_BUILD_CLI=ON -DBUILD_TESTING=ON \
    "-DMNN_SOURCE_DIR=$PROJECT_WINDOWS/MNNRuntime/MNN" \
    "-DMNN_RUNTIME_DIR=$PROJECT_WINDOWS/MNNRuntime/build"
"$CMAKE_EXE" --build "$BUILD_DIR" --parallel "$BUILD_JOBS"
"$CTEST_EXE" --test-dir "$BUILD_DIR" --output-on-failure
