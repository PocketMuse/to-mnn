#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
    echo "Usage: bash scripts/lint-cpp.sh [format-check|format|tidy]" >&2
    exit 2
fi
mode="${1:-format-check}"
case "$mode" in
    format-check|format|tidy) ;;
    *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# Runtime과 외부 소스는 대상에 포함하지 않는다.
mapfile -d '' files < <(find cpp/include cpp/src cpp/tests -type f \
    \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' \) -print0 | sort -z)
if [[ ${#files[@]} -eq 0 ]]; then
    echo "No converter C++ files found" >&2
    exit 1
fi

case "$mode" in
    format-check)
        clang-format-14 --dry-run --Werror "${files[@]}"
        ;;
    format)
        clang-format-14 -i "${files[@]}"
        ;;
    tidy)
        build_dir=$(mktemp -d /tmp/to-mnn-lint.XXXXXX)
        trap 'rm -rf -- "$build_dir"' EXIT
        cmake -S cpp -B "$build_dir" -DCMAKE_BUILD_TYPE=Debug \
            -DBUILD_TESTING=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
            -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON
        sources=()
        for file in "${files[@]}"; do
            if [[ "$file" == *.cpp ]]; then
                sources+=("$file")
            fi
        done
        # 성공 시 외부 헤더의 억제된 경고 통계는 숨기고, 실패 시 전체 진단을 표시한다.
        if ! clang-tidy-14 --quiet -p "$build_dir" "${sources[@]}" > "$build_dir/tidy.log" 2>&1; then
            cat "$build_dir/tidy.log"
            exit 1
        fi
        echo "clang-tidy: ${#sources[@]} translation units passed"
        ;;
esac
