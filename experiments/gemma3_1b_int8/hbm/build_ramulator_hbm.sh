#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/../../.." && pwd)
vendor_dir="$repo_dir/submodules/ramulator"
build_dir="$script_dir/build"
expected_pin=214f635845214adf030367939655d172ef0fed5f
if [[ ! -f "$vendor_dir/src/HBM.h" ]]; then
  echo 'Missing Ramulator source. Run: git submodule update --init submodules/ramulator' >&2
  exit 1
fi
actual_pin=$(git -C "$vendor_dir" rev-parse HEAD)
if [[ "$actual_pin" != "$expected_pin" ]]; then
  echo "Unexpected Ramulator revision: $actual_pin (expected $expected_pin)" >&2
  exit 1
fi
mkdir -p "$build_dir"
compiler=${CXX:-clang++}
sources=("$script_dir/ramulator_hbm.cpp" "$vendor_dir/src/Config.cpp"
         "$vendor_dir/src/HBM.cpp" "$vendor_dir/src/StatType.cpp")
binary="$build_dir/ramulator_hbm"
rebuild=false
[[ -f "$binary" ]] || rebuild=true
for input in "${sources[@]}" "${BASH_SOURCE[0]}" "$vendor_dir"/src/*.h; do
  [[ "$input" -nt "$binary" ]] && rebuild=true
done
if [[ "$rebuild" == false ]]; then
  echo "$binary"
  exit 0
fi
# One compiler invocation gives every translation unit the same architecture
# and flags, independent of the vendor Makefile's existing object files.
"$compiler" -O3 -std=c++11 -Wall -DRAMULATOR -isystem "$vendor_dir/src" \
  "${sources[@]}" -o "$binary"
echo "$build_dir/ramulator_hbm"
