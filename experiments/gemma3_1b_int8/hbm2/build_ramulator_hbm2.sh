#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/../../.." && pwd)
vendor_dir="$repo_dir/submodules/ramulator2"
expected_pin=c6c7a5de0cef35935d6fad7369e6271406798865
if [[ ! -f "$vendor_dir/src/ramulator/dram/impl/HBM2.cpp" ]]; then
  echo "Missing Ramulator2; run $script_dir/setup_ramulator2.sh" >&2
  exit 1
fi
actual_pin=$(git -C "$vendor_dir" rev-parse HEAD)
if [[ "$actual_pin" != "$expected_pin" ]]; then
  echo "Unexpected Ramulator2 revision: $actual_pin (expected $expected_pin)" >&2
  exit 1
fi
cmake_args=(-DCMAKE_BUILD_TYPE=Release)
if [[ "$(uname -s)" == Darwin ]]; then
  default_arch=$(uname -m)
  if [[ "$(/usr/sbin/sysctl -n hw.optional.arm64 2>/dev/null || true)" == 1 ]]; then
    default_arch=arm64
  fi
  cmake_args+=("-DCMAKE_OSX_ARCHITECTURES=${RAMULATOR2_ARCH:-$default_arch}")
fi
cmake -S "$script_dir" -B "$script_dir/build" "${cmake_args[@]}"
cmake --build "$script_dir/build" --parallel 2
echo "$script_dir/build/ramulator_hbm2"
