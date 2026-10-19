#!/usr/bin/env bash
# Build the pinned, genuine HBM2-capable Ramulator 2.1 C++ library.
set -euo pipefail
script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(CDPATH= cd -- "$script_dir/../../.." && pwd)
source_dir="$repo_dir/submodules/ramulator2"
build_dir="${RAMULATOR2_BUILD_DIR:-$source_dir/build}"
revision=$(tr -d '[:space:]' < "$script_dir/ramulator2_revision.txt")
portability_patch="$script_dir/patches/0001-clang-dependent-template.patch"
jobs="${RAMULATOR2_JOBS:-2}"
case "$jobs" in ''|*[!0-9]*) echo 'RAMULATOR2_JOBS must be a positive integer' >&2; exit 2;; esac
if [ "$jobs" -lt 1 ]; then echo 'RAMULATOR2_JOBS must be positive' >&2; exit 2; fi

if [ ! -e "$source_dir/.git" ]; then
  git clone --no-checkout https://github.com/CMU-SAFARI/ramulator2.git "$source_dir"
fi
if ! git -C "$source_dir" cat-file -e "$revision^{commit}" 2>/dev/null; then
  git -C "$source_dir" fetch origin "$revision"
fi
if ! git -C "$source_dir" diff --cached --quiet; then
  echo "Refusing staged modifications in $source_dir" >&2
  exit 1
fi
if git -C "$source_dir" diff --quiet; then
  git -C "$source_dir" checkout --detach "$revision"
  git -C "$source_dir" apply "$portability_patch"
else
  # Accept only our exact recorded patch on the pinned commit. Never overwrite
  # other source edits, including additional edits in the patched header.
  if [ "$(git -C "$source_dir" rev-parse HEAD)" != "$revision" ] ||
     ! git -C "$source_dir" diff --binary | cmp -s - "$portability_patch"; then
    echo "Refusing unrecognized tracked modifications in $source_dir" >&2
    exit 1
  fi
fi

# Keep fetched vendor and generated macOS library out of working-tree changes.
parent_exclude=$(git -C "$repo_dir" rev-parse --git-path info/exclude)
case "$parent_exclude" in /*) ;; *) parent_exclude="$repo_dir/$parent_exclude";; esac
mkdir -p "$(dirname -- "$parent_exclude")"
touch "$parent_exclude"
if ! grep -Fxq '/submodules/ramulator2/' "$parent_exclude"; then
  printf '\n/submodules/ramulator2/\n' >> "$parent_exclude"
fi
vendor_exclude=$(git -C "$source_dir" rev-parse --git-path info/exclude)
case "$vendor_exclude" in /*) ;; *) vendor_exclude="$source_dir/$vendor_exclude";; esac
mkdir -p "$(dirname -- "$vendor_exclude")"
touch "$vendor_exclude"
if ! grep -Fxq '/libramulator.dylib' "$vendor_exclude"; then
  printf '\n/libramulator.dylib\n' >> "$vendor_exclude"
fi

cmake_args=(-DCMAKE_BUILD_TYPE=Release -DRAMULATOR_PYTHON_BINDINGS=OFF)
if [ "$(uname -s)" = Darwin ]; then
  # Intel-installed CMake may otherwise cross-build x86_64 on Apple Silicon.
  default_arch=$(uname -m)
  if [ "$(/usr/sbin/sysctl -n hw.optional.arm64 2>/dev/null || true)" = 1 ]; then
    default_arch=arm64
  fi
  target_arch="${RAMULATOR2_ARCH:-$default_arch}"
  cmake_args+=("-DCMAKE_OSX_ARCHITECTURES=$target_arch")
fi
cmake -S "$source_dir" -B "$build_dir" "${cmake_args[@]}"
cmake --build "$build_dir" --target ramulator --parallel "$jobs"
printf '\nBuilt Ramulator 2.1 at %s\n' "$revision"
printf 'Applied portability patch: %s\n' "$portability_patch"
printf 'Library directory: %s\n' "$source_dir"
