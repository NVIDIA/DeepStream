#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Functions only: sourcing this file never installs, copies or loads a library.
# ELF metadata checks establish provider shape, not VPI runtime qualification.

ds_vpi_validate_x86_headers() {
  local root=$1 version compiler=${CC:-cc}
  if [[ ! -r "$root/include/vpi/VersionDef.h" ||
        ! -r "$root/include/vpi/algo/DCFTracker.h" ]]; then
    echo "error: x86 requires vendor VPI4.1.4 development headers at $root." >&2
    return 1
  fi
  version=$(printf '%s\n' 'NV_VPI_VERSION_MAJOR NV_VPI_VERSION_MINOR NV_VPI_VERSION_PATCH' |
    "$compiler" -E -P -x c -I"$root/include" -include "$root/include/vpi/VersionDef.h" -) || return 1
  version=$(printf '%s\n' "$version" | xargs | tr ' ' '.')
  if [[ "$version" != 4.1.4 ]]; then
    echo "error: x86 requires VPI4.1.4 headers; selected $version at $root." >&2
    return 1
  fi
}

ds_vpi_validate_x86_runtime() {
  local directory=$1 canonical runtime header dynamic
  canonical=$(readlink -e -- "$directory") || {
    echo "error: missing x86 VPI runtime directory: $directory" >&2; return 1;
  }
  runtime=$(readlink -e -- "$canonical/libnvvpi.so.4") || {
    echo "error: missing x86 VPI4.1.4 runtime alias in $directory." >&2; return 1;
  }
  if [[ "$runtime" != "$canonical/libnvvpi.so.4.1.4" || ! -f "$runtime" ]]; then
    echo "error: x86 VPI alias must resolve within $directory to libnvvpi.so.4.1.4." >&2
    return 1
  fi
  header=$(LC_ALL=C readelf -h -- "$runtime") || return 1
  dynamic=$(LC_ALL=C readelf -d -- "$runtime") || return 1
  if ! grep -Eq 'Class:[[:space:]]+ELF64' <<<"$header" ||
     ! grep -Eq 'Type:[[:space:]]+DYN' <<<"$header" ||
     ! grep -Eq 'Machine:[[:space:]]+Advanced Micro Devices X86-64' <<<"$header" ||
     ! grep -Eq '\(SONAME\).*\[libnvvpi\.so\.4\]' <<<"$dynamic"; then
    echo "error: $runtime must be an ELF64 x86-64 shared library with SONAME libnvvpi.so.4." >&2
    return 1
  fi
  printf '%s\n' "$runtime"
}

ds_vpi_validate_x86_bundle() {
  local directory=$1 runtime alias target
  runtime=$(ds_vpi_validate_x86_runtime "$directory") || return 1
  if [[ ! -L "$directory/libnvvpi.so.4" ||
        "$(readlink -- "$directory/libnvvpi.so.4")" != libnvvpi.so.4.1.4 ]]; then
    echo "error: SDK VPI alias must be relative: libnvvpi.so.4 -> libnvvpi.so.4.1.4." >&2
    return 1
  fi
  if [[ -e "$directory/libnvvpi.so" || -L "$directory/libnvvpi.so" ]]; then
    target=$(readlink -- "$directory/libnvvpi.so") || return 1
    alias=$(readlink -e -- "$directory/libnvvpi.so") || return 1
    if [[ ( "$target" != libnvvpi.so.4 && "$target" != libnvvpi.so.4.1.4 ) ||
          "$alias" != "$runtime" ]]; then
      echo "error: the optional SDK libnvvpi.so alias must be relative and select VPI4.1.4." >&2
      return 1
    fi
  fi
}

ds_vpi_cleanup_stage() {
  local directory=$1
  rm -f -- "$directory/libnvvpi.so.4.1.4" "$directory/libnvvpi.so.4" "$directory/libnvvpi.so"
  rmdir -- "$directory"
}

ds_vpi_stage_x86_runtime() {
  local vendor=$1 destination=$2 runtime temporary update_unversioned=0
  runtime=$(ds_vpi_validate_x86_runtime "$vendor") || return 1
  mkdir -p -- "$destination" || return 1
  if [[ -e "$destination/libnvvpi.so" || -L "$destination/libnvvpi.so" ]]; then
    update_unversioned=1
  fi
  temporary=$(mktemp -d "$destination/.ds-vpi-stage.XXXXXX") || return 1
  # Publish a complete real file first; change the loader alias only afterwards.
  # Do not delete the old regular 4.1.3 file: an input package may still own it.
  if ! cp -p -- "$runtime" "$temporary/libnvvpi.so.4.1.4" ||
     ! ln -s libnvvpi.so.4.1.4 "$temporary/libnvvpi.so.4" ||
     ! ds_vpi_validate_x86_runtime "$temporary" >/dev/null ||
     ! mv -fT -- "$temporary/libnvvpi.so.4.1.4" "$destination/libnvvpi.so.4.1.4"; then
    ds_vpi_cleanup_stage "$temporary"
    echo "error: failed to stage x86 VPI4.1.4; SDK loader aliases were not changed." >&2
    return 1
  fi
  if [[ "$update_unversioned" -eq 1 ]]; then
    if ! ln -s libnvvpi.so.4 "$temporary/libnvvpi.so" ||
       ! mv -fT -- "$temporary/libnvvpi.so" "$destination/libnvvpi.so"; then
      ds_vpi_cleanup_stage "$temporary"
      return 1
    fi
  fi
  if ! mv -fT -- "$temporary/libnvvpi.so.4" "$destination/libnvvpi.so.4"; then
    ds_vpi_cleanup_stage "$temporary"
    return 1
  fi
  ds_vpi_cleanup_stage "$temporary" || return 1
  ds_vpi_validate_x86_bundle "$destination"
}
