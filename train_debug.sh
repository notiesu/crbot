#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 PATH_TO_DIRECTORY" >&2
    exit 2
fi

src="$1"
if [[ ! -d "$src" ]]; then
    echo "Not a directory: $src" >&2
    exit 3
fi

dest="$PWD/clash-simulator/train"
dest_root="$PWD/clash-simulator"
name="$(basename "$src")"

if [[ -e "$dest" ]]; then
    rm -rf "$dest"
fi

cp -r "$src" "$dest"
cd "$dest_root"
python -m train.train