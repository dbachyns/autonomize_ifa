#!/usr/bin/env bash
# Download the FUNSD dataset (199 scanned forms; non-commercial research license) into data/external/funsd.
# Only needed to re-extract or re-score FUNSD; the extracted results are already in data/cache_funsd.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/external/funsd
curl -fL -o data/external/funsd/dataset.zip https://guillaumejaume.github.io/FUNSD/dataset.zip
unzip -q -o data/external/funsd/dataset.zip -d data/external/funsd
echo "FUNSD ready in data/external/funsd/dataset (test split: testing_data/)"
