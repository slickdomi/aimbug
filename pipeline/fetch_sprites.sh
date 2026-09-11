#!/bin/sh
# Downloads the female Drosophila photos used as targets (Wikimedia Commons) into data/raw/sprites.
# Then: docker run ... aimbug-pipeline python pipeline/build_sprites.py
set -eu
cd "$(dirname "$0")/.."
mkdir -p data/raw/sprites
docker run --rm --user 0 -v "$PWD/data/raw/sprites:/out" -w /out curlimages/curl:latest -sSfL -A "aimbug-sprite-fetch/0.1" \
  -o brecher_female_side.jpg "https://upload.wikimedia.org/wikipedia/commons/0/03/Drosophila_melanogaster_%E2%99%80_%2838978426500%29.jpg" \
  -o davis_female_standing.jpg "https://upload.wikimedia.org/wikipedia/commons/7/70/Standing_female_Drosophila_melanogaster.jpg"
