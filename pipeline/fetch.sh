#!/bin/sh
# Downloads the MaleCNS v1.0 flat-connectome tables into data/raw (inside Docker).
# Usage (from repo root): sh pipeline/fetch.sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p data/raw
BASE=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
docker run --rm --user 0 -v "$PWD/data/raw:/out" -w /out curlimages/curl:latest -sSfL --continue-at - \
  -o annotations.feather "$BASE/body-annotations-male-cns-v1.0-minconf-0.5.feather" \
  -o neurotransmitters.feather "$BASE/body-neurotransmitters-male-cns-v1.0.feather" \
  -o edges.feather "$BASE/connectome-weights-male-cns-v1.0-minconf-0.5.feather"
docker run --rm -v "$PWD/data/raw:/raw" -w /raw alpine:latest sha256sum annotations.feather neurotransmitters.feather edges.feather
