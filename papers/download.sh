#!/usr/bin/env bash
# Downloads every paper in papers.tsv into papers/<week>/<slug>.pdf
# and prints the arXiv title so you can sanity-check each ID.
# Usage: ./papers/download.sh   (needs network access to arxiv.org)
set -u
cd "$(dirname "$0")"
grep -v '^#' papers.tsv | while IFS=$'\t' read -r week slug id; do
  [ -z "${id:-}" ] && continue
  mkdir -p "$week"
  out="$week/$slug.pdf"
  [ -s "$out" ] && { echo "skip  $out"; continue; }
  if curl -fsSL -o "$out" "https://arxiv.org/pdf/$id"; then
    title=$(curl -fsSL "https://export.arxiv.org/api/query?id_list=$id" | tr '\n' ' ' | sed -n 's/.*<\/id> *<updated>[^<]*<\/updated> *<published>[^<]*<\/published> *<title>\([^<]*\)<\/title>.*/\1/p')
    echo "ok    $out  <- $id  \"$title\""
  else
    rm -f "$out"; echo "FAIL  $slug ($id)"
  fi
  sleep 3   # be polite to arXiv
done
