#!/usr/bin/env bash
# Build main.pdf with the user-local TeX Live installation.
set -e
cd "$(dirname "$0")"
TEX=/export/data/rheinric/texlive/2026/bin/x86_64-linux
"$TEX/pdflatex" -interaction=nonstopmode main.tex >/dev/null || true
"$TEX/bibtex" main >/dev/null || true
"$TEX/pdflatex" -interaction=nonstopmode main.tex >/dev/null || true
"$TEX/pdflatex" -interaction=nonstopmode main.tex >/dev/null || true
grep -nE "^!|undefined|Overfull|Font Warning" main.log || echo "clean build"
grep -o "Output written.*" main.log
