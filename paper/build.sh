#!/bin/bash
# Build ACC paper with tectonic (fast, single-binary LaTeX engine)
cd "$(dirname "$0")"
tectonic main.tex
echo "Done: main.pdf"
