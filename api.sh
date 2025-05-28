#!/bin/bash

if ! command -v pandoc > /dev/null; then
  echo "Pandoc is not installed. Please visit https://github.com/jgm/pandoc/releases/ to download and install Pandoc before proceeding."
  echo "After installation, please restart your terminal or system to ensure Pandoc is available in the PATH."
  exit 1
fi

rm -rf docs/api
rm -rf docs/sphinx_src

pip show sphinx_immaterial || pip install sphinx_immaterial
pip show nbsphinx || pip install nbsphinx

python docs/update_quairkit_rst.py
cp -r tutorials docs/sphinx_src/tutorials

sphinx-build -b html docs/sphinx_src docs/api

if command -v xdg-open > /dev/null; then
  xdg-open docs/api/index.html
elif command -v open > /dev/null; then
  open docs/api/index.html
elif command -v start > /dev/null; then
  start docs/api/index.html
else
  echo "Could not detect the web browser command to open the generated HTML file."
fi