#!/bin/bash

rm -rf docs/api
rm -rf docs/sphinx_src

pip show sphinx_immaterial || pip install sphinx_immaterial

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