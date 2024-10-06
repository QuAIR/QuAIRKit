# Configuration file for the Sphinx documentation builder.
#
# For a full list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.join('..', '..'))

# -- Project information -----------------------------------------------------

project = "QuAIRKit"
copyright = "2024, QuAIR"
author = "QuAIR"

# The full version, including alpha/beta/rc tags
release = "0.2.0"


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx_immaterial",
    "nbsphinx",
]
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_immaterial"
html_title = "QuAIRKit"
html_short_title = "QuAIRKit"
build_dir = "api"
html_theme_options = {
    "repo_url": 'https://github.com/QuAIR/QuAIRKit',
    "repo_name": 'QuAIRKit',
    "palette": { "primary": "green" },
    "version_dropdown": True,
}
html_favicon = '../favicon.svg'

master_doc = "index"

# Autodoc configurations
napoleon_numpy_docstring = False
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_warningiserror = False
autodoc_inherit_docstrings = False
autodoc_docstring_signature = False
autodoc_typehints_description_target = "documented"
autodoc_typehints_format = "short"
html_theme_options['version_info'] = [{'version': 'latest', 'title': 'latest', 'aliases': ['latest']}, {'version': 'v0.1.0', 'title': 'v0.1.0', 'aliases': ['v0.1.0']}, {'version': 'v0.2.0', 'title': 'v0.2.0', 'aliases': ['v0.2.0']}]
