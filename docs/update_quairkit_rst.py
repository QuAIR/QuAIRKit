"""
How to use?
Run this python file on the root directory of this repository (Attention: Not its subdirectory) and then all files in source will be generated.
"""

import os
import sys
from typing import List, Tuple

_platform_name = "QuAIRKit"

# set the module names you want to include in API documentation in _module_names
_module_names = ["quairkit", "tutorials"]

# set the file name you want to ignore in API documentation in _ignore_file_names
_ignore_file_names = ["quairkit.core.utils", "quairkit.core.intrinsic"]

# source directory for sphinx
_sphinx_source_dir = os.path.join(".", "docs", "sphinx_src")

def is_correct_directory():
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    required_file = os.path.join(script_dir, 'docs', 'quairkit')
    return os.path.exists(required_file)

def _list_quairkit_files(
    path: str = os.path.join("."),
    base_path: str = "",
    file_name_attr_list: List[Tuple[str, str]] = None,
) -> List[Tuple[str, str]]:
    """
    List files and folders in the given directory recursively.

    Args:
        path (str): The directory path to start listing from.
        base_path (str, optional): Base path for relative path calculation. Defaults to an empty string.
        result (List[Tuple[str, str]], optional): A list to store the result. Defaults to None.

    Returns:
        List[Tuple[str, str]]: A list of tuples where each tuple contains the relative path and the type (folder or python file).
    """
    if file_name_attr_list is None:
        file_name_attr_list = []

    for child in os.listdir(path):
        if child.startswith("__"):
            continue
        child_path = os.path.join(path, child)
        relative_path = os.path.join(base_path, child).replace(os.path.sep, ".")

        if os.path.isdir(child_path):
            if sub_list := _list_quairkit_files(child_path, relative_path, []):
                file_name_attr_list.append((f"{relative_path}", "folder"))
                file_name_attr_list.extend(sub_list)
        elif child.endswith(".py"):
            file_name_attr_list.append(
                (f"{relative_path.rstrip('.py')}", "python")
            )
        # elif child.endswith(".ipynb"):
        #     file_name_attr_list.append(
        #         (f"{relative_path.rstrip('.ipynb')}", "notebook")
        #     )

    file_name_attr_list = [
        sub_array
        for sub_array in file_name_attr_list
        if any(
            sub_array[0].startswith(module_name) for module_name in _module_names
        )
    ]

    file_name_attr_list = [
        sub_array
        for sub_array in file_name_attr_list
        if not any(
            sub_array[0].startswith(ignore_item) for ignore_item in _ignore_file_names
        )
    ]

    file_name_attr_list.sort()

    return file_name_attr_list


def _update_function_rst(
    file_list: List[Tuple[str, str]],
    source_directory: str = _sphinx_source_dir,
) -> None:
    """
    Create .rst files in the test directory based on the file list.

    Args:
        file_list (List[Tuple[str, str]]): A list of tuples where each tuple contains the relative path and the type (folder or python file).

        source_directory (str): The directory where .rst files will be created.
    """

    for file_name, file_type in file_list:
        rst_content = ""

        # .. title:: Your Title Here
        rst_content += f"{file_name}\n"
        rst_content += "=" * len(file_name) + "\n\n"
        rst_content += f".. automodule:: {file_name}\n"
        rst_content += "\t:members:\n\n"

        if file_type == "folder":
            rst_content += ".. toctree::\n    :maxdepth: 2\n\n"

            py_files = [
                item[0]
                for item in file_list
                if item[0].startswith(file_name)
                and item[0].count(".") == file_name.count(".") + 1
            ]
            for subfolder in py_files:
                rst_content += f"    {subfolder}\n"

        file_path = os.path.join(source_directory, f"{file_name}.rst")

        with open(file_path, "w") as file:
            file.write(rst_content)
    
    rst_content = \
"""\
tutorials
=========

.. toctree::
    :maxdepth: 4
    :glob:

    tutorials/*
"""

    file_path = os.path.join(source_directory, "tutorials.rst")

    with open(file_path, "w") as file:
        file.write(rst_content)

    return


def _update_index_rst(
    file_list: List[Tuple[str, str]], source_directory: str = _sphinx_source_dir
) -> None:
    """
    Args:
        source_directory (str, optional): Defaults to source_dir. The directory where .rst files will be created.
    """
    rst_content = ""
    if len(sys.argv) == 1:
        rst_content += f"""\
.. |quairkit| unicode:: U+1F951

Welcome to {_platform_name}'s documentation!
====================================

|quairkit| `Go to QuAIR-Platform Home <https://www.quairkit.com/>`_

"""

    rst_content += """\
.. toctree::
    :maxdepth: 1
"""
    rst_content += "".join(f"\n    {item}" for item in _module_names)
    file_path = os.path.join(source_directory, "index.rst")

    with open(file_path, "w") as file:
        file.write(rst_content)


def _update_conf_py(source_directory: str = _sphinx_source_dir):
    """_summary_

    Args:
        source_directory (str, optional): _description_. Defaults to source_dir.
    """
    rst_content = f"""\
# Configuration file for the Sphinx documentation builder.
#
# For a full list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.join('..', '..'))
# -- Project information -----------------------------------------------------

project = "{_platform_name}"
copyright = "2024, QuAIR"
author = "QuAIR"

# The full version, including alpha/beta/rc tags
release = "0.1.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx_immaterial",
    "nbsphinx",
"""
    if len(sys.argv) == 2 and sys.argv[1] == "wiki":
        rst_content += """\
    "sphinxcontrib.restbuilder",
]
# rst files for Github WiKi
rst_link_suffix = ""
"""
    else:
        rst_content += """\
]
"""

    rst_content += """\
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
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
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

master_doc = "index"

# Autodoc
napoleon_numpy_docstring = False
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_warningiserror = False
autodoc_inherit_docstrings = False
autodoc_docstring_signature = False
autodoc_typehints_description_target = "documented"
autodoc_typehints_format = "short"


"""

    file_path = os.path.join(source_directory, "conf.py")
    with open(file_path, "w") as file:
        file.write(rst_content)


def _create_redirect_html():
    html_content = """
<!DOCTYPE HTML>
<html lang="en">
    <head>
        <meta charset="utf-8">
        <meta http-equiv="refresh" content="0; url=latest/index.html" />
        <link rel="canonical" href="latest/index.html" />
    </head>
    <body>
        <p>If this page does not refresh automatically, then please direct your browser to
            <a href="latest/index.html">our latest docs</a>.
        </p>
    </body>
</html>
"""
    file_dir = os.path.join(".", "docs", "api")
    os.makedirs(file_dir, exist_ok=True)
    with open(os.path.join(file_dir, "index.html"), 'w', encoding='utf-8') as file:
        file.write(html_content)


if __name__ == "__main__":
    _current_script_path = os.path.abspath(__file__)
    _platform_dir_path = os.path.dirname(os.path.dirname(_current_script_path))

    _current_working_dir = os.getcwd()
    if _current_working_dir != _platform_dir_path:
        raise SystemExit(f"The current working directory is not {_platform_dir_path}.")
    result = _list_quairkit_files()
    os.makedirs(_sphinx_source_dir, exist_ok=True)
    _update_index_rst(result)
    _update_function_rst(result)
    _update_conf_py()
    _create_redirect_html()
