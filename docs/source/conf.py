# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import os
import sys

# source code is in project_root/python/ocr_stringdist
sys.path.insert(0, os.path.abspath("../../python"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "OCR-StringDist"
copyright = "2025, Niklas von Moers"
author = "Niklas von Moers"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions: list[str] = [
    "sphinx.ext.autodoc",  # Core library to pull documentation from docstrings
    "sphinx.ext.napoleon",  # Support for Google and NumPy style docstrings
    "sphinx.ext.intersphinx",  # Link to other projects' documentation
    "sphinx.ext.viewcode",  # Add links to source code
    "sphinx_mdinclude",  # Include Markdown
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path: list[str] = ["_static"]
