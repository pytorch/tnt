#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import pytorch_sphinx_theme
from sphinx.writers.html import HTMLTranslator
from torchtnt import __version__

sys.path.append(os.path.abspath("./ext"))

current_dir = os.path.dirname(__file__)
target_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, target_dir)
print(target_dir)

# -- Project information -----------------------------------------------------

project = "TorchTNT"
copyright = "2023, Meta"
author = "Meta"

# The full version, including alpha/beta/rc tags
if os.environ.get("RELEASE_BUILD", None):
    version = __version__
    release = __version__
else:
    version = "master (unstable)"
    release = "master"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

barebones = os.getenv("BAREBONES", None)
if barebones:
    extensions = ["fbcode"]
else:
    extensions = [
        "sphinx.ext.napoleon",
        "sphinx.ext.autodoc",
        "sphinx.ext.autosummary",
        "sphinx.ext.viewcode",
        "fbcode",
    ]

FBCODE = "fbcode" in os.getcwd()

if not FBCODE:
    extensions += [
        "sphinx.ext.intersphinx",
    ]
if FBCODE:
    nbsphinx_execute = "never"

html_context = {"fbcode": FBCODE}


# Add any paths that contain templates here, relative to this directory.
templates_path = ["templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_theme_options = {
    "display_version": True,
}
html_css_files = [
    "css/torchtnt.css",
]
html_js_files = [
    "js/torchtnt.js",
]
if FBCODE:
    # OSS uses sphinx v5 which bundles with jquery, internal is v6 which does not
    html_js_files.append("js/jquery.js")


class PatchedHTMLTranslator(HTMLTranslator):
    def visit_reference(self, node):
        if node.get("newtab") or not (
            node.get("target") or node.get("internal") or "refuri" not in node
        ):
            node["target"] = "_blank"
        super().visit_reference(node)


def setup(app):
    # NOTE: in Sphinx 1.8+ `html_css_files` is an official configuration value
    # and can be moved outside of this function (and the setup(app) function
    # can be deleted).

    # In Sphinx 1.8 it was renamed to `add_css_file`, 1.7 and prior it is
    # `add_stylesheet` (deprecated in 1.8).
    add_css = getattr(
        app, "add_css_file", getattr(app, "add_stylesheet", None)
    )  # noqa B009
    for css_file in html_css_files:
        add_css(css_file)
    app.set_translator("html", PatchedHTMLTranslator)


# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pytorch_sphinx_theme"
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]

# where to find external docs
intersphinx_mapping = {
    "torch": ("https://pytorch.org/docs/stable/", None),
}
