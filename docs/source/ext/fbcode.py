#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

from docutils import nodes
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import nested_parse_with_titles


class FbcodeDirective(SphinxDirective):
    # this enables content in the directive
    has_content = True

    def run(self):
        if "fbcode" not in os.getcwd():
            return []
        node = nodes.section()
        node.document = self.state.document
        nested_parse_with_titles(self.state, self.content, node)
        return node.children


def setup(app):
    app.add_directive("fbcode", FbcodeDirective)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
