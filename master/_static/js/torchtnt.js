/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

const NETWORK_TEST_URL = 'https://staticdocs.thefacebook.com/ping';
fetch(NETWORK_TEST_URL).then(() => {
    $("#redirect-banner").prependTo("body").show();
});
