#!/usr/bin/env bash
set -e

PYCMD=${PYCMD:="python"}
if [ "$1" == "coverage" ];
then
    coverage erase
    PYCMD="coverage run --parallel-mode --source torch "
    echo "coverage flag found. Setting python command to: \"$PYCMD\""
fi

pushd "$(dirname "$0")"

$PYCMD test_datasets.py
$PYCMD test_meters.py
$PYCMD test_transforms.py
