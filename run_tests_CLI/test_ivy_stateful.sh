#!/bin/bash -e
docker run --rm -v "$(pwd)":/ivy -v "$(pwd)"/.hypothesis:/.hypothesis unifyai/ivy:latest python3 -m pytest --backend "$1" ivy/ivy_tests/test_ivy/test_stateful/"$2".py