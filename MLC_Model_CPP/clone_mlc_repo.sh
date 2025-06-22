#!/bin/bash

git clone git@github.com:mlc-ai/mlc-llm.git
cd mlc-llm
git checkout a175d44
git submodule update --init --recursive