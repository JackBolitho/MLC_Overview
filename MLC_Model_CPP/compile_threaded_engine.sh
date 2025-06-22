#!/bin/bash

mkdir build
cd build
cmake -DUSE_METAL=ON ../CMakeLists.txt
cmake --build . -j16
cd ..
