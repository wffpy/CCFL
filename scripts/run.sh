#!/bin/bash
set -e
cd $(dirname $0)/..
pwd
mkdir -p build
cd build
cmake   -DNCCL_LIBRARY_DIR=/usr/lib/x86_64-linux-gnu -DNCCL_INCLUDE_DIR=/usr/include/ ..
make -j
#./ccf_test 
