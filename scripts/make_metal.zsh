#!/bin/zsh

SDK_PATH=$(xcrun --show-sdk-path)
$CLANGXX -std=c++26 -O3 -stdlib=libc++ -fobjc-arc -x objective-c++ \
-isysroot $SDK_PATH \
-F$SDK_PATH/System/Library/Frameworks \
-Iinclude -Iutils/include \
examples/backprop/src/backprop_mnist.mm \
-framework Foundation -framework Metal -framework MetalPerformanceShaders \
-o build-release/bin/hnnet-backprop-mnist-metal

