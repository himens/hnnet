#!/bin/zsh

SDK_PATH=$(xcrun --show-sdk-path)
SOURCE=$1
EXEC=hnnet-${${SOURCE:t:r}//_/-}-metal

$CLANGXX -std=c++26 -O3 -stdlib=libc++ -fobjc-arc -x objective-c++ \
-isysroot $SDK_PATH \
-F$SDK_PATH/System/Library/Frameworks \
-Iinclude -Iutils/include $SOURCE \
-framework Foundation -framework Metal -framework MetalPerformanceShaders \
-o build-release/bin/$EXEC

