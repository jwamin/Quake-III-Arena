#!/bin/sh
# Xcode "Build Engine" phase (runs before linking): builds the engine core as a
# static library (libquake3-engine.a) with code/unix/Makefile.sdl. The Metal +
# Swift platform layer (code/macos) is compiled by the app target itself.
set -eu

ENGINE_DIR="$DERIVED_FILE_DIR/engine"
case "$CONFIGURATION" in
  Debug) CFG=debug ;;
  *)     CFG=release ;;
esac

make -C "$SRCROOT/../../code/unix" -f Makefile.sdl \
     CONFIG="$CFG" BUILD="$ENGINE_DIR" \
     CC="xcrun clang -arch $ARCHS -mmacosx-version-min=$MACOSX_DEPLOYMENT_TARGET" AR="xcrun ar" \
     -j"$(sysctl -n hw.ncpu)" metal-enginelib
