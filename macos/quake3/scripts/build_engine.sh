#!/bin/sh
# Xcode "Build Engine" phase (runs before linking): builds the engine with its
# Metal + Swift platform layer (code/macos) as a static library, using
# code/unix/Makefile.sdl. The app links it; nothing else needs to be embedded.
set -eu

ENGINE_DIR="$DERIVED_FILE_DIR/engine"
case "$CONFIGURATION" in
  Debug) CFG=debug ;;
  *)     CFG=release ;;
esac

make -C "$SRCROOT/../../code/unix" -f Makefile.sdl \
     CONFIG="$CFG" BUILD="$ENGINE_DIR" \
     CC="xcrun clang -arch $ARCHS -mmacosx-version-min=$MACOSX_DEPLOYMENT_TARGET" AR="xcrun ar" \
     SWIFTC="xcrun swiftc -target $ARCHS-apple-macos$MACOSX_DEPLOYMENT_TARGET" \
     -j"$(sysctl -n hw.ncpu)" metal-applib
