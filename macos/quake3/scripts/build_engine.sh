#!/bin/sh
# Xcode "Build Engine" phase (runs before linking).
#  1. builds the engine as a static library with code/unix/Makefile.sdl
#  2. stages the SDL dylibs the app links against and ships in Contents/Frameworks
# Homebrew's sdl2 is sdl2-compat, which loads SDL3 at runtime from
# @loader_path/libSDL3.dylib, so both libraries are needed.
set -eu
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

ENGINE_DIR="$DERIVED_FILE_DIR/engine"
case "$CONFIGURATION" in
  Debug) CFG=debug ;;
  *)     CFG=release ;;
esac

make -C "$SRCROOT/../../code/unix" -f Makefile.sdl \
     CONFIG="$CFG" BUILD="$ENGINE_DIR" \
     CC="xcrun clang -arch $ARCHS -mmacosx-version-min=$MACOSX_DEPLOYMENT_TARGET" AR="xcrun ar" \
     -j"$(sysctl -n hw.ncpu)" applib

SDL2_SRC="$(pkg-config --variable=libdir sdl2)/libSDL2-2.0.0.dylib"
SDL3_SRC="$(pkg-config --variable=libdir sdl3)/libSDL3.0.dylib"
FW="$ENGINE_DIR/Frameworks"
mkdir -p "$FW"

stage() { # stage <source> <name>
  src=$(realpath "$1")
  if [ ! -f "$FW/$2" ] || [ "$src" -nt "$FW/$2" ]; then
    cp -f "$src" "$FW/$2"
    chmod u+w "$FW/$2"
    install_name_tool -id "@rpath/$2" "$FW/$2" 2>/dev/null
  fi
}
stage "$SDL2_SRC" libSDL2-2.0.0.dylib
stage "$SDL3_SRC" libSDL3.dylib
