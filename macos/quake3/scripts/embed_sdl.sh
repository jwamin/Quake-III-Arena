#!/bin/sh
# Xcode "Embed SDL" phase: copy the staged SDL dylibs into Contents/Frameworks
# and sign them with the app's identity (hardened runtime library validation).
set -eu
SRC="$DERIVED_FILE_DIR/engine/Frameworks"
DST="$TARGET_BUILD_DIR/$FRAMEWORKS_FOLDER_PATH"
IDENTITY="${EXPANDED_CODE_SIGN_IDENTITY:--}"
mkdir -p "$DST"
for lib in libSDL2-2.0.0.dylib libSDL3.dylib; do
  cp -f "$SRC/$lib" "$DST/$lib"
  codesign --force --sign "$IDENTITY" --options runtime --timestamp=none "$DST/$lib"
done
