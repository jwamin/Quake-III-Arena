#!/bin/sh
# Xcode "Build UI pk3" phase: compiles the baseq3 menus (code/q3_ui) to
# ui.qvm with the tree's own lcc/q3asm and packs it as zz_metal_ui.pk3 in the
# app's Resources. AppDelegate copies it into the Quake3 home folder at launch,
# where it overrides the retail ui.qvm (adds the Metal options to
# System > Graphics).
set -eu

# Host tools (lcc, q3asm) are built for this Mac, not with the app's
# SDK/deployment settings.
unset SDKROOT MACOSX_DEPLOYMENT_TARGET CFLAGS LDFLAGS

# Keep the lcc build between builds; only the UI module is recompiled.
export WORK="$PROJECT_TEMP_DIR/qvm"
export MODULES="ui"
export PK3="$TARGET_BUILD_DIR/$UNLOCALIZED_RESOURCES_FOLDER_PATH/zz_metal_ui.pk3"

CODE="$SRCROOT/../../code"
# up to date: nothing the UI is built from is newer than the pk3
if [ -f "$PK3" ] && [ -z "$(find "$CODE/q3_ui" "$CODE/game" "$CODE/ui" "$CODE/unix/build_qvms.sh" \
        -newer "$PK3" -print -quit)" ]; then
    echo "zz_metal_ui.pk3 is up to date"
    exit 0
fi

"$CODE/unix/build_qvms.sh"
