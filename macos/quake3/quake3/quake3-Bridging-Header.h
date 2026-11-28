// Engine entry points from code/unix/unix_main.c (built with -DQ3_APP).

// Runs the engine with Quake-style arguments. Does not return; the engine
// exits the process when it quits.
int Q3_Main(int argc, char *argv[]);

// Queues a "quit" command; the engine shuts down at the next frame.
void Q3_RequestQuit(void);

// The Metal + Swift platform layer (code/macos, compiled in this target):
// shared C types and the MacQ_* engine callbacks its Swift files call.
#include "../../../code/macos/mtl_bridge.h"
