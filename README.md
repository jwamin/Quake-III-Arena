# Quake III Arena — 64-bit Linux / macOS SDL2 port

A fork of [id-Software/Quake-III-Arena](https://github.com/id-Software/Quake-III-Arena)
(the 2005 GPL source release, version 1.32b) that builds and runs natively on a
current 64-bit Linux system. It was developed and tested on Fedora (x86_64,
gcc 16, Wayland, Mesa), and also builds and runs on macOS (Apple Silicon).

The original release only builds as 32-bit i386 and depends on things that no
longer exist on modern systems: `/dev/dsp` (OSS), XFree86 DGA/VidMode, nasm
assembly, and a JIT that assumes 32-bit pointers. This fork replaces those
with an SDL2 platform layer and fixes the 64-bit bugs. Gameplay code is
unchanged.

**No game data is included.** You need your own copy of Quake III Arena
(`pak0.pk3` and friends). The free demo works for the maps it contains.

## Status

Tested: client starts, menus, single player and bot matches on retail maps,
Team Arena (`fs_game missionpack`) loads, keyboard and mouse input, windowed
and fullscreen, dedicated server.

Not tested or not available:

- Sound opens a device through SDL, but has not been verified by ear.
- Hardware gamma is not available on Wayland (the game falls back to
  software overbright).
- No JIT: game logic runs in the QVM interpreter (fine for this game).
- No SMP renderer, no 3Dfx, no FreeType font generation (`ft2/` is not part of
  the release; fonts are prebuilt in the paks).
- Internet play and the original master server are not tested.

## Platforms

| Platform | Status |
|---|---|
| Linux x86_64 | built and played (gcc and clang) |
| Linux arm64 | builds in CI (not run) |
| macOS Apple Silicon | SDL2/OpenGL build from the makefile; Metal + Swift build from the makefile and as a sandboxed app (`macos/quake3`); maps, menus, cinematics and `vid_restart` checked with screenshots, input and sound not yet checked by hand; no CI job |
| macOS Intel | untested |
| FreeBSD, others with SDL2 | untested |
| Windows | not supported yet |

The engine is C-only (no JIT, no assembly) and 64-bit clean, so there is
nothing CPU-specific. Everything platform-dependent lives in the SDL2 layer
(`code/unix/sdl_glimp.c`, `sdl_snd.c`), or on macOS, optionally, in the
Metal + Swift layer (`code/macos`, see below).

## Building

Requirements: a C compiler (gcc or clang), make, SDL2 development files
(`SDL2-devel` on Fedora, `libsdl2-dev` on Debian/Ubuntu, `brew install sdl2`
on macOS), and python3 for the QVM script.

```sh
cd code/unix
make -f Makefile.sdl -j8                 # client + dedicated server
make -f Makefile.sdl CC=clang -j8        # another compiler
make -f Makefile.sdl CONFIG=debug -j8    # debug build
```

Output goes to `build/<os>-<arch>-<config>/` (`quake3` and `q3ded`). Delete
that directory to clean. Header dependencies are tracked, so rebuilds are
correct.

### macOS: Metal + Swift

On macOS the engine can also run without SDL and without OpenGL:

```sh
cd code/unix
make -f Makefile.sdl -j8 metal           # build/darwin-arm64-release/quake3-metal
                                         # and default.metallib beside it
```

Requirements: Xcode (clang, swiftc and the Metal toolchain; if `xcrun metal`
is missing, install it with `xcodebuild -downloadComponent MetalToolchain`).
Keep `default.metallib` next to `quake3-metal`; the renderer loads its
shaders from there.

`code/macos` holds the platform layer:

| File | |
|---|---|
| `Platform.swift` | AppKit window with a `CAMetalLayer`, event pump, keyboard, relative mouse |
| `MetalRenderer.swift` | Metal backend: pipeline/depth-stencil state caches, per-frame vertex arenas, textures, offscreen frame, present pass with the gamma ramp, read-back for screenshots |
| `Shaders.metal` | shaders for the fixed-function pipeline the renderer expects (vertex colour, two texture units with GL texture environments, alpha test, user clip plane) |
| `Audio.swift` | sound output through an `AVAudioSourceNode` |
| `mtl_qgl.c` | the renderer's `qgl*` calls (state, matrices, client arrays, immediate mode) implemented on top of the Metal backend; with `r_renderer opengl`, the same pointers bound to Apple's OpenGL framework |
| `mtl_rt.c` | ray tracing hooks: world acceleration structure, ray traced dynamic lights (`docs/metal-rt.md`) |
| `mac_glimp.c`, `mac_snd.c` | `GLimp_*`, `IN_*` and `SNDDMA_*` for the engine, key mapping |
| `mtl_bridge.h`, `mtl_swift.h`, `mac_gl.h` | the C/Swift interface, and the GL types and tokens the renderer source uses |

The renderer itself (`code/renderer`) is unchanged: it still issues its
OpenGL 1.x call sequence, and `mtl_qgl.c` turns each draw into one Metal draw
with the matching pipeline state. The frame is drawn at the engine resolution
(`r_mode`) into an offscreen target and scaled to the window, keeping the
aspect ratio. Gamma and overbright lighting are applied when the frame is
presented, so they also work in a window (the GL build turns overbright off
when windowed).

`quake3-metal` (and the app) can also render with OpenGL: `r_renderer
opengl` (or System Setup > METAL > Renderer, or the Renderer row on the
Graphics page), applied with `vid_restart`. The window then holds an OpenGL
(legacy profile) view instead of the Metal layer; MetalFX and ray tracing are
off, and there is no hardware gamma, so the renderer brightens in software
as Q3 does without gamma (darker than Metal).

The menus that come with the build (`zz_metal_ui.pk3`, see below) add two
System Setup tabs in the Metal build: **METAL** (renderer, MetalFX upscaling,
render scale; Accept restarts video) and **RAY TRACING** (ray traced lights,
clay mode; applied immediately, greyed out unless the Metal renderer runs).
The ray tracing and upscaling work is described in `docs/metal-rt.md`.

Not implemented: SMP renderer, joysticks/game controllers, clipboard paste.
`r_measureOverdraw` and light flares (`r_flares 1`) read the stencil and
depth buffers back, which stalls the frame (as on GL); overdraw is checked,
flares run but have not been checked by eye. The cloud-layer sky on q3dm17 is
black in both the GL and the Metal build (an existing engine bug, not yet
investigated); skyboxes such as q3dm1's draw.

On an M4, the `four` timedemo at 1024x768 runs at about 520 fps with Metal
and 573 fps with the SDL/OpenGL build.

### macOS app

`macos/quake3/quake3.xcodeproj` wraps the Metal + Swift engine in a
sandboxed, signed `quake3.app` (Apple Silicon only). It needs Xcode, nothing
else.

```sh
cd macos/quake3
xcodebuild -scheme quake3 -configuration Debug build   # or build and run in Xcode
```

The app target compiles the Metal + Swift layer itself: `code/macos` is in
the project as the "Metal port (code/macos)" folder group, so its Swift and C
files can be edited and debugged in Xcode. A "Build Engine" script phase runs
`make -f Makefile.sdl metal-enginelib`, which builds the rest of the engine as
`libquake3-engine.a` with `main()` renamed to `Q3_Main()` (`-DQ3_APP`). The
app links only system frameworks.

`Makefile.sdl` is the portable build. The original `Makefile` (i386 only,
X11/OSS) is kept as it was, apart from an x86_64 branch.

The GitHub Actions workflow (`.github/workflows/build.yml`) builds on Linux
x86_64 and Linux arm64 and runs the QVM build.

## Game data

Put your paks in `run/` at the repository root:

```
run/baseq3/pak0.pk3 ... pak8.pk3
run/missionpack/pak0.pk3        (Team Arena, optional)
```

The `run/` directory is git-ignored.

### Using the demo instead of the retail game

The original Q3A demo's game modules are for an older engine API, so they must
be rebuilt from this source. After placing the demo's `pak0.pk3` in
`run/baseq3/`:

```sh
cd code/unix
./build_qvms.sh      # builds lcc + q3asm, compiles game/cgame/ui to QVMs
                     # and writes run/baseq3/zz_vm_from_source.pk3
```

With the retail paks you do not need this; the QVMs in the retail `pak8.pk3`
work as they are.

## Running

```sh
cd code/unix
./run_q3.sh +set r_fullscreen 0 +devmap q3dm1
./run_q3.sh +set fs_game missionpack          # Team Arena
```

The home directory (config, saves, screenshots) is `~/.q3a` on Linux and
`~/Library/Application Support/Quake3` on macOS.

The macOS app asks for the game folder (the one containing `baseq3`) on first
launch and remembers it with a security-scoped bookmark; hold Option while
launching to choose again. Its config and saves live in the sandbox container,
`~/Library/Containers/com.jossy.quake3/Data/Library/Application Support/Quake3`.
`+` arguments passed to the app go to the game as usual.

`run_q3.sh` points the game at `run/` and sets `sv_pure 0`. Any extra
arguments are passed to the game. Console commands typed in-game need a
leading `/`, otherwise they are sent as chat.

## What changed from the original release

- **Portable build** `code/unix/Makefile.sdl`: C-only, interpreted QVMs,
  `-std=gnu89`, OS and architecture detection (Linux, macOS, x86_64, arm64).
  The legacy Makefile also got an x86_64 branch.
- **Portability groundwork**: the renderer takes its GL header from SDL (no
  GLX/X11 needed), macOS-specific context setup, home directory and
  `Sys_ProcessorCount`, arm64 CPU strings, and C versions of the float helpers
  on every non-i386 target. On macOS the SDL build uses the same
  `Com_Memcpy`/`Com_Memset`/`Snd_Memset` and network headers as Linux.
- **SDL2 platform layer**: `code/unix/sdl_glimp.c` (window, GL context, input,
  gamma) and `code/unix/sdl_snd.c` (audio). `linux_qgl.c` loads GL functions
  through SDL when built with `-DUSE_SDL`. The old X11/GLX/OSS files are kept
  for the i386 build.
- **64-bit fixes**: heap pointer truncation in `Hunk_Alloc`; `VM_Call`
  collecting variadic arguments from a 32-bit stack layout; `qsort` element
  size for the pak list; renderer draw-surface swap assuming an 8-byte struct;
  MD4 using a 64-bit `unsigned long` (this broke BSP/AAS checksums, so bots
  would not load); 16-byte alignment for zone and botlib allocations
  (`long double` tokens crash at `-O2` otherwise).
- **Demo restrictions are opt-in** (`+set fs_restrict 1`). A source build has
  no retail product id, so the original code always dropped into demo mode.
- **Metal + Swift platform layer for macOS** (`code/macos`): window, input
  and sound in Swift, and the renderer's GL calls translated to a Metal
  backend; used by `make metal` and the macOS app, which no longer needs SDL.
- **`code/unix/build_qvms.sh`** rebuilds the game, cgame and ui QVMs with the
  tree's own `lcc` and `q3asm`.

See `git log` for the individual commits.

## License

Quake III Arena source code is released under the GNU General Public License
v2; see `COPYING.txt`. The game data (paks, maps, textures, sounds) is **not**
covered by that license and is not included here. The original readme is
`README.txt`.
