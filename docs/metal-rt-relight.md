<!-- Status: planned, not implemented (written 2026-11-28). -->
# Plan: ray traced relight mode from the map's own lights (`r_rtRelight`)

## Context

Ray tracing currently only affects dynamic lights (rockets, muzzle flashes). The world's
lighting is baked into lightmaps, so environmental light never moves, never shadows
players, and players don't receive shadowed light from the map's lights. The user wants
a **full relight mode**: replace the lightmaps with real-time ray traced direct light,
inferred from game data, using **point/spot lights, surface lights and the sun**.

Findings that make this tractable:
- Retail BSPs keep the `light` entities (q3dm1: 87, 10 of them spotlights; q3dm17: 45)
  with `origin`, `light`, `_color`, `spawnflags`, `target`, `radius`. The renderer
  already keeps the entity text (`tr.world->entityString`, `tr_bsp.c` `R_LoadEntities`).
- 573 retail shaders have `q3map_surfacelight`; the renderer skips `q3map*` keywords
  (`tr_shader.c:1485`) but already parses `q3map_sun` into `tr.sunLight` /
  `tr.sunDirection` (`tr_shader.c:1442`).
- **The q3map source is in the repo** (`q3map/light.c`, `q3map/shaders.c`). Retail maps
  were lit by this q3map: direct light only (no bounce), plus a flat worldspawn ambient.
  So evaluating q3map's exact formulas per pixel with ray traced shadows should reproduce
  the baked lightmaps closely, which gives an objective accuracy target.

q3map formulas to reproduce (`q3map/light.c`):
- point: `photons = light * 7500` (default 300; `_color` normalized),
  `add = photons / max(d,16)^2 * N·L`; linear (spawnflags & 1):
  `add = N·L * photons / 8000 - d`; skip `add <= 1` (0–255 units).
- spot (`target` → entity with matching `targetname`): `radiusByDist = (radius(64)+16)/dist`;
  cone test with a 32-unit soft edge (`light.c` ~950).
- area (surface light): per subdivided patch, `photons = value * area * 0.25`,
  origin = patch centre + normal, `add = photons/d^2 * N·L * (−Nl·L)`; plus a 5%
  backsplash point light 24 units out (`q3map_backsplash` overrides); autosprite
  shaders become one point light (`value * 7500`); `cull none` = two-sided;
  colour = normalized average of `q3map_lightimage` / `qer_editorimage` / shader image.
- sun: `add = N·L_sun * sunLight` when the ray toward the sun escapes (sky is not in the
  acceleration structure, so "no hit" = sky).
- ambient: worldspawn `_color * ambient`, added everywhere.
- to framebuffer units (what a lightmap texel holds): `L * 2^(r_mapOverBrightBits − tr.overbrightBits) / 255`,
  then normalize by the max channel if > 1 (`ColorToBytes` + `R_ColorShiftLightingBytes`).

## Approach

### 1. Shader data for surface lights (`code/renderer/tr_shader.c`, `tr_local.h`)
- Parse `q3map_surfacelight <value>`, `q3map_lightsubdivide`, `q3map_backsplash`,
  `q3map_lightimage`, `qer_editorimage` into new `shader_t` fields (`surfaceLight`,
  `lightSubdivide`, `backsplashFraction/Distance`, `lightImageName`); keep skipping the rest.
- Reset `tr.sunLight` at world load so a map without a sun doesn't inherit one.

### 2. Light extraction at map load (new `code/macos/mtl_rtlights.c`, called from
`RB_Backend_WorldLoaded` in `mtl_rt.c`)
- Parse `tr.world->entityString` with `COM_ParseExt` (same pattern as `R_LoadEntities`):
  lights, spotlight targets (resolve `targetname`), worldspawn `ambient`/`_color`.
- Surface lights: walk model-0 surfaces (same SF_FACE / SF_GRID / SF_TRIANGLES walk as
  `RT_AddSurface` in `mtl_rt.c`) whose shader has `surfaceLight > 0`; split triangles
  until edges ≤ `r_rtAreaSubdivide` (default 128) and emit one area VPL per piece, plus
  backsplash; autosprite → point. Colour via `R_LoadImage` (`tr_image.c:1815`) average,
  cached per shader.
- Sun from `tr.sunLight` / `tr.sunDirection`.
- Per light: precompute range from the `add > 1` cutoff (`sqrt(photons)`, linear:
  `photons/8000`).
- **Static light index**: uniform 3D cell grid over the world bounds (128-unit cells);
  each cell lists the lights whose range reaches it, sorted by max contribution, capped
  (e.g. 256). Built once on the CPU; print stats (lights, VPLs, avg/max per cell).
- Hand to Swift via a new `MTLR_SetWorldLights(...)` (lights array, cell offsets/counts,
  index list, grid origin/size, ambient, sun, overbright scale); new structs in
  `mtl_bridge.h` (`mtl_rt_envlight_t`, matching an MSL struct).

### 3. Albedo + relight mask in the raster pass
- Stage rule in `tr_shade.c`, same pattern as `RB_ClayStage` / `s_clayWhite`:
  when `r_rtRelight` is on and the shader is **relightable** (SS_OPAQUE, not sky, no
  surface light, no additive/blended glow stages), lightmap bundles bind
  `tr.whiteImage` and world vertex-lit colours (`CGEN_VERTEX/EXACT_VERTEX`) become white,
  so the framebuffer holds albedo. Non-relightable shaders (emissive panels, glows,
  fullbright, effects) draw exactly as baked.
- Entities: in `R_SetupEntityLighting` (`tr_light.c`) with relight on, ambient = white,
  directed = 0 (like the existing `RB_Backend_RTDlights()` gate), so models also hold
  albedo and get RT light + shadows.
- Per-pixel mask: new backend hook `RB_Backend_SetRelightMark(int)` (none/set/clear),
  GL no-op; Metal: field in `mtl_state_t` → `MetalRenderer` depth-stencil state writes
  stencil bit 0x80 (REPLACE on depth pass) for relightable draws, clears it for other
  opaque draws. Depth target gets `.pixelFormatView`; the kernel reads an
  `.x32_stencil8` view. (`r_shadows 2` stencil shadows are disabled in relight mode.)

### 4. `rt_relight` kernel (`Shaders.metal`, dispatched from `MetalRenderer.swift`)
- Runs in `RB_Backend_OpaqueDone` **before** the existing dlight pass, only on pixels
  with the mask bit. Reuses position/normal reconstruction from `rt_dlights`
  (`rtWorldPos`, weapon depth un-hack, neighbour normals — factor into shared helpers).
- Look up the pixel's light cell, evaluate q3map's unshadowed `add` for each light (cheap
  math), trace shadow rays for the strongest `r_rtLightBudget` (default 16); remaining
  lights are scaled by the visible fraction of the traced ones (stable, no noise; a
  denoised stochastic version is phase 5 work). Sun: one ray if facing it.
- `colour = albedo * normalize(ambient + Σ add) * overbrightScale * r_rtRelightScale`.
- The existing dlight pass then runs on top unchanged; `r_rtShadows 2` debug view and
  `r_clay` keep working (clay × relight = lighting-only view of the relit world).

### 5. Cvars, menu, docs
- `r_rtRelight` 0/1 (archived, live), `r_rtLightBudget` (16), `r_rtAreaSubdivide`
  (128, latched: applies on map load), `r_rtRelightScale` (1.0, calibration),
  `r_rtSun` 0/1. Registered in `mac_glimp.c` with the other RT cvars.
- RAY TRACING tab (`code/q3_ui/ui_metal.c`): "Relight World: Off / On" and
  "Light Budget: 4 / 8 / 16 / 32"; rebuild `zz_metal_ui.pk3`.
- `docs/metal-rt.md`: new phase "2b: relight from map lights" with the formulas above,
  cvars, limits; README RT line.

## Critical files
- `code/macos/mtl_rt.c`, new `code/macos/mtl_rtlights.c`, `mtl_bridge.h`,
  `mtl_swift.h`, `MetalRenderer.swift`, `Shaders.metal`, `mtl_qgl.c`, `mac_glimp.c`
- `code/renderer/tr_shader.c`, `tr_local.h`, `tr_shade.c`, `tr_light.c`,
  `rb_backend.h`, `rb_gl.c` (GL no-ops)
- `code/unix/Makefile.sdl` (`METAL_LAYER += mtl_rtlights`); Xcode picks up `code/macos`
  automatically.
- `code/q3_ui/ui_metal.c`, `docs/metal-rt.md`, `README.md`

## Known limits (v1)
- Brush models (doors, lifts) and entities don't cast shadows until phase 4 puts them in
  the acceleration structure; alpha-tested surfaces don't cast shadows.
- Glow/emissive shaders keep their baked lighting (not relit).
- `_style` switchable lights ignored; translucent "filter" colouring of light ignored.

## Verification
1. Builds: `make -f Makefile.sdl all metal`, Xcode app; SDL/GL build unaffected.
2. Extraction stats on q3dm1/q3dm17/q3dm6: light/VPL counts printed; spotlight targets
   resolved; sun present on outdoor maps (q3dm17), absent indoors.
3. **Accuracy vs baked**: q3dm1 fixed view (`setviewpos 212 2360 82 -45`), `r_clay 1`,
   screenshots with `r_rtRelight 0` vs `1`; diff with the scratchpad `tools/diff`. Expect
   the same light placement and shadow shapes; tune `r_rtRelightScale` until mean
   brightness matches, then make that the default.
4. Dynamic shadows: relight + a player/bot in view (receives shadowed light);
   `r_rtShadows 2` and clay still correct.
5. Performance: `timedemo four` at 1024×768 and 1920×1080 internal with relight off/on
   and budgets 8/16/32; target ≥ 60 fps at 1080p internal on the M4.
6. Commit, update docs, push (as for previous phases).
