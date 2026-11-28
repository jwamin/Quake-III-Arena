# Metal / MetalFX hybrid ray-traced renderer

Branch: `metal-rt` (rebased onto `macos-metal-swift` on 2026-11-28).

## Decisions

- Keep the renderer front end and the `refexport_t` / `refimport_t` seam
  (`tr_public.h`). Rendering goes to Metal; OpenGL stays the default
  everywhere else and must keep building on Linux.
- Hybrid ray tracing: Metal raster for primary visibility, hardware RT for
  shadows, reflections and dynamic-light / GI terms.
- Target: macOS on Apple Silicon. MetalFX spatial, temporal and temporal
  denoised upscaling.
- Platform layer: the existing native AppKit + `CAMetalLayer` layer from
  `macos-metal-swift` (`code/macos`, Swift behind C headers). No SDL. The
  C89 engine files never see Swift or Objective-C; they talk to it only
  through `mtl_bridge.h` / `mtl_swift.h`.
- Settings are ordinary archived, latched cvars (`seta` lines in
  `q3config.cfg`, overridable with `+set` on the command line, applied with
  `vid_restart`). No new config mechanism.

## Checked API facts (Xcode 27.1 SDK, Apple M4, macOS 27.2)

| Feature | Availability | M4 |
|---|---|---|
| `MTLAccelerationStructure`, intersection queries | macOS 11+ (`supportsRaytracing`) | yes |
| Ray tracing from render (fragment) functions | macOS 12+ (`supportsRaytracingFromRender`) | yes |
| `MTLFXSpatialScaler` | macOS 13+ | yes |
| `MTLFXTemporalScaler` | macOS 13+ | yes, input scale 1.0–3.0 |
| `MTLFXTemporalDenoisedScaler` | macOS 26+ | yes, input scale 1.0–3.0 |

The denoised scaler takes color, depth, motion, diffuse albedo, specular
albedo, normal and roughness (plus optional specular hit distance, reactive
and denoise-strength masks), the camera's `worldToViewMatrix` /
`viewToClipMatrix`, and jitter. That is the G-buffer the RT phases must
produce.

M1/M2 expose `supportsRaytracing` but not the hardware RT units (M3+ has
them), so RT is slower there. Every RT feature checks support at start-up
and turns itself off with a console message when it is missing.

## Architecture

```
 engine (C89)                    code/renderer (front end, unchanged API)
      |                                   |
      |                       tr_backend.c / tr_shade.c  -- qgl* -->  mtl_qgl.c   (GL 1.x state machine on Metal)
      |                                   |                               |
      |                       rb_backend.h hooks (RB_Backend_*)  ------>  |   (scene begin/end, world loaded, camera)
      |                                                                   v
 mac_glimp.c (GLimp_*, IN_*, cvars) --- mtl_swift.h ---> MetalRenderer.swift
                                                           |  raster pass  -> color + depth (internal res)
                                                           |  RT passes    -> shadows / reflections / dlights (compute, intersection queries)
                                                           |  MetalFX      -> upscale to output res
                                                           |  present      -> gamma ramp, fit to drawable
                                                         Shaders.metal / RayTracing.metal
```

1. **Raster** stays on the existing `qgl*`-on-Metal path (`mtl_qgl.c`). It
   already renders every shader stage correctly, so primary visibility is
   not rewritten. `rb_backend.h` grows the hooks the RT and temporal passes
   need, implemented as no-ops in `rb_gl.c` and for real in the Metal build.
2. **Internal resolution.** The engine renders at `r_mode` size times
   `r_mtlRenderScale` (that is what `glConfig.vidWidth/Height` report, so
   the front end, 2D and screenshots work unchanged). MetalFX upscales to the
   `r_mode` size, and the present pass fits that to the window.
3. **Acceleration structures.** The world BSP (`tr.world` surfaces: faces,
   patches after tessellation, triangle soups) is static: one primitive
   acceleration structure, built once when the map loads (new hook from
   `RE_LoadWorldMap`). Entities (MD3 models, brush models) are per-frame
   instances in an instance acceleration structure, refitted each frame.
   Sky and fog surfaces and non-opaque shaders are left out, or marked
   non-opaque for alpha-tested geometry.
4. **G-buffer.** Shadows and dynamic lights need world position and normal.
   Phase 2 rebuilds position from depth and the inverse view-projection
   matrix. Normals and albedo for the denoiser come from a second color
   attachment written by the raster pass (phase 4).
5. **RT passes** run as compute kernels with intersection queries, between
   the end of the 3D view and the 2D/HUD draws (new `RB_Backend_EndScene`
   hook called at the end of `RB_DrawSurfs` for non-`RDF_NOWORLDMODEL`
   views):
   - dynamic lights (`backEnd.refdef.dlights`) with ray-traced shadows,
     added on top of the lightmapped image (replaces Q3's projected dlight
     pass when on);
   - sun shadows, for maps whose sky shader has `q3map_sun`;
   - reflections on surfaces whose shader has an environment-map stage
     (`tcGen environment`), one bounce, sampling the hit surface's lightmap
     and diffuse texture;
   - one-bounce GI from dynamic lights (experimental, needs the denoiser).
6. **Temporal data.** Temporal upscaling needs sub-pixel jitter (added to
   the projection in `R_SetupProjection` through a backend hook) and motion
   vectors (camera motion from depth plus the previous view-projection;
   entities through a reactive mask at first, per-entity motion later).

## Cvars (all `CVAR_ARCHIVE | CVAR_LATCH`, Metal build only)

| Cvar | Default | Meaning |
|---|---|---|
| `r_mtlUpscale` | 0 | 0 off, 1 MetalFX spatial, 2 temporal, 3 temporal denoised |
| `r_mtlRenderScale` | 1.0 | internal resolution as a fraction of `r_mode` (0.34–1.0) |
| `r_rtShadows` | 0 | 0 off, 1 dynamic-light shadows, 2 also sun shadows |
| `r_rtReflections` | 0 | ray-traced reflections on environment-mapped surfaces |
| `r_rtGI` | 0 | one-bounce indirect light from dynamic lights |
| `r_rtSamples` | 1 | rays per pixel per effect |

Example: `quake3-metal +set r_mtlUpscale 1 +set r_mtlRenderScale 0.5`.

## Phases

1. Cvars, internal resolution and MetalFX spatial upscaling. **(done)**
2. World acceleration structure and RT dynamic-light shadows (depth-based
   position reconstruction).
3. Camera jitter, motion vectors, MetalFX temporal upscaling.
4. G-buffer attachment (normal, albedo, roughness), RT reflections, entity
   instances.
5. Temporal denoised upscaling, sun shadows, GI.

Each phase is checked by running `quake3-metal` on retail maps with
screenshots, and by the Linux SDL/GL build staying green.
