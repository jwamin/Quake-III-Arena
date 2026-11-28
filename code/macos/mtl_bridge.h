/*
===========================================================================
Metal port: the interface between the C engine and the Swift platform layer.

This header is imported by Swift (as its bridging header) and included by C.
It holds plain C types only, so it can be imported without the engine
headers. It declares:
  - the vertex and per-draw state structures that mtl_qgl.c fills in and
    MetalRenderer.swift consumes (the uniform blocks match Shaders.swift);
  - the C callbacks the Swift side calls (MacQ_*), implemented in mac_glimp.c.

The functions implemented in Swift (@_cdecl) are declared for C in
mtl_swift.h, which Swift does not import.
===========================================================================
*/
#ifndef __MTL_BRIDGE_H__
#define __MTL_BRIDGE_H__

#include <stdint.h>

// One vertex as the Metal vertex shader reads it (32 bytes, tightly packed).
typedef struct {
	float   xyz[3];
	float   st0[2];
	float   st1[2];
	uint8_t color[4];
} mtl_vertex_t;

// Vertex stage uniforms (matches VSUniforms in Shaders.swift).
typedef struct {
	float   mvp[16];        // column-major, OpenGL clip space
	float   clipPlane[4];   // user clip plane in object space
	int32_t clipEnable;
	int32_t pad[3];
} mtl_vs_uniforms_t;

// Fragment stage uniforms (matches FSUniforms in Shaders.swift).
typedef struct {
	int32_t tex0Enable;
	int32_t tex1Enable;
	int32_t env0;           // MTL_ENV_*
	int32_t env1;
	int32_t alphaFunc;      // MTL_CMP_*; MTL_CMP_ALWAYS disables the test
	float   alphaRef;
	int32_t pad[2];
} mtl_fs_uniforms_t;

enum { MTL_ENV_MODULATE, MTL_ENV_REPLACE, MTL_ENV_DECAL, MTL_ENV_ADD };

// Same order as MTLCompareFunction.
enum { MTL_CMP_NEVER, MTL_CMP_LESS, MTL_CMP_EQUAL, MTL_CMP_LEQUAL,
       MTL_CMP_GREATER, MTL_CMP_NOTEQUAL, MTL_CMP_GEQUAL, MTL_CMP_ALWAYS };

// Same order as MTLBlendFactor (the subset OpenGL 1.x has).
enum { MTL_BLEND_ZERO, MTL_BLEND_ONE, MTL_BLEND_SRC_COLOR, MTL_BLEND_ONE_MINUS_SRC_COLOR,
       MTL_BLEND_SRC_ALPHA, MTL_BLEND_ONE_MINUS_SRC_ALPHA, MTL_BLEND_DST_COLOR,
       MTL_BLEND_ONE_MINUS_DST_COLOR, MTL_BLEND_DST_ALPHA, MTL_BLEND_ONE_MINUS_DST_ALPHA,
       MTL_BLEND_SRC_ALPHA_SATURATE };

// Same order as MTLStencilOperation (the subset OpenGL 1.x has).
enum { MTL_STENCIL_KEEP, MTL_STENCIL_ZERO, MTL_STENCIL_REPLACE, MTL_STENCIL_INCR,
       MTL_STENCIL_DECR, MTL_STENCIL_INVERT };

enum { MTL_CULL_NONE, MTL_CULL_FRONT, MTL_CULL_BACK };

enum { MTL_PRIM_TRIANGLES, MTL_PRIM_LINES, MTL_PRIM_POINTS };

enum { MTL_FILTER_NEAREST, MTL_FILTER_LINEAR };
enum { MTL_MIP_NONE, MTL_MIP_NEAREST, MTL_MIP_LINEAR };
enum { MTL_WRAP_REPEAT, MTL_WRAP_CLAMP };

enum { MTL_READ_RGB, MTL_READ_RGBA, MTL_READ_DEPTH, MTL_READ_STENCIL };

enum { MTL_CLEAR_COLOR = 1, MTL_CLEAR_DEPTH = 2, MTL_CLEAR_STENCIL = 4 };

// Everything the OpenGL state machine contributes to one draw.
typedef struct {
	mtl_vs_uniforms_t vs;
	mtl_fs_uniforms_t fs;

	int32_t  tex0, tex1;            // texture names (0 = white)

	// render pipeline: blending and color writes
	uint8_t  blendEnable;
	uint8_t  blendSrc, blendDst;    // MTL_BLEND_*
	uint8_t  colorMask;             // bit 0 red .. bit 3 alpha

	// depth / stencil
	uint8_t  depthTest, depthWrite, depthFunc;
	uint8_t  stencilTest, stencilFunc;
	uint8_t  stencilFail, stencilZFail, stencilZPass;
	uint8_t  stencilRef, stencilReadMask, stencilWriteMask;

	// rasterizer
	uint8_t  cullMode;              // MTL_CULL_*
	uint8_t  wireframe;
	uint8_t  polygonOffset;
	float    offsetFactor, offsetUnits;

	// viewport and scissor in OpenGL window coordinates (origin bottom left)
	int32_t  viewport[4];
	float    depthNear, depthFar;
	uint8_t  scissorEnable;
	int32_t  scissor[4];
} mtl_state_t;

// ---- ray tracing (mtl_rt.c -> MetalRenderer.swift) ----

#define MTL_RT_MAX_LIGHTS	32		// MAX_DLIGHTS

// One dynamic light (matches RTLight in Shaders.metal).
typedef struct {
	float   origin[3];
	float   radius;
	float   color[3];       // 0..1, as in dlight_t
	float   pad;
} mtl_rt_light_t;

// Everything the dynamic-light pass needs for one view.
typedef struct {
	float   projection[16]; // OpenGL clip from eye, column-major
	float   modelview[16];  // eye from world, column-major
	float   viewOrigin[3];
	int32_t viewport[4];    // OpenGL window coordinates (origin bottom left)
	int32_t numLights;
	int32_t debugView;      // r_rtShadows 2: show only the RT light (blue: shadowed)
	int32_t lightBudget;    // relight: shadow rays per pixel (r_rtLightBudget)
	float   relightScale;   // relight: r_rtRelightScale
	int32_t sun;            // relight: r_rtSun
	int32_t relightMode;    // relight: r_rtRelight (2: no shadow rays, for diagnosis)
	mtl_rt_light_t lights[MTL_RT_MAX_LIGHTS];
} mtl_rt_view_t;

// ---- relighting from the map's own lights (mtl_rtlights.c, docs/metal-rt-relight.md) ----

#define MTL_RT_MAX_BUDGET   32

enum { MTL_ENVLIGHT_POINT, MTL_ENVLIGHT_SPOT, MTL_ENVLIGHT_AREA };

// One map light in q3map's terms (matches EnvLight in Shaders.metal, 64 bytes).
typedef struct {
	float   origin[3];
	float   photons;        // q3map photons
	float   color[3];       // normalized (max component 1)
	float   range;          // beyond this q3map's add drops to <= 1
	float   normal[3];      // spotlight direction / area light normal
	float   radiusByDist;   // spotlight cone
	int32_t type;           // MTL_ENVLIGHT_*
	int32_t linear;         // spawnflags 1: linear falloff
	int32_t twoSided;       // area light emitting on both sides
	int32_t pad;
} mtl_rt_envlight_t;

// Map-wide relight data (matches EnvInfo in Shaders.metal).
typedef struct {
	float   gridOrigin[3];  // light index grid
	float   cellSize;
	int32_t gridDims[3];
	int32_t numLights;
	float   ambient[3];     // worldspawn _color * ambient (q3map units, 0-255)
	float   lightScale;     // q3map units -> lightmap texel: 2^(mapOverBright - overbright) / 255
	float   sunColor[3];    // q3map_sun colour * intensity; 0 = no sun
	float   pad0;
	float   sunDir[3];      // towards the sun
	float   pad1;
} mtl_rt_envinfo_t;

// relight stencil mark for opaque draws (mtl_qgl.c)
enum { MTL_RELIGHT_NONE, MTL_RELIGHT_SET, MTL_RELIGHT_CLEAR };
#define MTL_RELIGHT_STENCIL_BIT 0x80

// ---- implemented in C (mac_glimp.c), called from Swift ----

// A key went down or up. macKey is the virtual key code (kVK_*), ch the
// character the key types without modifiers (0 when it types nothing).
void MacQ_KeyEvent( int macKey, int ch, int down );
// Text input, one Unicode scalar at a time.
void MacQ_CharEvent( int ch );
void MacQ_MouseMove( int dx, int dy );
void MacQ_MouseButton( int button, int down );
void MacQ_MouseWheel( int delta );
void MacQ_Quit( void );
// Print to the engine console.
void MacQ_Print( const char *msg );

#endif
