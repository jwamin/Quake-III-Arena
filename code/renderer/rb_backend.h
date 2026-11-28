/*
===========================================================================
rb_backend.h -- contract between the shared renderer front end and a
rendering backend.

Two backends are intended to implement this header:

  * OpenGL (default):   rb_gl.c, plus the existing tr_backend.c, tr_shade.c,
                        tr_shadows.c, tr_sky.c, tr_flares.c, tr_init.c and
                        the platform GLimp_* layer.
  * Metal (RENDERER_MTL): code/renderer_mtl/ (not yet written).

Shared front-end files (tr_cmds.c, tr_main.c, tr_shader.c, ...) must not
call qgl* directly.  Anything that needs the graphics API goes through a
function declared here, so a backend can be swapped at build time.

Ownership decisions so far:
  * tr_image.c stays GL-only (decision: "option B").  The Metal backend
    provides its own texture upload (resample, picmip, mip chain) rather
    than splitting Upload32; the shared image loaders (TGA/JPG/PCX) will
    be factored out when the Metal backend is written.
  * tr_init.c, tr_surface.c, tr_sky.c, tr_flares.c still contain GL and
    are not yet split.

This header is a work in progress: it grows as GL calls are moved out of
shared files.  See the Metal/ray-tracing plan for the remaining steps.
===========================================================================
*/
#ifndef RB_BACKEND_H
#define RB_BACKEND_H

/*
** Draw buffer ids carried in drawBufferCommand_t.buffer.
**
** The OpenGL backend uses the GL enums directly so that its behaviour is
** unchanged.  Other backends use opaque small integers.
*/
#ifdef RENDERER_MTL
enum {
	RB_DRAWBUFFER_FRONT,
	RB_DRAWBUFFER_BACK,
	RB_DRAWBUFFER_BACK_LEFT,
	RB_DRAWBUFFER_BACK_RIGHT
};
#else
#define RB_DRAWBUFFER_FRONT		GL_FRONT
#define RB_DRAWBUFFER_BACK		GL_BACK
#define RB_DRAWBUFFER_BACK_LEFT		GL_BACK_LEFT
#define RB_DRAWBUFFER_BACK_RIGHT	GL_BACK_RIGHT
#endif

/*
** Turn the overdraw measurement mode (r_measureOverdraw) on or off.
** Called from the front end with the render thread already synchronised.
*/
void		RB_Backend_SetOverdrawMeasure( qboolean enable );

/*
** Returns 0 when the graphics API reports no error, otherwise a
** backend-specific nonzero error code suitable for printing with %x.
** Called from the front end with the render thread already synchronised.
*/
int		RB_Backend_GetError( void );

/*
** True when the backend can combine two texture stages in a single pass.
** Used by the shader parser (CollapseMultitexture).
*/
qboolean	RB_Backend_MultitextureAvailable( void );

/*
** Prepare the backend for a batch of RB_Backend_DebugPolygon calls
** (flat white texture, front-face culling).
*/
void		RB_Backend_BeginDebugDraw( void );

/*
** Draw a debug polygon for r_debugSurface.  color is a 3-bit mask
** (bit 0 = red, 1 = green, 2 = blue).  Called by R_DebugGraphics through
** ri.CM_DrawDebugSurface, so it must keep this signature.
*/
void		RB_Backend_DebugPolygon( int color, int numPoints, float *points );

/*
** Ray tracing hooks (docs/metal-rt.md).  The OpenGL backend implements them
** as no-ops; the Metal build implements them in code/macos/mtl_rt.c.
*/

/*
** tr.world has just been loaded (RE_LoadWorldMap): build whatever the
** backend traces rays against.
*/
void		RB_Backend_WorldLoaded( void );

/*
** The opaque surfaces of the current view have been drawn (the depth buffer
** is complete for them); translucent surfaces follow.  Called once per view
** from RB_RenderDrawSurfList, with the last batch already flushed.
*/
void		RB_Backend_OpaqueDone( void );

/*
** True when the backend lights dynamic lights itself (ray traced), so the
** renderer must not add them as well (projected dlight passes, dlights in
** entity lighting).
*/
qboolean	RB_Backend_RTDlights( void );

/*
** Relighting (r_rtRelight, docs/metal-rt-relight.md): the backend replaces
** the lightmaps with ray traced light from the map's own lights. While it is
** active the renderer draws relightable surfaces with white lightmaps (so the
** framebuffer holds albedo) and tells the backend, per opaque batch, whether
** its pixels are to be relit.
*/
enum {
	RB_RELIGHT_NONE,		// not an opaque world-view draw: leave the mask alone
	RB_RELIGHT_SET,			// relight these pixels
	RB_RELIGHT_CLEAR		// opaque but not relit (drawn with baked lighting)
};
qboolean	RB_Backend_RelightActive( void );
void		RB_Backend_SetRelightMark( int mark );

#endif	// RB_BACKEND_H
