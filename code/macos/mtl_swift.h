/*
===========================================================================
Metal port: functions implemented in Swift (@_cdecl), for the C side.
===========================================================================
*/
#ifndef __MTL_SWIFT_H__
#define __MTL_SWIFT_H__

#include <stdint.h>
#include <stdbool.h>
#include "mtl_bridge.h"

// ---- window and events (Platform.swift) ----

// Opens the game window. Metal: a Metal layer and the renderer; the engine
// renders at renderWidth x renderHeight, and with upscale set (1 MetalFX
// spatial, 2 temporal, 3 temporal denoised) MetalFX scales that to
// outputWidth x outputHeight; the result is scaled to fit the window.
// opengl: an OpenGL (legacy profile) view and context, made current, at
// outputWidth x outputHeight (render size and upscale are ignored).
bool MacPlat_InitVideo( int outputWidth, int outputHeight, int renderWidth, int renderHeight, int upscale,
                        bool fullscreen, bool opengl, char *rendererName, int rendererNameSize );
// OpenGL only: presents the back buffer.
void MacPlat_SwapBuffers( void );
void MacPlat_ShutdownVideo( void );
// Delivers pending AppKit events through the MacQ_* callbacks.
void MacPlat_PumpEvents( void );
// Hides the cursor and reports relative motion while grabbed.
void MacPlat_GrabMouse( bool grab );
void MacPlat_SetSwapInterval( int interval );

// ---- rendering (MetalRenderer.swift) ----

// Space in this frame's vertex and index buffers for the next MTLR_Draw.
mtl_vertex_t *MTLR_AllocVertices( int count );
uint32_t *MTLR_AllocIndices( int count );
// Draws the vertices (and indices, when numIndexes > 0) allocated last.
void MTLR_Draw( const mtl_state_t *state, int primitive, int numVertexes, int numIndexes );
// Clears the scissor rectangle of the state (or the viewport when the scissor is off).
void MTLR_Clear( const mtl_state_t *state, int flags, const float color[4], float depth, int stencil );
// Reads back the frame so far; rows bottom to top like glReadPixels.
void MTLR_ReadPixels( int x, int y, int width, int height, int format, void *out );
void MTLR_EndFrame( void );
void MTLR_Finish( void );
void MTLR_SetGamma( const uint8_t *red, const uint8_t *green, const uint8_t *blue );

// alphaOne: the texture has no alpha channel (an RGB internal format); it reads as 1.
void MTLR_TexImage( int texnum, int level, int width, int height, const void *rgba, bool alphaOne );
void MTLR_TexSubImage( int texnum, int level, int x, int y, int width, int height, const void *rgba );
void MTLR_TexParams( int texnum, int minFilter, int magFilter, int mipFilter, int wrapS, int wrapT );
void MTLR_DeleteTexture( int texnum );

// ---- ray tracing (MetalRenderer.swift) ----

// Builds the acceleration structure for the static world from triangles
// (xyz per vertex, three indexes per triangle). Replaces the previous one;
// numIndexes 0 just drops it. Returns false when ray tracing is unavailable.
bool MTLR_SetWorldGeometry( const float *xyz, int numVerts, const uint32_t *indexes, int numIndexes );
// True while there is a world acceleration structure to trace against.
bool MTLR_HasWorld( void );
// Adds the view's dynamic lights, with ray traced shadows, to the opaque
// image drawn so far.
void MTLR_TraceDlights( const mtl_rt_view_t *view );

// ---- sound (Audio.swift) ----

// Plays the engine's ring buffer of interleaved 16-bit samples. *position is
// the next sample (not frame) the device reads; Swift advances it.
bool MacAudio_Start( int16_t *buffer, int samples, int channels, int rate, volatile int32_t *position );
void MacAudio_Stop( void );

#endif
