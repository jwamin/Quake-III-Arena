/*
===========================================================================
Metal port: the GLimp_* / IN_* platform interface on top of the Swift layer.

Window, Metal device and AppKit events live in Swift (Platform.swift,
MetalRenderer.swift). This file is the engine-facing side: it fills in
glConfig, owns the input cvars, maps macOS key codes to Quake keys and
queues the events Swift reports through the MacQ_* callbacks.
===========================================================================
*/
#include <ctype.h>

#include "../renderer/tr_local.h"
#include "../client/client.h"
#include "../unix/linux_local.h"
#include "../unix/unix_glw.h"
#include "mtl_swift.h"

glwstate_t glw_state;

cvar_t *in_mouse;
cvar_t *in_subframe;    // read by unix_shared.c
cvar_t *in_nograb;

// Metal-only display cvars (see docs/metal-rt.md)
cvar_t *r_renderer;         // "metal" or "opengl" (latched)
cvar_t *r_mtlUpscale;       // 0 off, 1 MetalFX spatial, 2 temporal, 3 temporal denoised
cvar_t *r_mtlRenderScale;   // internal resolution as a fraction of r_mode
cvar_t *r_rtShadows;        // 1: ray traced dynamic lights and shadows, 2: debug view (mtl_rt.c); not latched
cvar_t *r_rtDebugLight;     // "x y z radius": a white test light for the RT pass

static qboolean mouse_avail;
static qboolean mouse_active;
static qboolean video_up;

/*
==============================================================================

KEYBOARD

==============================================================================
*/

// macOS virtual key codes (Carbon kVK_*) for keys that do not type a character
static int MacKeyToQuake( int macKey ) {
	switch ( macKey ) {
	case 0x24: return K_ENTER;          // Return
	case 0x30: return K_TAB;
	case 0x31: return K_SPACE;
	case 0x33: return K_BACKSPACE;      // Delete
	case 0x35: return K_ESCAPE;
	case 0x37: case 0x36: return K_COMMAND;
	case 0x38: case 0x3C: return K_SHIFT;
	case 0x39: return K_CAPSLOCK;
	case 0x3A: case 0x3D: return K_ALT; // Option
	case 0x3B: case 0x3E: return K_CTRL;
	case 0x7A: return K_F1;   case 0x78: return K_F2;   case 0x63: return K_F3;
	case 0x76: return K_F4;   case 0x60: return K_F5;   case 0x61: return K_F6;
	case 0x62: return K_F7;   case 0x64: return K_F8;   case 0x65: return K_F9;
	case 0x6D: return K_F10;  case 0x67: return K_F11;  case 0x6F: return K_F12;
	case 0x69: return K_F13;  case 0x6B: return K_F14;  case 0x71: return K_F15;
	case 0x72: return K_INS;            // Help / Insert
	case 0x73: return K_HOME;
	case 0x74: return K_PGUP;
	case 0x75: return K_DEL;            // Forward Delete
	case 0x77: return K_END;
	case 0x79: return K_PGDN;
	case 0x7B: return K_LEFTARROW;
	case 0x7C: return K_RIGHTARROW;
	case 0x7D: return K_DOWNARROW;
	case 0x7E: return K_UPARROW;
	// keypad
	case 0x41: return K_KP_DEL;
	case 0x43: return K_KP_STAR;
	case 0x45: return K_KP_PLUS;
	case 0x47: return K_KP_NUMLOCK;     // Clear
	case 0x4B: return K_KP_SLASH;
	case 0x4C: return K_KP_ENTER;
	case 0x4E: return K_KP_MINUS;
	case 0x51: return K_KP_EQUALS;
	case 0x52: return K_KP_INS;
	case 0x53: return K_KP_END;
	case 0x54: return K_KP_DOWNARROW;
	case 0x55: return K_KP_PGDN;
	case 0x56: return K_KP_LEFTARROW;
	case 0x57: return K_KP_5;
	case 0x58: return K_KP_RIGHTARROW;
	case 0x59: return K_KP_HOME;
	case 0x5B: return K_KP_UPARROW;
	case 0x5C: return K_KP_PGUP;
	// the key left of 1 always opens the console, whatever the layout types
	case 0x32: case 0x0A: return '`';
	}
	return 0;
}

// the Quake key each macOS key went down as, so the release matches
static int keyDownAs[256];

void MacQ_KeyEvent( int macKey, int ch, int down ) {
	int t = Sys_Milliseconds();
	int key;

	if ( macKey < 0 || macKey > 255 ) {
		return;
	}
	if ( down ) {
		key = MacKeyToQuake( macKey );
		if ( !key && ch > 32 && ch < 127 ) {
			key = tolower( ch );
		}
		keyDownAs[macKey] = key;
	} else {
		key = keyDownAs[macKey];
		keyDownAs[macKey] = 0;
		if ( !key ) {
			key = MacKeyToQuake( macKey );
		}
	}
	if ( key ) {
		Sys_QueEvent( t, SE_KEY, key, down ? qtrue : qfalse, 0, NULL );
	}
}

void MacQ_CharEvent( int ch ) {
	if ( ch == 0x7f ) {
		ch = 8;                 // Delete types DEL; the console wants backspace
	} else if ( ch == 3 ) {
		ch = 13;                // keypad Enter
	}
	if ( ( ch >= 32 && ch < 127 ) || ( ch > 0 && ch < 32 ) ) {
		Sys_QueEvent( Sys_Milliseconds(), SE_CHAR, ch, 0, 0, NULL );
	}
}

/*
==============================================================================

MOUSE

==============================================================================
*/

void MacQ_MouseMove( int dx, int dy ) {
	if ( mouse_active && ( dx || dy ) ) {
		Sys_QueEvent( Sys_Milliseconds(), SE_MOUSE, dx, dy, 0, NULL );
	}
}

void MacQ_MouseButton( int button, int down ) {
	if ( button >= 0 && button < 5 ) {
		Sys_QueEvent( Sys_Milliseconds(), SE_KEY, K_MOUSE1 + button, down ? qtrue : qfalse, 0, NULL );
	}
}

void MacQ_MouseWheel( int delta ) {
	int t = Sys_Milliseconds();
	int key = ( delta > 0 ) ? K_MWHEELUP : K_MWHEELDOWN;
	Sys_QueEvent( t, SE_KEY, key, qtrue, 0, NULL );
	Sys_QueEvent( t, SE_KEY, key, qfalse, 0, NULL );
}

void MacQ_Quit( void ) {
	Cbuf_ExecuteText( EXEC_APPEND, "quit\n" );
}

void MacQ_Print( const char *msg ) {
	Com_Printf( "%s", msg );
}

static void ActivateMouse( qboolean on ) {
	if ( on && in_nograb && in_nograb->integer ) {
		on = qfalse;
	}
	if ( on == mouse_active || !video_up ) {
		return;
	}
	mouse_active = on;
	MacPlat_GrabMouse( on );
}

void IN_ActivateMouse( void )   { if ( mouse_avail ) ActivateMouse( qtrue ); }
void IN_DeactivateMouse( void ) { ActivateMouse( qfalse ); }

void IN_Init( void ) {
	Com_Printf( "\n------- Input Initialization -------\n" );
	in_mouse = Cvar_Get( "in_mouse", "1", CVAR_ARCHIVE );
	in_subframe = Cvar_Get( "in_subframe", "1", CVAR_ARCHIVE );
	in_nograb = Cvar_Get( "in_nograb", "0", 0 );
	mouse_avail = in_mouse->value ? qtrue : qfalse;
	Com_Printf( "------------------------------------\n" );
}

void IN_Shutdown( void ) {
	IN_DeactivateMouse();
	mouse_avail = qfalse;
}

void IN_Frame( void ) {
	if ( ( cls.keyCatchers & KEYCATCH_CONSOLE ) && !r_fullscreen->integer ) {
		// let the pointer go while the console is down in a window
		IN_DeactivateMouse();
		return;
	}
	IN_ActivateMouse();
}

void IN_Activate( void ) {}
void IN_StartupJoystick( void ) {}
void IN_JoyMove( void ) {}

void Sys_SendKeyEvents( void ) {
	if ( video_up ) {
		MacPlat_PumpEvents();
	}
}

void KBD_Init( void ) {}
void KBD_Close( void ) {}

/*
==============================================================================

VIDEO

==============================================================================
*/

// Metal (MetalRenderer.swift via mtl_qgl.c), or OpenGL through the OpenGL
// framework (r_renderer opengl): both share the window and event code.
qboolean mac_metalRenderer;     // read by mtl_rt.c

void GLimp_Init( void ) {
	char name[256];
	int mode = r_mode->integer;
	int outputWidth, outputHeight, upscale;
	float scale;
	qboolean metal;

	InitSig();

	r_renderer = ri.Cvar_Get( "r_renderer", "metal", CVAR_ARCHIVE | CVAR_LATCH );
	r_mtlUpscale = ri.Cvar_Get( "r_mtlUpscale", "0", CVAR_ARCHIVE | CVAR_LATCH );
	r_mtlRenderScale = ri.Cvar_Get( "r_mtlRenderScale", "1.0", CVAR_ARCHIVE | CVAR_LATCH );
	r_rtShadows = ri.Cvar_Get( "r_rtShadows", "0", CVAR_ARCHIVE );
	r_rtDebugLight = ri.Cvar_Get( "r_rtDebugLight", "", CVAR_CHEAT );

	metal = Q_stricmp( r_renderer->string, "opengl" ) != 0;
	if ( metal && Q_stricmp( r_renderer->string, "metal" ) ) {
		ri.Printf( PRINT_ALL, "...unknown r_renderer \"%s\", using metal\n", r_renderer->string );
	}
	mac_metalRenderer = metal;

	if ( !QGL_Init( metal ? NULL : "opengl" ) ) {
		ri.Error( ERR_FATAL, metal ? "GLimp_Init() - could not set up the Metal translation layer\n"
		                           : "GLimp_Init() - could not load OpenGL\n" );
	}

	ri.Printf( PRINT_ALL, "Initializing %s display\n", metal ? "Metal" : "OpenGL" );
	if ( !R_GetModeInfo( &glConfig.vidWidth, &glConfig.vidHeight, &glConfig.windowAspect, mode ) ) {
		ri.Printf( PRINT_ALL, "...invalid mode %d, using 640x480\n", mode );
		R_GetModeInfo( &glConfig.vidWidth, &glConfig.vidHeight, &glConfig.windowAspect, 3 );
	}
	ri.Printf( PRINT_ALL, "...setting mode %d: %d %d%s\n", mode, glConfig.vidWidth, glConfig.vidHeight,
	           r_fullscreen->integer ? " fullscreen" : "" );

	// Metal: the engine renders at the internal size; MetalFX upscales it to
	// the r_mode size. glConfig reports the internal size, so the front end,
	// 2D drawing and screenshots all work in internal pixels.
	outputWidth = glConfig.vidWidth;
	outputHeight = glConfig.vidHeight;
	upscale = metal ? r_mtlUpscale->integer : 0;
	if ( upscale < 0 || upscale > 3 ) {
		ri.Printf( PRINT_ALL, "...r_mtlUpscale %d out of range, using 0\n", upscale );
		upscale = 0;
	}
	scale = r_mtlRenderScale->value;
	if ( scale < 0.34f ) {
		scale = 0.34f;
	} else if ( scale > 1.0f ) {
		scale = 1.0f;
	}
	if ( upscale && scale < 1.0f ) {
		glConfig.vidWidth = (int)( outputWidth * scale + 0.5f ) & ~1;
		glConfig.vidHeight = (int)( outputHeight * scale + 0.5f ) & ~1;
		ri.Printf( PRINT_ALL, "...rendering at %d %d, upscaled to %d %d\n",
		           glConfig.vidWidth, glConfig.vidHeight, outputWidth, outputHeight );
	} else {
		upscale = 0;
	}

	name[0] = 0;
	if ( !MacPlat_InitVideo( outputWidth, outputHeight, glConfig.vidWidth, glConfig.vidHeight, upscale,
	                         r_fullscreen->integer != 0, !metal, name, sizeof( name ) ) ) {
		ri.Error( ERR_FATAL, "GLimp_Init() - could not create the game window\n" );
	}
	video_up = qtrue;

	glConfig.isFullscreen = r_fullscreen->integer ? qtrue : qfalse;
	glConfig.colorBits = 32;
	glConfig.depthBits = 24;
	glConfig.stencilBits = 8;
	glConfig.driverType = GLDRV_ICD;
	glConfig.hardwareType = GLHW_GENERIC;
	glConfig.displayFrequency = 60;
	glConfig.stereoEnabled = qfalse;
	glConfig.textureCompression = TC_NONE;

	if ( metal ) {
		glConfig.deviceSupportsGamma = qtrue;   // applied when the frame is presented
		glConfig.textureEnvAddAvailable = qtrue;
		glConfig.maxActiveTextures = 2;
		Q_strncpyz( glConfig.vendor_string, "Apple", sizeof( glConfig.vendor_string ) );
		Q_strncpyz( glConfig.renderer_string, name[0] ? name : "Metal", sizeof( glConfig.renderer_string ) );
		Q_strncpyz( glConfig.version_string, "Metal (OpenGL 1.1 subset)", sizeof( glConfig.version_string ) );
		Q_strncpyz( glConfig.extensions_string, (const char *)qglGetString( GL_EXTENSIONS ), sizeof( glConfig.extensions_string ) );
		ri.Printf( PRINT_ALL, "...using GL_ARB_multitexture, GL_EXT_texture_env_add, GL_EXT_compiled_vertex_array (Metal)\n" );
	} else {
		GLint units = 1;
		// no hardware gamma for an OpenGL window: the renderer brightens in software
		glConfig.deviceSupportsGamma = qfalse;
		Q_strncpyz( glConfig.vendor_string, (const char *)qglGetString( GL_VENDOR ), sizeof( glConfig.vendor_string ) );
		Q_strncpyz( glConfig.renderer_string, (const char *)qglGetString( GL_RENDERER ), sizeof( glConfig.renderer_string ) );
		Q_strncpyz( glConfig.version_string, (const char *)qglGetString( GL_VERSION ), sizeof( glConfig.version_string ) );
		Q_strncpyz( glConfig.extensions_string, (const char *)qglGetString( GL_EXTENSIONS ), sizeof( glConfig.extensions_string ) );
		glConfig.textureEnvAddAvailable = strstr( glConfig.extensions_string, "GL_EXT_texture_env_add" ) ? qtrue : qfalse;
		if ( qglActiveTextureARB && strstr( glConfig.extensions_string, "GL_ARB_multitexture" ) ) {
			qglGetIntegerv( GL_MAX_TEXTURE_UNITS_ARB, &units );
		} else {
			qglActiveTextureARB = NULL;
			qglClientActiveTextureARB = NULL;
			qglMultiTexCoord2fARB = NULL;
		}
		glConfig.maxActiveTextures = units;
		if ( !strstr( glConfig.extensions_string, "GL_EXT_compiled_vertex_array" ) ) {
			qglLockArraysEXT = NULL;
			qglUnlockArraysEXT = NULL;
		}
		ri.Printf( PRINT_ALL, "...OpenGL: %s, %s, %d texture units\n", glConfig.renderer_string,
		           glConfig.version_string, units );
	}

	ri.Cvar_Set( "r_lastValidRenderer", glConfig.renderer_string );
	MacPlat_SetSwapInterval( r_swapInterval->integer );
	r_swapInterval->modified = qfalse;
}

void GLimp_Shutdown( void ) {
	IN_DeactivateMouse();
	if ( video_up ) {
		MacPlat_ShutdownVideo();
		video_up = qfalse;
	}
	memset( &glConfig, 0, sizeof( glConfig ) );
	memset( &glState, 0, sizeof( glState ) );
	QGL_Shutdown();
}

void GLimp_EndFrame( void ) {
	if ( r_swapInterval->modified ) {
		r_swapInterval->modified = qfalse;
		MacPlat_SetSwapInterval( r_swapInterval->integer );
	}
	if ( mac_metalRenderer ) {
		MTLR_EndFrame();
	} else {
		MacPlat_SwapBuffers();
	}
}

void GLimp_SetGamma( unsigned char red[256], unsigned char green[256], unsigned char blue[256] ) {
	if ( mac_metalRenderer ) {
		MTLR_SetGamma( red, green, blue );
	}
}

void GLimp_LogComment( char *comment ) {
	if ( glw_state.log_fp ) {
		fprintf( glw_state.log_fp, "%s", comment );
	}
}

// no SMP renderer
void GLimp_RenderThreadWrapper( void *stub ) {}
qboolean GLimp_SpawnRenderThread( void (*function)( void ) ) {
	ri.Printf( PRINT_WARNING, "ERROR: SMP support was disabled at compile time\n" );
	return qfalse;
}
void *GLimp_RendererSleep( void ) { return NULL; }
void GLimp_FrontEndSleep( void ) {}
void GLimp_WakeRenderer( void *data ) {}
