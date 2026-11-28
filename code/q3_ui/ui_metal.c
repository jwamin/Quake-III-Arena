/*
===========================================================================
METAL and RAY TRACING pages of System Setup (macOS Metal build).

The engine registers r_renderer only in the Metal build (code/macos), so
these tabs appear only there. See docs/metal-rt.md for the cvars.

  METAL        r_renderer (Metal / OpenGL), r_mtlUpscale, r_mtlRenderScale;
               latched, applied with Accept (vid_restart)
  RAY TRACING  r_rtShadows (Off / On / Debug View), r_clay; applied
               immediately, need the Metal
               renderer

The other System Setup pages add the two tabs with UI_MetalTabs_Add.
===========================================================================
*/

#include "ui_local.h"

#define ART_FRAMEL			"menu/art/frame2_l"
#define ART_FRAMER			"menu/art/frame1_r"
#define ART_BACK0			"menu/art/back_0"
#define ART_BACK1			"menu/art/back_1"
#define ART_ACCEPT0			"menu/art/accept_0"
#define ART_ACCEPT1			"menu/art/accept_1"

#define ID_TAB_GRAPHICS		200
#define ID_TAB_DISPLAY		201
#define ID_TAB_SOUND		202
#define ID_TAB_NETWORK		203
#define ID_TAB_METAL		204
#define ID_TAB_RT			205
#define ID_BACK				206
#define ID_APPLY			207
#define ID_RENDERER			208
#define ID_UPSCALE			209
#define ID_SCALE			210
#define ID_RTLIGHTS			211
#define ID_CLAY				212

enum { TAB_NONE, TAB_METAL, TAB_RT };

/*
=======================================================================

SHARED

=======================================================================
*/

qboolean UI_MetalTabsAvailable( void ) {
	char	buf[16];

	trap_Cvar_VariableStringBuffer( "r_renderer", buf, sizeof( buf ) );
	return buf[0] != 0;
}

// the renderer running now (not a pending r_renderer change)
qboolean UI_MetalRendererActive( void ) {
	return Q_stricmpn( uis.glconfig.renderer_string, "Metal", 5 ) == 0;
}

static void UI_MetalTabs_Event( void *ptr, int event ) {
	if ( event != QM_ACTIVATED ) {
		return;
	}
	switch ( ( (menucommon_s *)ptr )->id ) {
	case ID_TAB_GRAPHICS:	UI_PopMenu(); UI_GraphicsOptionsMenu(); break;
	case ID_TAB_DISPLAY:	UI_PopMenu(); UI_DisplayOptionsMenu(); break;
	case ID_TAB_SOUND:		UI_PopMenu(); UI_SoundOptionsMenu(); break;
	case ID_TAB_NETWORK:	UI_PopMenu(); UI_NetworkOptionsMenu(); break;
	case ID_TAB_METAL:		UI_PopMenu(); UI_MetalOptionsMenu(); break;
	case ID_TAB_RT:			UI_PopMenu(); UI_RayTracingOptionsMenu(); break;
	}
}

static void UI_MetalTabs_Item( menutext_s *t, int id, const char *name, int y, qboolean current ) {
	t->generic.type		= MTYPE_PTEXT;
	t->generic.flags	= QMF_RIGHT_JUSTIFY | ( current ? 0 : QMF_PULSEIFFOCUS );
	t->generic.id		= id;
	t->generic.callback	= UI_MetalTabs_Event;
	t->generic.x		= 216;
	t->generic.y		= y;
	t->string			= (char *)name;
	t->style			= UI_RIGHT;
	t->color			= color_red;
}

/*
=================
UI_MetalTabs_Add

Adds the METAL and RAY TRACING tabs under NETWORK on a System Setup page.
Does nothing when the engine has no Metal renderer.
=================
*/
void UI_MetalTabs_Add( menuframework_s *menu, menutext_s *metal, menutext_s *rt ) {
	if ( !UI_MetalTabsAvailable() ) {
		return;
	}
	UI_MetalTabs_Item( metal, ID_TAB_METAL, "METAL", 240 + 2 * PROP_HEIGHT, qfalse );
	UI_MetalTabs_Item( rt, ID_TAB_RT, "RAY TRACING", 240 + 3 * PROP_HEIGHT, qfalse );
	Menu_AddItem( menu, metal );
	Menu_AddItem( menu, rt );
}

/*
=======================================================================

PAGES

=======================================================================
*/

static const char *renderer_names[] = { "Metal", "OpenGL", 0 };
static const char *upscale_names[] = { "Off", "MetalFX Spatial", 0 };
static const char *scale_names[] = { "100%", "75%", "67%", "50%", "33%", 0 };
static const float scale_values[] = { 1.0f, 0.75f, 0.67f, 0.5f, 0.34f };
#define NUM_SCALES	( sizeof( scale_values ) / sizeof( scale_values[0] ) )
static const char *onoff_names[] = { "Off", "On", 0 };
static const char *rtlights_names[] = { "Off", "On", "Debug View", 0 };	// r_rtShadows 0..2

typedef struct {
	menuframework_s	menu;

	menutext_s		banner;
	menubitmap_s	framel;
	menubitmap_s	framer;

	menutext_s		graphics;
	menutext_s		display;
	menutext_s		sound;
	menutext_s		network;
	menutext_s		metal;
	menutext_s		rt;

	// METAL
	menulist_s		renderer;
	menulist_s		upscale;
	menulist_s		scale;
	int				initRenderer, initUpscale, initScale;

	// RAY TRACING
	menulist_s		rtLights;
	menulist_s		clay;

	menubitmap_s	apply;
	menubitmap_s	back;

	int				tab;
} metalOptions_t;

static metalOptions_t	s_metal;

static void MetalOptions_Cache( void ) {
	trap_R_RegisterShaderNoMip( ART_FRAMEL );
	trap_R_RegisterShaderNoMip( ART_FRAMER );
	trap_R_RegisterShaderNoMip( ART_BACK0 );
	trap_R_RegisterShaderNoMip( ART_BACK1 );
	trap_R_RegisterShaderNoMip( ART_ACCEPT0 );
	trap_R_RegisterShaderNoMip( ART_ACCEPT1 );
}

static int MetalOptions_ScaleIndex( float scale ) {
	int		i, best = 0;

	for ( i = 1; i < NUM_SCALES; i++ ) {
		if ( fabs( scale_values[i] - scale ) < fabs( scale_values[best] - scale ) ) {
			best = i;
		}
	}
	return best;
}

// grey out what does not apply, show Accept when a latched setting changed
static void MetalOptions_Update( void ) {
	if ( s_metal.tab == TAB_METAL ) {
		qboolean opengl = s_metal.renderer.curvalue == 1;

		if ( opengl ) {
			s_metal.upscale.generic.flags |= QMF_GRAYED;
		} else {
			s_metal.upscale.generic.flags &= ~QMF_GRAYED;
		}
		if ( opengl || s_metal.upscale.curvalue == 0 ) {
			s_metal.scale.generic.flags |= QMF_GRAYED;
		} else {
			s_metal.scale.generic.flags &= ~QMF_GRAYED;
		}

		if ( s_metal.renderer.curvalue != s_metal.initRenderer
			|| s_metal.upscale.curvalue != s_metal.initUpscale
			|| s_metal.scale.curvalue != s_metal.initScale ) {
			s_metal.apply.generic.flags &= ~( QMF_HIDDEN | QMF_INACTIVE );
		} else {
			s_metal.apply.generic.flags |= QMF_HIDDEN | QMF_INACTIVE;
		}
	}
}

static void MetalOptions_Event( void *ptr, int event ) {
	if ( event != QM_ACTIVATED ) {
		return;
	}
	switch ( ( (menucommon_s *)ptr )->id ) {
	case ID_BACK:
		UI_PopMenu();
		break;

	case ID_APPLY:
		trap_Cvar_Set( "r_renderer", s_metal.renderer.curvalue ? "opengl" : "metal" );
		trap_Cvar_SetValue( "r_mtlUpscale", s_metal.upscale.curvalue );
		trap_Cvar_SetValue( "r_mtlRenderScale", scale_values[s_metal.scale.curvalue] );
		trap_Cmd_ExecuteText( EXEC_APPEND, "vid_restart\n" );
		break;

	case ID_RTLIGHTS:
		// not latched: takes effect on the next frame
		trap_Cvar_SetValue( "r_rtShadows", s_metal.rtLights.curvalue );
		break;

	case ID_CLAY:
		trap_Cvar_SetValue( "r_clay", s_metal.clay.curvalue );
		break;
	}
}

static void MetalOptions_Draw( void ) {
	const char	*note = NULL;

	MetalOptions_Update();
	Menu_Draw( &s_metal.menu );

	if ( s_metal.tab == TAB_METAL ) {
		UI_DrawString( 400, 240 + 3 * ( BIGCHAR_HEIGHT + 2 ), uis.glconfig.renderer_string,
			UI_CENTER | UI_SMALLFONT, text_color_normal );
	} else if ( !UI_MetalRendererActive() ) {
		note = "Needs the Metal renderer (METAL tab)";
	}
	if ( note ) {
		UI_DrawString( 400, 240 + 2 * ( BIGCHAR_HEIGHT + 2 ), note, UI_CENTER | UI_SMALLFONT, text_color_normal );
	}
}

static void MetalOptions_List( menulist_s *l, int id, const char *name, const char **items, int y ) {
	l->generic.type		= MTYPE_SPINCONTROL;
	l->generic.name		= (char *)name;
	l->generic.flags	= QMF_PULSEIFFOCUS | QMF_SMALLFONT;
	l->generic.callback	= MetalOptions_Event;
	l->generic.id		= id;
	l->generic.x		= 400;
	l->generic.y		= y;
	l->itemnames		= items;
}

static void MetalOptions_Init( int tab ) {
	int		y;
	char	buf[16];

	memset( &s_metal, 0, sizeof( s_metal ) );
	s_metal.tab = tab;
	MetalOptions_Cache();

	s_metal.menu.wrapAround = qtrue;
	s_metal.menu.fullscreen = qtrue;
	s_metal.menu.draw = MetalOptions_Draw;

	s_metal.banner.generic.type		= MTYPE_BTEXT;
	s_metal.banner.generic.flags	= QMF_CENTER_JUSTIFY;
	s_metal.banner.generic.x		= 320;
	s_metal.banner.generic.y		= 16;
	s_metal.banner.string			= "SYSTEM SETUP";
	s_metal.banner.color			= color_white;
	s_metal.banner.style			= UI_CENTER;

	s_metal.framel.generic.type		= MTYPE_BITMAP;
	s_metal.framel.generic.name		= ART_FRAMEL;
	s_metal.framel.generic.flags	= QMF_INACTIVE;
	s_metal.framel.generic.x		= 0;
	s_metal.framel.generic.y		= 78;
	s_metal.framel.width			= 256;
	s_metal.framel.height			= 329;

	s_metal.framer.generic.type		= MTYPE_BITMAP;
	s_metal.framer.generic.name		= ART_FRAMER;
	s_metal.framer.generic.flags	= QMF_INACTIVE;
	s_metal.framer.generic.x		= 376;
	s_metal.framer.generic.y		= 76;
	s_metal.framer.width			= 256;
	s_metal.framer.height			= 334;

	UI_MetalTabs_Item( &s_metal.graphics, ID_TAB_GRAPHICS, "GRAPHICS", 240 - 2 * PROP_HEIGHT, qfalse );
	UI_MetalTabs_Item( &s_metal.display, ID_TAB_DISPLAY, "DISPLAY", 240 - PROP_HEIGHT, qfalse );
	UI_MetalTabs_Item( &s_metal.sound, ID_TAB_SOUND, "SOUND", 240, qfalse );
	UI_MetalTabs_Item( &s_metal.network, ID_TAB_NETWORK, "NETWORK", 240 + PROP_HEIGHT, qfalse );
	UI_MetalTabs_Item( &s_metal.metal, ID_TAB_METAL, "METAL", 240 + 2 * PROP_HEIGHT, tab == TAB_METAL );
	UI_MetalTabs_Item( &s_metal.rt, ID_TAB_RT, "RAY TRACING", 240 + 3 * PROP_HEIGHT, tab == TAB_RT );

	s_metal.back.generic.type		= MTYPE_BITMAP;
	s_metal.back.generic.name		= ART_BACK0;
	s_metal.back.generic.flags		= QMF_LEFT_JUSTIFY | QMF_PULSEIFFOCUS;
	s_metal.back.generic.callback	= MetalOptions_Event;
	s_metal.back.generic.id			= ID_BACK;
	s_metal.back.generic.x			= 0;
	s_metal.back.generic.y			= 480 - 64;
	s_metal.back.width				= 128;
	s_metal.back.height				= 64;
	s_metal.back.focuspic			= ART_BACK1;

	s_metal.apply.generic.type		= MTYPE_BITMAP;
	s_metal.apply.generic.name		= ART_ACCEPT0;
	s_metal.apply.generic.flags		= QMF_RIGHT_JUSTIFY | QMF_PULSEIFFOCUS | QMF_HIDDEN | QMF_INACTIVE;
	s_metal.apply.generic.callback	= MetalOptions_Event;
	s_metal.apply.generic.id		= ID_APPLY;
	s_metal.apply.generic.x			= 640;
	s_metal.apply.generic.y			= 480 - 64;
	s_metal.apply.width				= 128;
	s_metal.apply.height			= 64;
	s_metal.apply.focuspic			= ART_ACCEPT1;

	Menu_AddItem( &s_metal.menu, &s_metal.banner );
	Menu_AddItem( &s_metal.menu, &s_metal.framel );
	Menu_AddItem( &s_metal.menu, &s_metal.framer );
	Menu_AddItem( &s_metal.menu, &s_metal.graphics );
	Menu_AddItem( &s_metal.menu, &s_metal.display );
	Menu_AddItem( &s_metal.menu, &s_metal.sound );
	Menu_AddItem( &s_metal.menu, &s_metal.network );
	Menu_AddItem( &s_metal.menu, &s_metal.metal );
	Menu_AddItem( &s_metal.menu, &s_metal.rt );

	y = 240 - 2 * ( BIGCHAR_HEIGHT + 2 );
	if ( tab == TAB_METAL ) {
		MetalOptions_List( &s_metal.renderer, ID_RENDERER, "Renderer:", renderer_names, y );
		y += BIGCHAR_HEIGHT + 2;
		MetalOptions_List( &s_metal.upscale, ID_UPSCALE, "Upscaling:", upscale_names, y );
		y += BIGCHAR_HEIGHT + 2;
		MetalOptions_List( &s_metal.scale, ID_SCALE, "Render Scale:", scale_names, y );

		trap_Cvar_VariableStringBuffer( "r_renderer", buf, sizeof( buf ) );
		s_metal.renderer.curvalue = Q_stricmp( buf, "opengl" ) == 0;
		s_metal.upscale.curvalue = trap_Cvar_VariableValue( "r_mtlUpscale" ) != 0;
		s_metal.scale.curvalue = MetalOptions_ScaleIndex( trap_Cvar_VariableValue( "r_mtlRenderScale" ) );
		s_metal.initRenderer = s_metal.renderer.curvalue;
		s_metal.initUpscale = s_metal.upscale.curvalue;
		s_metal.initScale = s_metal.scale.curvalue;

		Menu_AddItem( &s_metal.menu, &s_metal.renderer );
		Menu_AddItem( &s_metal.menu, &s_metal.upscale );
		Menu_AddItem( &s_metal.menu, &s_metal.scale );
	} else {
		MetalOptions_List( &s_metal.rtLights, ID_RTLIGHTS, "Ray Traced Lights:", rtlights_names, y );
		y += BIGCHAR_HEIGHT + 2;
		// lighting only, to look at the ray traced light
		MetalOptions_List( &s_metal.clay, ID_CLAY, "Clay Mode:", onoff_names, y );
		s_metal.rtLights.curvalue = trap_Cvar_VariableValue( "r_rtShadows" );
		if ( s_metal.rtLights.curvalue < 0 || s_metal.rtLights.curvalue > 2 ) {
			s_metal.rtLights.curvalue = 1;
		}
		s_metal.clay.curvalue = trap_Cvar_VariableValue( "r_clay" ) != 0;
		if ( !UI_MetalRendererActive() ) {
			s_metal.rtLights.generic.flags |= QMF_GRAYED;
			s_metal.clay.generic.flags |= QMF_GRAYED;
		}
		Menu_AddItem( &s_metal.menu, &s_metal.rtLights );
		Menu_AddItem( &s_metal.menu, &s_metal.clay );
	}

	Menu_AddItem( &s_metal.menu, &s_metal.back );
	Menu_AddItem( &s_metal.menu, &s_metal.apply );
}

void UI_MetalOptionsMenu( void ) {
	MetalOptions_Init( TAB_METAL );
	UI_PushMenu( &s_metal.menu );
	Menu_SetCursorToItem( &s_metal.menu, &s_metal.metal );
}

void UI_RayTracingOptionsMenu( void ) {
	MetalOptions_Init( TAB_RT );
	UI_PushMenu( &s_metal.menu );
	Menu_SetCursorToItem( &s_metal.menu, &s_metal.rt );
}
