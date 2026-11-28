/*
===========================================================================
rb_gl.c -- OpenGL implementation of the functions declared in
rb_backend.h that used to live as raw qgl* calls inside shared
front-end files.

The bodies are moved verbatim from their original locations
(tr_cmds.c RE_BeginFrame, tr_main.c R_DebugPolygon, tr_shader.c
CollapseMultitexture).  Do not change behaviour here.
===========================================================================
*/
#include "tr_local.h"

/*
===============
RB_Backend_SetOverdrawMeasure

Was inline in RE_BeginFrame (tr_cmds.c).
===============
*/
void RB_Backend_SetOverdrawMeasure( qboolean enable ) {
	if ( enable ) {
		qglEnable( GL_STENCIL_TEST );
		qglStencilMask( ~0U );
		qglClearStencil( 0U );
		qglStencilFunc( GL_ALWAYS, 0U, ~0U );
		qglStencilOp( GL_KEEP, GL_INCR, GL_INCR );
	} else {
		qglDisable( GL_STENCIL_TEST );
	}
}

/*
===============
RB_Backend_GetError

Was inline in RE_BeginFrame (tr_cmds.c).
===============
*/
int RB_Backend_GetError( void ) {
	int		err;

	err = qglGetError();
	if ( err == GL_NO_ERROR ) {
		return 0;
	}
	return err;
}

/*
===============
RB_Backend_MultitextureAvailable

Was "qglActiveTextureARB != NULL" in CollapseMultitexture (tr_shader.c).
===============
*/
qboolean RB_Backend_MultitextureAvailable( void ) {
	return qglActiveTextureARB ? qtrue : qfalse;
}

/*
================
RB_Backend_BeginDebugDraw

Was the GL_Bind/GL_Cull pair at the top of R_DebugGraphics (tr_main.c).
================
*/
void RB_Backend_BeginDebugDraw( void ) {
	GL_Bind( tr.whiteImage);
	GL_Cull( CT_FRONT_SIDED );
}

/*
================
RB_Backend_DebugPolygon

Was R_DebugPolygon (tr_main.c).
================
*/
void RB_Backend_DebugPolygon( int color, int numPoints, float *points ) {
	int		i;

	GL_State( GLS_DEPTHMASK_TRUE | GLS_SRCBLEND_ONE | GLS_DSTBLEND_ONE );

	// draw solid shade

	qglColor3f( color&1, (color>>1)&1, (color>>2)&1 );
	qglBegin( GL_POLYGON );
	for ( i = 0 ; i < numPoints ; i++ ) {
		qglVertex3fv( points + i * 3 );
	}
	qglEnd();

	// draw wireframe outline
	GL_State( GLS_POLYMODE_LINE | GLS_DEPTHMASK_TRUE | GLS_SRCBLEND_ONE | GLS_DSTBLEND_ONE );
	qglDepthRange( 0, 0 );
	qglColor3f( 1, 1, 1 );
	qglBegin( GL_POLYGON );
	for ( i = 0 ; i < numPoints ; i++ ) {
		qglVertex3fv( points + i * 3 );
	}
	qglEnd();
	qglDepthRange( 0, 1 );
}
