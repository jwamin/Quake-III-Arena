/*
===========================================================================
Metal port: the ray tracing hooks of rb_backend.h (docs/metal-rt.md).

  RB_Backend_WorldLoaded  collects the static world's opaque triangles and
                          has MetalRenderer build an acceleration structure.
  RB_Backend_OpaqueDone   after the opaque part of a world view: adds the
                          view's dynamic lights with ray traced shadows
                          (r_rtShadows).
  RB_Backend_RTDlights    tells the renderer not to add dlights itself.
===========================================================================
*/
#include "../renderer/tr_local.h"
#include "mtl_swift.h"

extern cvar_t *r_rtShadows;     // mac_glimp.c
extern cvar_t *r_rtDebugLight;  // mac_glimp.c: "x y z radius" adds a white test light
extern qboolean mac_metalRenderer;  // mac_glimp.c: false with r_renderer opengl
extern cvar_t *r_rtRelight;         // mac_glimp.c: relight from the map's lights
extern cvar_t *r_rtLightBudget;
extern cvar_t *r_rtRelightScale;
extern cvar_t *r_rtSun;

void RT_BuildWorldLights( void );       // mtl_rtlights.c
void MTL_SetRelightMark( int mark );    // mtl_qgl.c

/*
==============================================================================

WORLD GEOMETRY

==============================================================================
*/

/*
** Surfaces that block light: opaque world surfaces. Left out: sky, nodraw,
** everything that is not SS_OPAQUE (portals, decals, translucent, fog) and
** alpha-tested surfaces (grates, foliage), which would shadow as solid
** quads. Brush models (doors, lifts) move, so only model 0 is used.
*/
static qboolean RT_SurfaceCastsShadow( const msurface_t *surf ) {
	const shader_t *sh = surf->shader;

	if ( !sh || sh->isSky || ( sh->surfaceFlags & ( SURF_NODRAW | SURF_SKY ) ) ) {
		return qfalse;
	}
	if ( sh->sort != SS_OPAQUE ) {
		return qfalse;
	}
	if ( sh->stages[0] && ( sh->stages[0]->stateBits & GLS_ATEST_BITS ) ) {
		return qfalse;
	}
	return qtrue;
}

// vertices and triangles one surface contributes
static void RT_CountSurface( const surfaceType_t *data, int *verts, int *indexes ) {
	switch ( *data ) {
	case SF_FACE: {
		const srfSurfaceFace_t *face = (const srfSurfaceFace_t *)data;
		*verts += face->numPoints;
		*indexes += face->numIndices;
		break;
	}
	case SF_GRID: {
		const srfGridMesh_t *grid = (const srfGridMesh_t *)data;
		*verts += grid->width * grid->height;
		*indexes += ( grid->width - 1 ) * ( grid->height - 1 ) * 6;
		break;
	}
	case SF_TRIANGLES: {
		const srfTriangles_t *tris = (const srfTriangles_t *)data;
		*verts += tris->numVerts;
		*indexes += tris->numIndexes;
		break;
	}
	default:
		break;
	}
}

static void RT_AddSurface( const surfaceType_t *data, float *xyz, int *numVerts, uint32_t *idx, int *numIndexes ) {
	int base = *numVerts;
	int i, x, y;

	switch ( *data ) {
	case SF_FACE: {
		const srfSurfaceFace_t *face = (const srfSurfaceFace_t *)data;
		const int *indices = (const int *)( (const byte *)face + face->ofsIndices );
		for ( i = 0; i < face->numPoints; i++ ) {
			VectorCopy( face->points[i], xyz + 3 * ( base + i ) );
		}
		for ( i = 0; i < face->numIndices; i++ ) {
			idx[( *numIndexes )++] = base + indices[i];
		}
		*numVerts += face->numPoints;
		break;
	}
	case SF_GRID: {
		const srfGridMesh_t *grid = (const srfGridMesh_t *)data;
		int w = grid->width, h = grid->height;
		for ( i = 0; i < w * h; i++ ) {
			VectorCopy( grid->verts[i].xyz, xyz + 3 * ( base + i ) );
		}
		for ( y = 0; y < h - 1; y++ ) {
			for ( x = 0; x < w - 1; x++ ) {
				int v0 = base + y * w + x;
				int v1 = v0 + 1, v2 = v0 + w, v3 = v2 + 1;
				idx[( *numIndexes )++] = v0;
				idx[( *numIndexes )++] = v2;
				idx[( *numIndexes )++] = v1;
				idx[( *numIndexes )++] = v1;
				idx[( *numIndexes )++] = v2;
				idx[( *numIndexes )++] = v3;
			}
		}
		*numVerts += w * h;
		break;
	}
	case SF_TRIANGLES: {
		const srfTriangles_t *tris = (const srfTriangles_t *)data;
		for ( i = 0; i < tris->numVerts; i++ ) {
			VectorCopy( tris->verts[i].xyz, xyz + 3 * ( base + i ) );
		}
		for ( i = 0; i < tris->numIndexes; i++ ) {
			idx[( *numIndexes )++] = base + tris->indexes[i];
		}
		*numVerts += tris->numVerts;
		break;
	}
	default:
		break;
	}
}

void RB_Backend_WorldLoaded( void ) {
	const bmodel_t *world;
	float *xyz;
	uint32_t *idx;
	int maxVerts = 0, maxIndexes = 0, numVerts = 0, numIndexes = 0;
	int i;

	if ( !mac_metalRenderer ) {
		return;     // OpenGL: nothing to trace with
	}
	if ( !tr.world || !tr.world->bmodels ) {
		MTLR_SetWorldGeometry( NULL, 0, NULL, 0 );
		return;
	}
	world = &tr.world->bmodels[0];

	for ( i = 0; i < world->numSurfaces; i++ ) {
		if ( RT_SurfaceCastsShadow( &world->firstSurface[i] ) ) {
			RT_CountSurface( world->firstSurface[i].data, &maxVerts, &maxIndexes );
		}
	}
	if ( !maxIndexes ) {
		MTLR_SetWorldGeometry( NULL, 0, NULL, 0 );
		return;
	}

	xyz = ri.Malloc( maxVerts * 3 * sizeof( float ) );
	idx = ri.Malloc( maxIndexes * sizeof( uint32_t ) );
	for ( i = 0; i < world->numSurfaces; i++ ) {
		if ( RT_SurfaceCastsShadow( &world->firstSurface[i] ) ) {
			RT_AddSurface( world->firstSurface[i].data, xyz, &numVerts, idx, &numIndexes );
		}
	}

	if ( !MTLR_SetWorldGeometry( xyz, numVerts, idx, numIndexes ) ) {
		ri.Printf( PRINT_ALL, "Ray tracing is not available on this GPU; r_rtShadows has no effect\n" );
	}

	ri.Free( idx );
	ri.Free( xyz );

	// the map's own lights, for r_rtRelight (built whenever ray tracing works,
	// so the cvar can be toggled live)
	if ( MTLR_HasWorld() ) {
		RT_BuildWorldLights();
	} else {
		MTLR_SetWorldLights( NULL, NULL, NULL, 0, NULL, 0, NULL );
	}
}

/*
==============================================================================

PER VIEW

==============================================================================
*/

qboolean RB_Backend_RTDlights( void ) {
	return ( r_rtShadows && r_rtShadows->integer && MTLR_HasWorld() ) ? qtrue : qfalse;
}

qboolean RB_Backend_RelightActive( void ) {
	return ( mac_metalRenderer && r_rtRelight && r_rtRelight->integer && MTLR_HasWorldLights() ) ? qtrue : qfalse;
}

void RB_Backend_SetRelightMark( int mark ) {
	MTL_SetRelightMark( mark == RB_RELIGHT_SET ? MTL_RELIGHT_SET
	                  : mark == RB_RELIGHT_CLEAR ? MTL_RELIGHT_CLEAR : MTL_RELIGHT_NONE );
}

void RB_Backend_OpaqueDone( void ) {
	static mtl_rt_view_t view;
	const viewParms_t *vp = &backEnd.viewParms;
	int i, n;
	qboolean relight = RB_Backend_RelightActive();

	if ( ( !RB_Backend_RTDlights() && !relight ) || ( backEnd.refdef.rdflags & RDF_NOWORLDMODEL ) ) {
		return;
	}

	Com_Memcpy( view.projection, vp->projectionMatrix, sizeof( view.projection ) );
	Com_Memcpy( view.modelview, vp->world.modelMatrix, sizeof( view.modelview ) );
	VectorCopy( vp->or.origin, view.viewOrigin );
	view.viewport[0] = vp->viewportX;
	view.viewport[1] = vp->viewportY;
	view.viewport[2] = vp->viewportWidth;
	view.viewport[3] = vp->viewportHeight;
	view.numLights = 0;
	view.debugView = 0;

	// relight first: the dynamic lights then go on top of the relit image
	if ( relight ) {
		view.lightBudget = r_rtLightBudget->integer;
		if ( view.lightBudget < 1 ) view.lightBudget = 1;
		if ( view.lightBudget > MTL_RT_MAX_BUDGET ) view.lightBudget = MTL_RT_MAX_BUDGET;
		view.relightScale = r_rtRelightScale->value;
		view.sun = r_rtSun->integer;
		view.relightMode = r_rtRelight->integer;
		MTLR_Relight( &view );
	}
	if ( !RB_Backend_RTDlights() || !r_dynamiclight->integer ) {
		return;
	}

	n = backEnd.refdef.num_dlights;
	if ( n > MTL_RT_MAX_LIGHTS ) {
		n = MTL_RT_MAX_LIGHTS;
	}
	for ( i = 0; i < n; i++ ) {
		const dlight_t *dl = &backEnd.refdef.dlights[i];
		VectorCopy( dl->origin, view.lights[i].origin );
		view.lights[i].radius = dl->radius;
		VectorCopy( dl->color, view.lights[i].color );
		view.lights[i].pad = 0;
	}
	if ( r_rtDebugLight && r_rtDebugLight->string[0] && n < MTL_RT_MAX_LIGHTS ) {
		mtl_rt_light_t *l = &view.lights[n];
		if ( sscanf( r_rtDebugLight->string, "%f %f %f %f", &l->origin[0], &l->origin[1], &l->origin[2],
		             &l->radius ) == 4 && l->radius > 0 ) {
			l->color[0] = l->color[1] = l->color[2] = 1.0f;
			l->pad = 0;
			n++;
		}
	}
	view.debugView = ( r_rtShadows->integer == 2 );
	if ( !n && !view.debugView ) {
		return;
	}
	view.numLights = n;

	MTLR_TraceDlights( &view );
}
