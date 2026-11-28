/*
===========================================================================
Metal port: the map's own lights, for ray traced relighting (r_rtRelight).

Retail maps were lit by q3map (q3map/light.c) with direct light only:
light entities (point and spot), light-emitting surfaces (shaders with
q3map_surfacelight) and the sky's sun, plus the worldspawn ambient. This
file rebuilds those lights exactly as q3map defined them, so the GPU can
evaluate q3map's formulas per pixel with ray traced shadows
(rt_relight in Shaders.metal), and builds a static grid that lists, for
each 128-unit cell, the lights that can reach it.

See docs/metal-rt-relight.md.
===========================================================================
*/
#include "../renderer/tr_local.h"
#include <stdlib.h>
#include "mtl_swift.h"

extern cvar_t *r_rtAreaSubdivide;   // mac_glimp.c
extern cvar_t *r_rtCellLights;

void R_LoadImage( const char *name, byte **pic, int *width, int *height );    // tr_image.c

// q3map/light.c constants
#define POINT_SCALE         7500.0f
#define LINEAR_SCALE        ( 1.0f / 8000.0f )
#define AREA_SCALE          0.25f
#define BACKSPLASH_FRACTION 0.05f           // q3map/shaders.c defaults
#define BACKSPLASH_DISTANCE 24.0f

#define MAX_ENV_LIGHTS      32768
#define MAX_RANGE           8192.0f
#define CELL_SIZE           128.0f
#define MAX_CELL_LIGHTS     256             // upper limit for r_rtCellLights
#define SKY_SUBDIVIDE_MIN   256.0f          // sky patches: max(r_rtAreaSubdivide * 4, this)
// area patches stand in for one polygon that q3map lit with an exact form
// factor and no cutoff, so they keep contributing down to a small fraction
#define AREA_MIN_ADD        0.05f

static mtl_rt_envlight_t   *envLights;
static vec3_t               worldColor;     // worldspawn _color, normalized (white if none)
static int                  numEnvLights;
static qboolean             lightsOverflowed;

/*
==============================================================================

LIGHT LIST

==============================================================================
*/

static mtl_rt_envlight_t *RT_NewLight( int type, const vec3_t origin, const vec3_t color, float photons ) {
	mtl_rt_envlight_t *l;

	if ( numEnvLights >= MAX_ENV_LIGHTS ) {
		lightsOverflowed = qtrue;
		return NULL;
	}
	l = &envLights[numEnvLights++];
	Com_Memset( l, 0, sizeof( *l ) );
	l->type = type;
	VectorCopy( origin, l->origin );
	VectorCopy( color, l->color );
	l->photons = photons;
	return l;
}

// q3map ColorNormalize: scale so the largest component is 1
static void RT_ColorNormalize( const vec3_t in, vec3_t out ) {
	float max = in[0];

	if ( in[1] > max ) max = in[1];
	if ( in[2] > max ) max = in[2];
	if ( max <= 0 ) {
		VectorSet( out, 1, 1, 1 );
		return;
	}
	VectorScale( in, 1.0f / max, out );
}

/*
==============================================================================

ENTITY LIGHTS (q3map CreateEntityLights)

==============================================================================
*/

#define MAX_PARSED_ENTS 4096

typedef struct {
	char    classname[64];
	char    targetname[64];
	char    target[64];
	vec3_t  origin;
	vec3_t  color;
	qboolean hasColor;
	float   light;
	float   radius;
	int     spawnflags;
	float   ambient;
} rtEnt_t;

static void RT_ParseEntities( rtEnt_t *ents, int *numEnts ) {
	char    *p = tr.world->entityString;
	char    *token;
	char    key[MAX_TOKEN_CHARS];
	rtEnt_t *e;

	*numEnts = 0;
	while ( p ) {
		token = COM_ParseExt( &p, qtrue );
		if ( !token[0] ) {
			break;
		}
		if ( token[0] != '{' || *numEnts >= MAX_PARSED_ENTS ) {
			break;
		}
		e = &ents[( *numEnts )++];
		Com_Memset( e, 0, sizeof( *e ) );
		while ( 1 ) {
			token = COM_ParseExt( &p, qtrue );
			if ( !token[0] || token[0] == '}' ) {
				break;
			}
			Q_strncpyz( key, token, sizeof( key ) );
			token = COM_ParseExt( &p, qtrue );
			if ( !token[0] || token[0] == '}' ) {
				break;
			}
			if ( !Q_stricmp( key, "classname" ) ) {
				Q_strncpyz( e->classname, token, sizeof( e->classname ) );
			} else if ( !Q_stricmp( key, "targetname" ) ) {
				Q_strncpyz( e->targetname, token, sizeof( e->targetname ) );
			} else if ( !Q_stricmp( key, "target" ) ) {
				Q_strncpyz( e->target, token, sizeof( e->target ) );
			} else if ( !Q_stricmp( key, "origin" ) ) {
				sscanf( token, "%f %f %f", &e->origin[0], &e->origin[1], &e->origin[2] );
			} else if ( !Q_stricmp( key, "_color" ) ) {
				e->hasColor = sscanf( token, "%f %f %f", &e->color[0], &e->color[1], &e->color[2] ) == 3;
			} else if ( !Q_stricmp( key, "light" ) || ( !Q_stricmp( key, "_light" ) && !e->light ) ) {
				e->light = atof( token );
			} else if ( !Q_stricmp( key, "radius" ) ) {
				e->radius = atof( token );
			} else if ( !Q_stricmp( key, "spawnflags" ) ) {
				e->spawnflags = atoi( token );
			} else if ( !Q_stricmp( key, "ambient" ) ) {
				e->ambient = atof( token );
			}
		}
	}
}

static void RT_AddEntityLights( vec3_t ambient ) {
	static rtEnt_t ents[MAX_PARSED_ENTS];
	int numEnts, i, j;

	VectorClear( ambient );
	if ( !tr.world->entityString ) {
		return;
	}
	RT_ParseEntities( ents, &numEnts );

	// worldspawn: ambient = _color * ambient (q3map light.c)
	VectorSet( worldColor, 1, 1, 1 );
	if ( numEnts && !Q_stricmp( ents[0].classname, "worldspawn" ) && ents[0].hasColor ) {
		VectorScale( ents[0].color, ents[0].ambient, ambient );
		RT_ColorNormalize( ents[0].color, worldColor );
	}

	for ( i = 0; i < numEnts; i++ ) {
		rtEnt_t *e = &ents[i];
		mtl_rt_envlight_t *l;
		vec3_t color;
		float intensity;

		if ( Q_stricmpn( e->classname, "light", 5 ) ) {
			continue;
		}
		intensity = e->light ? e->light : 300;
		if ( e->hasColor ) {
			RT_ColorNormalize( e->color, color );
		} else {
			VectorSet( color, 1, 1, 1 );
		}
		l = RT_NewLight( MTL_ENVLIGHT_POINT, e->origin, color, intensity * POINT_SCALE );
		if ( !l ) {
			return;
		}
		l->linear = ( e->spawnflags & 1 ) != 0;

		// lights with a target are spotlights
		if ( e->target[0] ) {
			for ( j = 0; j < numEnts; j++ ) {
				if ( !strcmp( ents[j].targetname, e->target ) ) {
					break;
				}
			}
			if ( j < numEnts ) {
				vec3_t dir;
				float dist, radius;

				VectorSubtract( ents[j].origin, e->origin, dir );
				dist = VectorNormalize( dir );
				radius = e->radius ? e->radius : 64;
				if ( !dist ) {
					dist = 64;
				}
				VectorCopy( dir, l->normal );
				l->radiusByDist = ( radius + 16 ) / dist;
				l->type = MTL_ENVLIGHT_SPOT;
			}
		}
	}
}

/*
==============================================================================

SURFACE LIGHTS (q3map CreateSurfaceLights / SubdivideAreaLight)

==============================================================================
*/

// q3map's light colour: average of q3map_lightimage, qer_editorimage or the
// shader's own image, normalized. Retail paks lack many editor images (the
// q3dm1 sky's stars_red.tga, for one). Then the shader's first stage texture
// (the image actually seen) stands in, except for skies: their cloud
// textures normalize to an almost pure colour far from what the maps were
// lit with, so they use the worldspawn _color (the ambient tint the designer
// matched to the sky), or q3map's own fallback, white.
static void RT_ShaderLightColor( const shader_t *sh, vec3_t color ) {
	const char *names[4];
	char name[MAX_QPATH];
	byte *pic = NULL;
	int w = 0, h = 0, i, n, k;
	double sum[3];

	names[0] = sh->lightImage;
	names[1] = sh->editorImage;
	names[2] = sh->name;
	names[3] = ( !sh->isSky && sh->stages[0] && sh->stages[0]->bundle[0].image[0] )
	         ? sh->stages[0]->bundle[0].image[0]->imgName : "";
	for ( n = 0; n < 4 && !pic; n++ ) {
		if ( !names[n][0] ) {
			continue;
		}
		Q_strncpyz( name, names[n], sizeof( name ) );
		R_LoadImage( name, &pic, &w, &h );
		if ( !pic ) {
			COM_StripExtension( names[n], name );
			Q_strcat( name, sizeof( name ), ".tga" );
			R_LoadImage( name, &pic, &w, &h );
		}
		if ( !pic ) {
			COM_StripExtension( names[n], name );
			Q_strcat( name, sizeof( name ), ".jpg" );
			R_LoadImage( name, &pic, &w, &h );
		}
	}
	if ( !pic ) {
		if ( sh->isSky ) {
			VectorCopy( worldColor, color );
		} else {
			VectorSet( color, 1, 1, 1 );
		}
		return;
	}
	sum[0] = sum[1] = sum[2] = 0;
	for ( i = 0; i < w * h; i++ ) {
		for ( k = 0; k < 3; k++ ) {
			sum[k] += pic[i * 4 + k];
		}
	}
	ri.Free( pic );
	color[0] = sum[0];
	color[1] = sum[1];
	color[2] = sum[2];
	RT_ColorNormalize( color, color );
}

typedef struct {
	const shader_t  *shader;
	vec3_t          color;
	float           subdivide;
	qboolean        twoSided;
} rtEmitter_t;

static void RT_EmitPatch( const rtEmitter_t *em, const vec3_t a, const vec3_t b, const vec3_t c,
                          const vec3_t normal, float area ) {
	const shader_t *sh = em->shader;
	mtl_rt_envlight_t *l;
	vec3_t origin;
	float photons, fraction, distance;

	VectorAdd( a, b, origin );
	VectorAdd( origin, c, origin );
	VectorScale( origin, 1.0f / 3.0f, origin );
	VectorAdd( origin, normal, origin );     // q3map pushes the light off the surface

	photons = sh->surfaceLight * area * AREA_SCALE;
	l = RT_NewLight( MTL_ENVLIGHT_AREA, origin, em->color, photons );
	if ( !l ) {
		return;
	}
	VectorCopy( normal, l->normal );
	l->twoSided = em->twoSided;

	// backsplash: a weaker point light in front of the surface
	fraction = sh->hasBacksplash ? sh->backsplashFraction : BACKSPLASH_FRACTION;
	distance = sh->hasBacksplash ? sh->backsplashDistance : BACKSPLASH_DISTANCE;
	if ( fraction > 0 ) {
		vec3_t splash;
		VectorMA( origin, distance, normal, splash );
		RT_NewLight( MTL_ENVLIGHT_POINT, splash, em->color, photons * fraction );
	}
}

// split the triangle until no edge is longer than the subdivision size
static void RT_SubdivideTriangle( const rtEmitter_t *em, const vec3_t a, const vec3_t b, const vec3_t c,
                                  const vec3_t normal, int depth ) {
	vec3_t ab, bc, ca, cross, mid;
	float lab, lbc, lca, area;

	VectorSubtract( b, a, ab );
	VectorSubtract( c, b, bc );
	VectorSubtract( a, c, ca );
	lab = VectorLength( ab );
	lbc = VectorLength( bc );
	lca = VectorLength( ca );

	if ( depth < 16 && ( lab > em->subdivide || lbc > em->subdivide || lca > em->subdivide ) ) {
		if ( lab >= lbc && lab >= lca ) {
			VectorAdd( a, b, mid ); VectorScale( mid, 0.5f, mid );
			RT_SubdivideTriangle( em, a, mid, c, normal, depth + 1 );
			RT_SubdivideTriangle( em, mid, b, c, normal, depth + 1 );
		} else if ( lbc >= lca ) {
			VectorAdd( b, c, mid ); VectorScale( mid, 0.5f, mid );
			RT_SubdivideTriangle( em, a, b, mid, normal, depth + 1 );
			RT_SubdivideTriangle( em, a, mid, c, normal, depth + 1 );
		} else {
			VectorAdd( c, a, mid ); VectorScale( mid, 0.5f, mid );
			RT_SubdivideTriangle( em, a, b, mid, normal, depth + 1 );
			RT_SubdivideTriangle( em, mid, b, c, normal, depth + 1 );
		}
		return;
	}
	CrossProduct( ab, ca, cross );
	area = 0.5f * VectorLength( cross );
	if ( area > 0.01f ) {
		RT_EmitPatch( em, a, b, c, normal, area );
	}
}

// one triangle of an emitting surface; reference is the surface's facing
static void RT_EmitTriangle( const rtEmitter_t *em, const vec3_t a, const vec3_t b, const vec3_t c,
                             const vec3_t reference ) {
	vec3_t e1, e2, normal, back;

	VectorSubtract( b, a, e1 );
	VectorSubtract( c, a, e2 );
	CrossProduct( e1, e2, normal );
	if ( VectorNormalize( normal ) == 0 ) {
		return;
	}
	if ( DotProduct( normal, reference ) < 0 ) {
		VectorNegate( normal, normal );
	}
	RT_SubdivideTriangle( em, a, b, c, normal, 0 );
	if ( em->twoSided ) {
		VectorNegate( normal, back );
		RT_SubdivideTriangle( em, a, b, c, back, 0 );
	}
}

static qboolean RT_IsAutosprite( const shader_t *sh ) {
	int i;
	for ( i = 0; i < sh->numDeforms; i++ ) {
		if ( sh->deforms[i].deformation == DEFORM_AUTOSPRITE || sh->deforms[i].deformation == DEFORM_AUTOSPRITE2 ) {
			return qtrue;
		}
	}
	return qfalse;
}

static void RT_AddSurfaceLight( const msurface_t *surf, rtEmitter_t *em ) {
	const surfaceType_t *data = surf->data;
	int i, x, y;

	switch ( *data ) {
	case SF_FACE: {
		const srfSurfaceFace_t *face = (const srfSurfaceFace_t *)data;
		const int *idx = (const int *)( (const byte *)face + face->ofsIndices );

		if ( RT_IsAutosprite( em->shader ) ) {
			vec3_t center = { 0, 0, 0 };
			for ( i = 0; i < face->numPoints; i++ ) {
				VectorAdd( center, face->points[i], center );
			}
			if ( face->numPoints ) {
				VectorScale( center, 1.0f / face->numPoints, center );
				RT_NewLight( MTL_ENVLIGHT_POINT, center, em->color, em->shader->surfaceLight * POINT_SCALE );
			}
			return;
		}
		for ( i = 0; i + 2 < face->numIndices; i += 3 ) {
			RT_EmitTriangle( em, face->points[idx[i]], face->points[idx[i + 1]], face->points[idx[i + 2]],
			                 face->plane.normal );
		}
		break;
	}
	case SF_GRID: {
		const srfGridMesh_t *grid = (const srfGridMesh_t *)data;
		int w = grid->width;
		for ( y = 0; y < grid->height - 1; y++ ) {
			for ( x = 0; x < w - 1; x++ ) {
				const drawVert_t *v0 = &grid->verts[y * w + x], *v1 = v0 + 1, *v2 = v0 + w, *v3 = v2 + 1;
				vec3_t n;
				VectorAdd( v0->normal, v3->normal, n );
				RT_EmitTriangle( em, v0->xyz, v2->xyz, v1->xyz, n );
				RT_EmitTriangle( em, v1->xyz, v2->xyz, v3->xyz, n );
			}
		}
		break;
	}
	case SF_TRIANGLES: {
		const srfTriangles_t *tris = (const srfTriangles_t *)data;
		for ( i = 0; i + 2 < tris->numIndexes; i += 3 ) {
			const drawVert_t *a = &tris->verts[tris->indexes[i]];
			const drawVert_t *b = &tris->verts[tris->indexes[i + 1]];
			const drawVert_t *c = &tris->verts[tris->indexes[i + 2]];
			vec3_t n;
			VectorAdd( a->normal, b->normal, n );
			VectorAdd( n, c->normal, n );
			RT_EmitTriangle( em, a->xyz, b->xyz, c->xyz, n );
		}
		break;
	}
	default:
		break;
	}
}

static void RT_AddSurfaceLights( int *numEmitters ) {
	const bmodel_t *world = &tr.world->bmodels[0];
	const shader_t *lastShader = NULL;
	rtEmitter_t em;
	float subdivide = r_rtAreaSubdivide ? r_rtAreaSubdivide->value : 128;
	int i;

	if ( subdivide < 16 ) {
		subdivide = 16;
	}
	*numEmitters = 0;
	Com_Memset( &em, 0, sizeof( em ) );
	for ( i = 0; i < world->numSurfaces; i++ ) {
		const msurface_t *surf = &world->firstSurface[i];
		const shader_t *sh = surf->shader;

		if ( !sh || sh->surfaceLight <= 0 ) {
			continue;
		}
		if ( sh != lastShader ) {
			em.shader = sh;
			RT_ShaderLightColor( sh, em.color );
			ri.Printf( PRINT_DEVELOPER, "RT relight: surface light %s: %g, colour %.2f %.2f %.2f%s\n", sh->name,
			           sh->surfaceLight, em.color[0], em.color[1], em.color[2], sh->isSky ? " (sky)" : "" );
			em.twoSided = sh->cullType == CT_TWO_SIDED;
			// sky emitters are far from what they light: coarser patches
			em.subdivide = sh->isSky ? ( subdivide * 4 > SKY_SUBDIVIDE_MIN ? subdivide * 4 : SKY_SUBDIVIDE_MIN ) : subdivide;
			if ( sh->lightSubdivide > 0 && sh->lightSubdivide < em.subdivide ) {
				em.subdivide = sh->lightSubdivide;
			}
			lastShader = sh;
		}
		RT_AddSurfaceLight( surf, &em );
		( *numEmitters )++;
	}
}

/*
==============================================================================

LIGHT GRID

==============================================================================
*/

// distance at which q3map's add falls to its cutoff (1 for point and spot
// lights; area patches go lower, see AREA_MIN_ADD)
static float RT_LightRange( const mtl_rt_envlight_t *l ) {
	float range = l->linear ? l->photons * LINEAR_SCALE
	            : l->type == MTL_ENVLIGHT_AREA ? sqrt( l->photons / AREA_MIN_ADD ) : sqrt( l->photons );
	return range > MAX_RANGE ? MAX_RANGE : range;
}

/*
** q3map's add for one light at a sample (q3map/light.c LightingAtSample,
** without occlusion); the GPU twin is q3mapAdd in Shaders.metal
*/
static float RT_Q3mapAdd( const mtl_rt_envlight_t *l, const vec3_t P, const vec3_t N ) {
	vec3_t L, dir;
	float dist, d, angle, add;

	VectorSubtract( l->origin, P, L );
	dist = VectorLength( L );
	if ( dist < 1e-4f ) {
		return 0;
	}
	VectorScale( L, 1.0f / dist, dir );
	d = dist < 16 ? 16 : dist;
	angle = DotProduct( N, dir );

	if ( l->type == MTL_ENVLIGHT_SPOT ) {
		vec3_t pointAtDist, toSample;
		float distByNormal = -DotProduct( L, l->normal ), radiusAtDist, sampleRadius, cone;
		if ( distByNormal < 0 ) {
			return 0;
		}
		VectorMA( l->origin, distByNormal, l->normal, pointAtDist );
		radiusAtDist = l->radiusByDist * distByNormal;
		VectorSubtract( P, pointAtDist, toSample );
		sampleRadius = VectorLength( toSample );
		if ( sampleRadius >= radiusAtDist ) {
			return 0;
		}
		cone = sampleRadius <= radiusAtDist - 32 ? 1.0f : ( radiusAtDist - sampleRadius ) / 32.0f;
		add = l->photons / ( d * d ) * angle * cone;
		return add > 1 ? add : 0;
	}
	if ( l->type == MTL_ENVLIGHT_AREA ) {
		float facing;
		if ( angle <= 0 ) {
			return 0;
		}
		facing = -DotProduct( l->normal, dir );
		if ( l->twoSided ) {
			facing = fabs( facing );
		}
		angle *= facing;
		if ( angle <= 0 ) {
			return 0;
		}
	}
	if ( l->linear ) {
		add = angle * l->photons * LINEAR_SCALE - d;
	} else {
		add = l->photons / ( d * d ) * angle;
	}
	// q3map's cutoff for point lights; area lights have none
	if ( l->type == MTL_ENVLIGHT_AREA ) {
		return add > 0 ? add : 0;
	}
	return add > 1 ? add : 0;
}

typedef struct {
	int     light;
	float   score;
} rtCellEntry_t;

static int RT_CompareEntries( const void *a, const void *b ) {
	float sa = ( (const rtCellEntry_t *)a )->score, sb = ( (const rtCellEntry_t *)b )->score;
	return sa < sb ? 1 : sa > sb ? -1 : 0;
}

// The build buffers can reach tens of megabytes on light-heavy maps, more
// than the engine's zone allocator holds, so they come from the system heap.
void RT_BuildWorldLights( void ) {
	mtl_rt_envinfo_t info;
	const bmodel_t *world;
	vec3_t mins, maxs;
	int dims[3], numCells, i, x, y, z, numEmitters, numEntityLights, total = 0, maxPerCell, cappedCells;
	int *counts;
	uint32_t *cellStart = NULL, *indexes;
	float *cubes;
	int cellLimit = r_rtCellLights ? r_rtCellLights->integer : 48;
	static const vec3_t axes[6] = { { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 } };
	rtCellEntry_t *entries = NULL, *cellBuf;

	if ( !tr.world || !tr.world->bmodels ) {
		MTLR_SetWorldLights( NULL, NULL, NULL, 0, NULL, 0, NULL );
		return;
	}
	world = &tr.world->bmodels[0];

	envLights = malloc( MAX_ENV_LIGHTS * sizeof( *envLights ) );
	numEnvLights = 0;
	lightsOverflowed = qfalse;
	Com_Memset( &info, 0, sizeof( info ) );

	RT_AddEntityLights( info.ambient );
	numEntityLights = numEnvLights;
	RT_AddSurfaceLights( &numEmitters );

	for ( z = 0; z < numEnvLights; z++ ) {
		envLights[z].range = RT_LightRange( &envLights[z] );
	}

	// sun (tr_shader.c parses q3map_sun into tr.sunLight / tr.sunDirection)
	VectorCopy( tr.sunLight, info.sunColor );
	VectorCopy( tr.sunDirection, info.sunDir );
	info.lightScale = (float)( 1 << ( r_mapOverBrightBits->integer - tr.overbrightBits ) ) / 255.0f;

	// grid over the world bounds
	VectorCopy( world->bounds[0], mins );
	VectorCopy( world->bounds[1], maxs );
	for ( x = 0; x < 3; x++ ) {
		dims[x] = (int)ceil( ( maxs[x] - mins[x] ) / CELL_SIZE );
		if ( dims[x] < 1 ) dims[x] = 1;
		info.gridOrigin[x] = mins[x];
		info.gridDims[x] = dims[x];
	}
	info.cellSize = CELL_SIZE;
	info.numLights = numEnvLights;
	numCells = dims[0] * dims[1] * dims[2];

	// pass 1: count, pass 2: fill (light, score) per cell, then sort and cap
	counts = malloc( numCells * sizeof( int ) );
	Com_Memset( counts, 0, numCells * sizeof( int ) );
	for ( z = 0; z < 2; z++ ) {
		int pass = z;
		if ( pass == 1 ) {
			total = 0;
			for ( x = 0; x < numCells; x++ ) {
				total += counts[x];
			}
			entries = malloc( ( total ? total : 1 ) * sizeof( rtCellEntry_t ) );
			cellStart = malloc( ( numCells + 1 ) * sizeof( uint32_t ) );
			cellStart[0] = 0;
			for ( x = 0; x < numCells; x++ ) {
				cellStart[x + 1] = cellStart[x] + counts[x];
				counts[x] = 0;
			}
		}
		for ( i = 0; i < numEnvLights; i++ ) {
			const mtl_rt_envlight_t *l = &envLights[i];
			int lo[3], hi[3], c[3];
			for ( x = 0; x < 3; x++ ) {
				lo[x] = (int)floor( ( l->origin[x] - l->range - mins[x] ) / CELL_SIZE );
				hi[x] = (int)floor( ( l->origin[x] + l->range - mins[x] ) / CELL_SIZE );
				if ( lo[x] < 0 ) lo[x] = 0;
				if ( hi[x] > dims[x] - 1 ) hi[x] = dims[x] - 1;
			}
			for ( c[2] = lo[2]; c[2] <= hi[2]; c[2]++ ) {
				for ( c[1] = lo[1]; c[1] <= hi[1]; c[1]++ ) {
					for ( c[0] = lo[0]; c[0] <= hi[0]; c[0]++ ) {
						// sphere vs cell box
						float d2 = 0;
						int cell;
						for ( x = 0; x < 3; x++ ) {
							float bmin = mins[x] + c[x] * CELL_SIZE, bmax = bmin + CELL_SIZE, v = l->origin[x];
							if ( v < bmin ) d2 += ( bmin - v ) * ( bmin - v );
							else if ( v > bmax ) d2 += ( v - bmax ) * ( v - bmax );
						}
						if ( d2 > l->range * l->range ) {
							continue;
						}
						cell = c[0] + c[1] * dims[0] + c[2] * dims[0] * dims[1];
						if ( pass == 1 ) {
							rtCellEntry_t *ent = &entries[cellStart[cell] + counts[cell]];
							// a global priority, not the strength at this cell, so
							// neighbouring cells pick the same lights and nothing
							// jumps between traced and ambient cube at cell edges:
							// map lights (point, spot) first, then area patches,
							// each by strength
							ent->light = i;
							ent->score = ( l->type == MTL_ENVLIGHT_AREA ? 0.0f : 1e12f ) + l->photons;
						}
						counts[cell]++;
					}
				}
			}
		}
	}

	// strongest first; the first cellLimit lights of a cell are evaluated per
	// pixel with shadow rays, the rest are folded into the cell's ambient
	// cube (unshadowed light from +x -x +y -y +z -z at the cell centre)
	if ( cellLimit < 1 ) cellLimit = 1;
	if ( cellLimit > MAX_CELL_LIGHTS ) cellLimit = MAX_CELL_LIGHTS;
	cubes = malloc( numCells * 6 * 4 * sizeof( float ) );
	Com_Memset( cubes, 0, numCells * 6 * 4 * sizeof( float ) );
	indexes = malloc( ( total ? total : 1 ) * sizeof( uint32_t ) );
	cellBuf = entries;
	total = 0;
	maxPerCell = 0;
	cappedCells = 0;
	for ( x = 0; x < numCells; x++ ) {
		int n = counts[x];
		rtCellEntry_t *list = &cellBuf[cellStart[x]];
		qsort( list, n, sizeof( *list ), RT_CompareEntries );
		if ( n > cellLimit ) {
			vec3_t center;
			int cx = x % dims[0], cy = ( x / dims[0] ) % dims[1], cz = x / ( dims[0] * dims[1] ), a;

			center[0] = mins[0] + ( cx + 0.5f ) * CELL_SIZE;
			center[1] = mins[1] + ( cy + 0.5f ) * CELL_SIZE;
			center[2] = mins[2] + ( cz + 0.5f ) * CELL_SIZE;
			for ( y = cellLimit; y < n; y++ ) {
				const mtl_rt_envlight_t *l = &envLights[list[y].light];
				for ( a = 0; a < 6; a++ ) {
					float add = RT_Q3mapAdd( l, center, axes[a] );
					if ( add > 0 ) {
						float *c = &cubes[( x * 6 + a ) * 4];
						c[0] += l->color[0] * add;
						c[1] += l->color[1] * add;
						c[2] += l->color[2] * add;
					}
				}
			}
			n = cellLimit;
			cappedCells++;
		}
		if ( n > maxPerCell ) {
			maxPerCell = n;
		}
		counts[x] = n;
		for ( y = 0; y < n; y++ ) {
			indexes[total + y] = list[y].light;
		}
		total += n;
	}
	cellStart[0] = 0;
	for ( x = 0; x < numCells; x++ ) {
		cellStart[x + 1] = cellStart[x] + counts[x];
	}

	ri.Printf( PRINT_ALL, "RT relight: %d lights (%d from surface lights on %d surfaces)%s, sun %s, "
	           "%d cells, %.1f lights/cell avg, %d max%s\n",
	           numEnvLights, numEnvLights - numEntityLights, numEmitters, lightsOverflowed ? " [light limit reached]" : "",
	           VectorLength( info.sunColor ) > 0 ? "on" : "none", numCells, (float)total / numCells, maxPerCell,
	           cappedCells ? va( ", %d cells with an ambient cube", cappedCells ) : "" );

	MTLR_SetWorldLights( &info, envLights, cellStart, numCells, indexes, total, cubes );

	free( cubes );
	free( indexes );
	free( entries );
	free( cellStart );
	free( counts );
	free( envLights );
	envLights = NULL;
}
