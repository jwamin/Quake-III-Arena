/*
===========================================================================
Metal port: the renderer's qgl* entry points, implemented on Metal.

The Quake III renderer drives a small fixed-function OpenGL 1.x subset:
state toggles, two texture units with texture environments, client vertex
arrays plus a little immediate mode, a matrix stack, a user clip plane and
stencil. This file keeps that state in plain C and, at each draw, turns it
into one mtl_state_t plus a packed vertex/index stream that the Swift Metal
backend (MetalRenderer.swift) encodes. No OpenGL library is involved.

Only the calls the renderer makes are implemented; QGL_Init binds them to
the qgl* function pointers declared in renderer/qgl.h.
===========================================================================
*/
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <dlfcn.h>

#include "../renderer/tr_local.h"
#include "mtl_swift.h"

#define MAX_MATRIX_DEPTH   32

// relight stencil mark for the next draws (MTL_RELIGHT_*), set by mtl_rt.c
static int relightMark;

void MTL_SetRelightMark( int mark ) {
	relightMark = mark;
}
#define MAX_TEXNUM         16384

typedef struct {
	qboolean    enabled;
	int         size;       // components
	GLenum      type;
	int         stride;     // bytes, already resolved from 0
	const byte  *pointer;
} clientArray_t;

typedef struct {
	GLenum  minFilter, magFilter, wrapS, wrapT;
} texParams_t;

static struct {
	// matrices
	int         matrixMode;             // 0 modelview, 1 projection
	float       stack[2][MAX_MATRIX_DEPTH][16];
	int         depth[2];

	// capabilities
	qboolean    blend, depthTest, alphaTest, cullFace, clipPlane0;
	qboolean    polygonOffsetFill, stencilTest, scissorTest;
	qboolean    texture2D[2];

	GLenum      blendSrc, blendDst;
	GLenum      depthFunc;
	GLenum      alphaFunc;
	float       alphaRef;
	GLboolean   depthMask;
	GLboolean   colorMask[4];
	GLenum      cullFaceMode;
	GLenum      polygonMode;
	float       offsetFactor, offsetUnits;

	GLenum      stencilFunc;
	int         stencilRef;
	GLuint      stencilValueMask, stencilWriteMask;
	GLenum      stencilFail, stencilZFail, stencilZPass;

	int         viewport[4];
	float       depthNear, depthFar;
	int         scissor[4];

	float       clearColor[4];
	float       clearDepth;
	int         clearStencil;

	float       eyePlane[4];            // clip plane 0 in eye space

	// textures
	int         activeTexture, clientActiveTexture;
	GLuint      bound[2];
	GLenum      texEnv[2];

	// client arrays
	clientArray_t vertexArray, colorArray, texCoordArray[2];
	int         lockFirst, lockCount;

	// immediate mode
	GLenum      primitive;
	qboolean    inBegin;
	byte        color[4];
	float       texCoord[2][2];
	mtl_vertex_t *imm;
	int         immCount, immMax;

	texParams_t *texParams;             // per texture name
} gl;

/*
==============================================================================

MATRICES

==============================================================================
*/

static void Mat_Identity( float *m ) {
	memset( m, 0, 16 * sizeof( float ) );
	m[0] = m[5] = m[10] = m[15] = 1.0f;
}

// out = a * b, column-major (out may not alias a or b)
static void Mat_Multiply( const float *a, const float *b, float *out ) {
	int c, r;
	for ( c = 0; c < 4; c++ ) {
		for ( r = 0; r < 4; r++ ) {
			out[c * 4 + r] = a[0 * 4 + r] * b[c * 4 + 0] + a[1 * 4 + r] * b[c * 4 + 1]
			               + a[2 * 4 + r] * b[c * 4 + 2] + a[3 * 4 + r] * b[c * 4 + 3];
		}
	}
}

// general 4x4 inverse; returns qfalse when singular
static qboolean Mat_Inverse( const float *m, float *out ) {
	float inv[16], det;
	int i;

	inv[0] = m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
	inv[4] = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
	inv[8] = m[4]*m[9]*m[15] - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
	inv[12] = -m[4]*m[9]*m[14] + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
	inv[1] = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
	inv[5] = m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
	inv[9] = -m[0]*m[9]*m[15] + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
	inv[13] = m[0]*m[9]*m[14] - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
	inv[2] = m[1]*m[6]*m[15] - m[1]*m[7]*m[14] - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[7] - m[13]*m[3]*m[6];
	inv[6] = -m[0]*m[6]*m[15] + m[0]*m[7]*m[14] + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[7] + m[12]*m[3]*m[6];
	inv[10] = m[0]*m[5]*m[15] - m[0]*m[7]*m[13] - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1]*m[7] - m[12]*m[3]*m[5];
	inv[14] = -m[0]*m[5]*m[14] + m[0]*m[6]*m[13] + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[6] + m[12]*m[2]*m[5];
	inv[3] = -m[1]*m[6]*m[11] + m[1]*m[7]*m[10] + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[9]*m[2]*m[7] + m[9]*m[3]*m[6];
	inv[7] = m[0]*m[6]*m[11] - m[0]*m[7]*m[10] - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[8]*m[2]*m[7] - m[8]*m[3]*m[6];
	inv[11] = -m[0]*m[5]*m[11] + m[0]*m[7]*m[9] + m[4]*m[1]*m[11] - m[4]*m[3]*m[9] - m[8]*m[1]*m[7] + m[8]*m[3]*m[5];
	inv[15] = m[0]*m[5]*m[10] - m[0]*m[6]*m[9] - m[4]*m[1]*m[10] + m[4]*m[2]*m[9] + m[8]*m[1]*m[6] - m[8]*m[2]*m[5];

	det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
	if ( det == 0 ) {
		return qfalse;
	}
	det = 1.0f / det;
	for ( i = 0; i < 16; i++ ) {
		out[i] = inv[i] * det;
	}
	return qtrue;
}

static float *CurrentMatrix( void ) {
	return gl.stack[gl.matrixMode][gl.depth[gl.matrixMode]];
}

static void MultCurrent( const float *m ) {
	float tmp[16];
	Mat_Multiply( CurrentMatrix(), m, tmp );
	memcpy( CurrentMatrix(), tmp, sizeof( tmp ) );
}

static void APIENTRY mtl_MatrixMode( GLenum mode ) {
	gl.matrixMode = ( mode == GL_PROJECTION ) ? 1 : 0;
}

static void APIENTRY mtl_LoadIdentity( void ) {
	Mat_Identity( CurrentMatrix() );
}

static void APIENTRY mtl_LoadMatrixf( const GLfloat *m ) {
	memcpy( CurrentMatrix(), m, 16 * sizeof( float ) );
}

static void APIENTRY mtl_PushMatrix( void ) {
	int mode = gl.matrixMode;
	if ( gl.depth[mode] + 1 >= MAX_MATRIX_DEPTH ) {
		return;
	}
	memcpy( gl.stack[mode][gl.depth[mode] + 1], gl.stack[mode][gl.depth[mode]], 16 * sizeof( float ) );
	gl.depth[mode]++;
}

static void APIENTRY mtl_PopMatrix( void ) {
	if ( gl.depth[gl.matrixMode] > 0 ) {
		gl.depth[gl.matrixMode]--;
	}
}

static void APIENTRY mtl_Ortho( GLdouble l, GLdouble r, GLdouble b, GLdouble t, GLdouble n, GLdouble f ) {
	float m[16];
	Mat_Identity( m );
	m[0] = 2.0f / ( r - l );
	m[5] = 2.0f / ( t - b );
	m[10] = -2.0f / ( f - n );
	m[12] = -( r + l ) / ( r - l );
	m[13] = -( t + b ) / ( t - b );
	m[14] = -( f + n ) / ( f - n );
	MultCurrent( m );
}

static void APIENTRY mtl_Translatef( GLfloat x, GLfloat y, GLfloat z ) {
	float m[16];
	Mat_Identity( m );
	m[12] = x;
	m[13] = y;
	m[14] = z;
	MultCurrent( m );
}

/*
==============================================================================

STATE

==============================================================================
*/

static void SetCap( GLenum cap, qboolean on ) {
	switch ( cap ) {
	case GL_BLEND:               gl.blend = on; break;
	case GL_DEPTH_TEST:          gl.depthTest = on; break;
	case GL_ALPHA_TEST:          gl.alphaTest = on; break;
	case GL_CULL_FACE:           gl.cullFace = on; break;
	case GL_CLIP_PLANE0:         gl.clipPlane0 = on; break;
	case GL_POLYGON_OFFSET_FILL: gl.polygonOffsetFill = on; break;
	case GL_STENCIL_TEST:        gl.stencilTest = on; break;
	case GL_SCISSOR_TEST:        gl.scissorTest = on; break;
	case GL_TEXTURE_2D:          gl.texture2D[gl.activeTexture] = on; break;
	default: break;
	}
}

static void APIENTRY mtl_Enable( GLenum cap )  { SetCap( cap, qtrue ); }
static void APIENTRY mtl_Disable( GLenum cap ) { SetCap( cap, qfalse ); }

static void APIENTRY mtl_BlendFunc( GLenum s, GLenum d ) { gl.blendSrc = s; gl.blendDst = d; }
static void APIENTRY mtl_DepthFunc( GLenum f ) { gl.depthFunc = f; }
static void APIENTRY mtl_DepthMask( GLboolean m ) { gl.depthMask = m; }
static void APIENTRY mtl_AlphaFunc( GLenum f, GLclampf ref ) { gl.alphaFunc = f; gl.alphaRef = ref; }
static void APIENTRY mtl_CullFace( GLenum mode ) { gl.cullFaceMode = mode; }
static void APIENTRY mtl_PolygonMode( GLenum face, GLenum mode ) { gl.polygonMode = mode; }
static void APIENTRY mtl_PolygonOffset( GLfloat factor, GLfloat units ) { gl.offsetFactor = factor; gl.offsetUnits = units; }
static void APIENTRY mtl_ShadeModel( GLenum mode ) {}
static void APIENTRY mtl_LineWidth( GLfloat w ) {}
static void APIENTRY mtl_DrawBuffer( GLenum mode ) {}
static void APIENTRY mtl_CallList( GLuint list ) {}

static void APIENTRY mtl_ColorMask( GLboolean r, GLboolean g, GLboolean b, GLboolean a ) {
	gl.colorMask[0] = r; gl.colorMask[1] = g; gl.colorMask[2] = b; gl.colorMask[3] = a;
}

static void APIENTRY mtl_StencilFunc( GLenum func, GLint ref, GLuint mask ) {
	gl.stencilFunc = func; gl.stencilRef = ref; gl.stencilValueMask = mask;
}
static void APIENTRY mtl_StencilOp( GLenum fail, GLenum zfail, GLenum zpass ) {
	gl.stencilFail = fail; gl.stencilZFail = zfail; gl.stencilZPass = zpass;
}
static void APIENTRY mtl_StencilMask( GLuint mask ) { gl.stencilWriteMask = mask; }

static void APIENTRY mtl_Viewport( GLint x, GLint y, GLsizei w, GLsizei h ) {
	gl.viewport[0] = x; gl.viewport[1] = y; gl.viewport[2] = w; gl.viewport[3] = h;
}
static void APIENTRY mtl_Scissor( GLint x, GLint y, GLsizei w, GLsizei h ) {
	gl.scissor[0] = x; gl.scissor[1] = y; gl.scissor[2] = w; gl.scissor[3] = h;
}
static void APIENTRY mtl_DepthRange( GLclampd n, GLclampd f ) { gl.depthNear = n; gl.depthFar = f; }

static void APIENTRY mtl_ClearColor( GLclampf r, GLclampf g, GLclampf b, GLclampf a ) {
	gl.clearColor[0] = r; gl.clearColor[1] = g; gl.clearColor[2] = b; gl.clearColor[3] = a;
}
static void APIENTRY mtl_ClearDepth( GLclampd d ) { gl.clearDepth = d; }
static void APIENTRY mtl_ClearStencil( GLint s ) { gl.clearStencil = s; }

// The plane is transformed into eye space by the modelview matrix current now.
static void APIENTRY mtl_ClipPlane( GLenum plane, const GLdouble *eq ) {
	float inv[16];
	const float *mv = gl.stack[0][gl.depth[0]];
	int j;

	if ( plane != GL_CLIP_PLANE0 ) {
		return;
	}
	if ( !Mat_Inverse( mv, inv ) ) {
		Mat_Identity( inv );
	}
	// row vector eq times the inverse
	for ( j = 0; j < 4; j++ ) {
		gl.eyePlane[j] = eq[0] * inv[j * 4 + 0] + eq[1] * inv[j * 4 + 1]
		               + eq[2] * inv[j * 4 + 2] + eq[3] * inv[j * 4 + 3];
	}
}

static GLenum APIENTRY mtl_GetError( void ) { return GL_NO_ERROR; }
static void APIENTRY mtl_Finish( void ) { MTLR_Finish(); }

static void APIENTRY mtl_GetIntegerv( GLenum pname, GLint *params ) {
	switch ( pname ) {
	case GL_MAX_TEXTURE_SIZE:        *params = 8192; break;
	case GL_MAX_ACTIVE_TEXTURES_ARB: *params = 2; break;
	default:                         *params = 0; break;
	}
}

static const GLubyte * APIENTRY mtl_GetString( GLenum name ) {
	switch ( name ) {
	case GL_VENDOR:     return (const GLubyte *)"Apple";
	case GL_RENDERER:   return (const GLubyte *)"Metal";
	case GL_VERSION:    return (const GLubyte *)"1.1 Metal";
	case GL_EXTENSIONS: return (const GLubyte *)"GL_ARB_multitexture GL_EXT_texture_env_add GL_EXT_compiled_vertex_array";
	}
	return (const GLubyte *)"";
}

/*
==============================================================================

TEXTURES

==============================================================================
*/

static texParams_t *TexParams( GLuint texnum ) {
	if ( texnum >= MAX_TEXNUM ) {
		return NULL;
	}
	return &gl.texParams[texnum];
}

static int MapFilter( GLenum f )  { return ( f == GL_NEAREST || f == GL_NEAREST_MIPMAP_NEAREST || f == GL_NEAREST_MIPMAP_LINEAR ) ? MTL_FILTER_NEAREST : MTL_FILTER_LINEAR; }
static int MapMip( GLenum f ) {
	switch ( f ) {
	case GL_NEAREST_MIPMAP_NEAREST:
	case GL_LINEAR_MIPMAP_NEAREST: return MTL_MIP_NEAREST;
	case GL_NEAREST_MIPMAP_LINEAR:
	case GL_LINEAR_MIPMAP_LINEAR:  return MTL_MIP_LINEAR;
	}
	return MTL_MIP_NONE;
}
static int MapWrap( GLenum w ) { return ( w == GL_REPEAT ) ? MTL_WRAP_REPEAT : MTL_WRAP_CLAMP; }

static void SendTexParams( GLuint texnum ) {
	texParams_t *p = TexParams( texnum );
	if ( !p || !texnum ) {
		return;
	}
	MTLR_TexParams( texnum, MapFilter( p->minFilter ), MapFilter( p->magFilter ), MapMip( p->minFilter ),
	                MapWrap( p->wrapS ), MapWrap( p->wrapT ) );
}

static void APIENTRY mtl_BindTexture( GLenum target, GLuint texture ) {
	gl.bound[gl.activeTexture] = texture;
}

static void APIENTRY mtl_DeleteTextures( GLsizei n, const GLuint *textures ) {
	int i;
	for ( i = 0; i < n; i++ ) {
		texParams_t *p = TexParams( textures[i] );
		MTLR_DeleteTexture( textures[i] );
		if ( p ) {
			p->minFilter = GL_NEAREST_MIPMAP_LINEAR;
			p->magFilter = GL_LINEAR;
			p->wrapS = p->wrapT = GL_REPEAT;
		}
	}
}

static void APIENTRY mtl_TexImage2D( GLenum target, GLint level, GLint internalformat, GLsizei width,
                                      GLsizei height, GLint border, GLenum format, GLenum type, const GLvoid *pixels ) {
	GLuint tex = gl.bound[gl.activeTexture];
	qboolean alphaOne;

	if ( !tex || !pixels ) {
		return;
	}
	switch ( internalformat ) {
	case 3: case GL_RGB: case GL_RGB4: case GL_RGB5: case GL_RGB8:
	case GL_RGB_S3TC: case GL_RGB4_S3TC: case GL_LUMINANCE: case GL_LUMINANCE8:
		alphaOne = qtrue;
		break;
	default:
		alphaOne = qfalse;
		break;
	}
	MTLR_TexImage( tex, level, width, height, pixels, alphaOne );
	if ( level == 0 ) {
		SendTexParams( tex );
	}
}

static void APIENTRY mtl_TexSubImage2D( GLenum target, GLint level, GLint x, GLint y, GLsizei width,
                                         GLsizei height, GLenum format, GLenum type, const GLvoid *pixels ) {
	GLuint tex = gl.bound[gl.activeTexture];
	if ( tex && pixels ) {
		MTLR_TexSubImage( tex, level, x, y, width, height, pixels );
	}
}

static void APIENTRY mtl_TexParameterf( GLenum target, GLenum pname, GLfloat param ) {
	GLuint tex = gl.bound[gl.activeTexture];
	texParams_t *p = TexParams( tex );

	if ( !p ) {
		return;
	}
	switch ( pname ) {
	case GL_TEXTURE_MIN_FILTER: p->minFilter = (GLenum)param; break;
	case GL_TEXTURE_MAG_FILTER: p->magFilter = (GLenum)param; break;
	case GL_TEXTURE_WRAP_S:     p->wrapS = (GLenum)param; break;
	case GL_TEXTURE_WRAP_T:     p->wrapT = (GLenum)param; break;
	default: return;
	}
	SendTexParams( tex );
}

static void APIENTRY mtl_TexParameterfv( GLenum target, GLenum pname, const GLfloat *params ) {
	if ( pname != GL_TEXTURE_BORDER_COLOR ) {
		mtl_TexParameterf( target, pname, params[0] );
	}
}

static void APIENTRY mtl_TexEnvf( GLenum target, GLenum pname, GLfloat param ) {
	if ( pname == GL_TEXTURE_ENV_MODE ) {
		gl.texEnv[gl.activeTexture] = (GLenum)param;
	}
}

static void APIENTRY mtl_ActiveTextureARB( GLenum texture ) {
	gl.activeTexture = ( texture == GL_TEXTURE1_ARB ) ? 1 : 0;
}

static void APIENTRY mtl_ClientActiveTextureARB( GLenum texture ) {
	gl.clientActiveTexture = ( texture == GL_TEXTURE1_ARB ) ? 1 : 0;
}

/*
==============================================================================

CLIENT ARRAYS

==============================================================================
*/

static int TypeSize( GLenum type ) {
	switch ( type ) {
	case GL_UNSIGNED_BYTE: case GL_BYTE: return 1;
	case GL_SHORT: case GL_UNSIGNED_SHORT: return 2;
	}
	return 4;
}

static void SetArray( clientArray_t *a, GLint size, GLenum type, GLsizei stride, const GLvoid *ptr ) {
	a->size = size;
	a->type = type;
	a->stride = stride ? stride : size * TypeSize( type );
	a->pointer = ptr;
}

static void APIENTRY mtl_VertexPointer( GLint size, GLenum type, GLsizei stride, const GLvoid *ptr ) {
	SetArray( &gl.vertexArray, size, type, stride, ptr );
}
static void APIENTRY mtl_ColorPointer( GLint size, GLenum type, GLsizei stride, const GLvoid *ptr ) {
	SetArray( &gl.colorArray, size, type, stride, ptr );
}
static void APIENTRY mtl_TexCoordPointer( GLint size, GLenum type, GLsizei stride, const GLvoid *ptr ) {
	SetArray( &gl.texCoordArray[gl.clientActiveTexture], size, type, stride, ptr );
}

static clientArray_t *ArrayFor( GLenum array ) {
	switch ( array ) {
	case GL_VERTEX_ARRAY:        return &gl.vertexArray;
	case GL_COLOR_ARRAY:         return &gl.colorArray;
	case GL_TEXTURE_COORD_ARRAY: return &gl.texCoordArray[gl.clientActiveTexture];
	}
	return NULL;
}

static void APIENTRY mtl_EnableClientState( GLenum array ) {
	clientArray_t *a = ArrayFor( array );
	if ( a ) a->enabled = qtrue;
}
static void APIENTRY mtl_DisableClientState( GLenum array ) {
	clientArray_t *a = ArrayFor( array );
	if ( a ) a->enabled = qfalse;
}

static void APIENTRY mtl_LockArraysEXT( GLint first, GLint count ) {
	gl.lockFirst = first;
	gl.lockCount = count;
}
static void APIENTRY mtl_UnlockArraysEXT( void ) {
	gl.lockCount = 0;
}

// Fetches array element i into a packed vertex, using the current values
// for arrays that are not enabled.
static void FetchVertex( int i, mtl_vertex_t *v ) {
	const clientArray_t *a;
	int t;

	a = &gl.vertexArray;
	if ( a->enabled && a->pointer ) {
		const float *p = (const float *)( a->pointer + i * a->stride );
		v->xyz[0] = p[0];
		v->xyz[1] = p[1];
		v->xyz[2] = ( a->size > 2 ) ? p[2] : 0.0f;
	} else {
		v->xyz[0] = v->xyz[1] = v->xyz[2] = 0.0f;
	}

	for ( t = 0; t < 2; t++ ) {
		float *st = t ? v->st1 : v->st0;
		a = &gl.texCoordArray[t];
		if ( a->enabled && a->pointer ) {
			const float *p = (const float *)( a->pointer + i * a->stride );
			st[0] = p[0];
			st[1] = p[1];
		} else {
			st[0] = gl.texCoord[t][0];
			st[1] = gl.texCoord[t][1];
		}
	}

	a = &gl.colorArray;
	if ( a->enabled && a->pointer ) {
		const byte *p = a->pointer + i * a->stride;
		v->color[0] = p[0];
		v->color[1] = p[1];
		v->color[2] = p[2];
		v->color[3] = ( a->size > 3 ) ? p[3] : 255;
	} else {
		*(int *)v->color = *(int *)gl.color;
	}
}

/*
==============================================================================

DRAWING

==============================================================================
*/

static int MapCompare( GLenum f ) {
	switch ( f ) {
	case GL_NEVER:    return MTL_CMP_NEVER;
	case GL_LESS:     return MTL_CMP_LESS;
	case GL_EQUAL:    return MTL_CMP_EQUAL;
	case GL_LEQUAL:   return MTL_CMP_LEQUAL;
	case GL_GREATER:  return MTL_CMP_GREATER;
	case GL_NOTEQUAL: return MTL_CMP_NOTEQUAL;
	case GL_GEQUAL:   return MTL_CMP_GEQUAL;
	}
	return MTL_CMP_ALWAYS;
}

static int MapBlend( GLenum f ) {
	switch ( f ) {
	case GL_ZERO:                return MTL_BLEND_ZERO;
	case GL_ONE:                 return MTL_BLEND_ONE;
	case GL_SRC_COLOR:           return MTL_BLEND_SRC_COLOR;
	case GL_ONE_MINUS_SRC_COLOR: return MTL_BLEND_ONE_MINUS_SRC_COLOR;
	case GL_SRC_ALPHA:           return MTL_BLEND_SRC_ALPHA;
	case GL_ONE_MINUS_SRC_ALPHA: return MTL_BLEND_ONE_MINUS_SRC_ALPHA;
	case GL_DST_COLOR:           return MTL_BLEND_DST_COLOR;
	case GL_ONE_MINUS_DST_COLOR: return MTL_BLEND_ONE_MINUS_DST_COLOR;
	case GL_DST_ALPHA:           return MTL_BLEND_DST_ALPHA;
	case GL_ONE_MINUS_DST_ALPHA: return MTL_BLEND_ONE_MINUS_DST_ALPHA;
	case GL_SRC_ALPHA_SATURATE:  return MTL_BLEND_SRC_ALPHA_SATURATE;
	}
	return MTL_BLEND_ONE;
}

static int MapStencilOp( GLenum op ) {
	switch ( op ) {
	case GL_ZERO:    return MTL_STENCIL_ZERO;
	case GL_REPLACE: return MTL_STENCIL_REPLACE;
	case GL_INCR:    return MTL_STENCIL_INCR;
	case GL_DECR:    return MTL_STENCIL_DECR;
	case GL_INVERT:  return MTL_STENCIL_INVERT;
	}
	return MTL_STENCIL_KEEP;
}

static int MapEnv( GLenum env ) {
	switch ( env ) {
	case GL_REPLACE: return MTL_ENV_REPLACE;
	case GL_DECAL:   return MTL_ENV_DECAL;
	case GL_ADD:     return MTL_ENV_ADD;
	}
	return MTL_ENV_MODULATE;
}

static void BuildState( mtl_state_t *s ) {
	const float *mv = gl.stack[0][gl.depth[0]];
	const float *proj = gl.stack[1][gl.depth[1]];
	int j;

	memset( s, 0, sizeof( *s ) );

	Mat_Multiply( proj, mv, s->vs.mvp );
	s->vs.clipEnable = gl.clipPlane0;
	if ( gl.clipPlane0 ) {
		// object space plane = eye plane * modelview
		for ( j = 0; j < 4; j++ ) {
			s->vs.clipPlane[j] = gl.eyePlane[0] * mv[j * 4 + 0] + gl.eyePlane[1] * mv[j * 4 + 1]
			                   + gl.eyePlane[2] * mv[j * 4 + 2] + gl.eyePlane[3] * mv[j * 4 + 3];
		}
	}

	s->fs.tex0Enable = gl.texture2D[0];
	s->fs.tex1Enable = gl.texture2D[1];
	s->fs.env0 = MapEnv( gl.texEnv[0] );
	s->fs.env1 = MapEnv( gl.texEnv[1] );
	s->fs.alphaFunc = gl.alphaTest ? MapCompare( gl.alphaFunc ) : MTL_CMP_ALWAYS;
	s->fs.alphaRef = gl.alphaRef;
	s->tex0 = gl.texture2D[0] ? gl.bound[0] : 0;
	s->tex1 = gl.texture2D[1] ? gl.bound[1] : 0;

	s->blendEnable = gl.blend;
	s->blendSrc = MapBlend( gl.blendSrc );
	s->blendDst = MapBlend( gl.blendDst );
	s->colorMask = ( gl.colorMask[0] ? 1 : 0 ) | ( gl.colorMask[1] ? 2 : 0 )
	             | ( gl.colorMask[2] ? 4 : 0 ) | ( gl.colorMask[3] ? 8 : 0 );

	// OpenGL does not write depth while the depth test is disabled
	s->depthTest = gl.depthTest;
	s->depthWrite = gl.depthTest && gl.depthMask;
	s->depthFunc = MapCompare( gl.depthFunc );

	s->stencilTest = gl.stencilTest;
	s->stencilFunc = MapCompare( gl.stencilFunc );
	s->stencilFail = MapStencilOp( gl.stencilFail );
	s->stencilZFail = MapStencilOp( gl.stencilZFail );
	s->stencilZPass = MapStencilOp( gl.stencilZPass );
	s->stencilRef = (uint8_t)gl.stencilRef;
	s->stencilReadMask = (uint8_t)gl.stencilValueMask;
	s->stencilWriteMask = (uint8_t)gl.stencilWriteMask;
	if ( relightMark != MTL_RELIGHT_NONE && !gl.stencilTest ) {
		// relight mask (r_rtRelight): set or clear one stencil bit where the
		// surface is drawn, for the rt_relight pass
		s->stencilTest = 1;
		s->stencilFunc = MTL_CMP_ALWAYS;
		s->stencilFail = MTL_STENCIL_KEEP;
		s->stencilZFail = MTL_STENCIL_KEEP;
		s->stencilZPass = MTL_STENCIL_REPLACE;
		s->stencilRef = relightMark == MTL_RELIGHT_SET ? MTL_RELIGHT_STENCIL_BIT : 0;
		s->stencilReadMask = 0;
		s->stencilWriteMask = MTL_RELIGHT_STENCIL_BIT;
	}

	if ( gl.cullFace ) {
		s->cullMode = ( gl.cullFaceMode == GL_BACK ) ? MTL_CULL_BACK : MTL_CULL_FRONT;
	} else {
		s->cullMode = MTL_CULL_NONE;
	}
	s->wireframe = ( gl.polygonMode == GL_LINE );
	s->polygonOffset = gl.polygonOffsetFill;
	s->offsetFactor = gl.offsetFactor;
	s->offsetUnits = gl.offsetUnits;

	memcpy( s->viewport, gl.viewport, sizeof( s->viewport ) );
	s->depthNear = gl.depthNear;
	s->depthFar = gl.depthFar;
	s->scissorEnable = gl.scissorTest;
	memcpy( s->scissor, gl.scissor, sizeof( s->scissor ) );
}

static void APIENTRY mtl_DrawElements( GLenum mode, GLsizei count, GLenum type, const GLvoid *indices ) {
	mtl_state_t state;
	mtl_vertex_t *verts;
	uint32_t *idx;
	int numVerts, i;

	if ( count <= 0 || mode != GL_TRIANGLES ) {
		return;
	}

	if ( gl.lockCount > 0 ) {
		numVerts = gl.lockFirst + gl.lockCount;
	} else {
		numVerts = 0;
		for ( i = 0; i < count; i++ ) {
			int v = ( type == GL_UNSIGNED_SHORT ) ? ( (const unsigned short *)indices )[i] : ( (const unsigned int *)indices )[i];
			if ( v >= numVerts ) numVerts = v + 1;
		}
	}

	verts = MTLR_AllocVertices( numVerts );
	idx = MTLR_AllocIndices( count );
	if ( !verts || !idx ) {
		return;
	}
	for ( i = 0; i < numVerts; i++ ) {
		FetchVertex( i, &verts[i] );
	}
	if ( type == GL_UNSIGNED_SHORT ) {
		for ( i = 0; i < count; i++ ) idx[i] = ( (const unsigned short *)indices )[i];
	} else {
		memcpy( idx, indices, count * sizeof( uint32_t ) );
	}

	BuildState( &state );
	MTLR_Draw( &state, MTL_PRIM_TRIANGLES, numVerts, count );
}

/*
==============================================================================

IMMEDIATE MODE

==============================================================================
*/

static void APIENTRY mtl_Begin( GLenum mode ) {
	gl.primitive = mode;
	gl.inBegin = qtrue;
	gl.immCount = 0;
}

static mtl_vertex_t *ImmNext( void ) {
	if ( gl.immCount == gl.immMax ) {
		gl.immMax = gl.immMax ? gl.immMax * 2 : 1024;
		gl.imm = realloc( gl.imm, gl.immMax * sizeof( mtl_vertex_t ) );
	}
	return &gl.imm[gl.immCount++];
}

static void ImmVertex( float x, float y, float z ) {
	mtl_vertex_t *v;
	if ( !gl.inBegin ) {
		return;
	}
	v = ImmNext();
	v->xyz[0] = x; v->xyz[1] = y; v->xyz[2] = z;
	v->st0[0] = gl.texCoord[0][0]; v->st0[1] = gl.texCoord[0][1];
	v->st1[0] = gl.texCoord[1][0]; v->st1[1] = gl.texCoord[1][1];
	*(int *)v->color = *(int *)gl.color;
}

static void APIENTRY mtl_Vertex2f( GLfloat x, GLfloat y ) { ImmVertex( x, y, 0 ); }
static void APIENTRY mtl_Vertex3f( GLfloat x, GLfloat y, GLfloat z ) { ImmVertex( x, y, z ); }
static void APIENTRY mtl_Vertex3fv( const GLfloat *v ) { ImmVertex( v[0], v[1], v[2] ); }

static void APIENTRY mtl_ArrayElement( GLint i ) {
	if ( gl.inBegin ) {
		FetchVertex( i, ImmNext() );
	}
}

static void APIENTRY mtl_TexCoord2f( GLfloat s, GLfloat t ) { gl.texCoord[0][0] = s; gl.texCoord[0][1] = t; }
static void APIENTRY mtl_TexCoord2fv( const GLfloat *v ) { gl.texCoord[0][0] = v[0]; gl.texCoord[0][1] = v[1]; }
static void APIENTRY mtl_MultiTexCoord2fARB( GLenum target, GLfloat s, GLfloat t ) {
	int unit = ( target == GL_TEXTURE1_ARB ) ? 1 : 0;
	gl.texCoord[unit][0] = s;
	gl.texCoord[unit][1] = t;
}

static byte ClampByte( float f ) {
	if ( f <= 0 ) return 0;
	if ( f >= 1 ) return 255;
	return (byte)( f * 255.0f + 0.5f );
}

static void APIENTRY mtl_Color4f( GLfloat r, GLfloat g, GLfloat b, GLfloat a ) {
	gl.color[0] = ClampByte( r ); gl.color[1] = ClampByte( g );
	gl.color[2] = ClampByte( b ); gl.color[3] = ClampByte( a );
}
static void APIENTRY mtl_Color3f( GLfloat r, GLfloat g, GLfloat b ) { mtl_Color4f( r, g, b, 1 ); }
static void APIENTRY mtl_Color4ubv( const GLubyte *v ) { memcpy( gl.color, v, 4 ); }

static void APIENTRY mtl_End( void ) {
	mtl_state_t state;
	mtl_vertex_t *verts;
	uint32_t *idx;
	int n = gl.immCount, numIndexes = 0, prim = MTL_PRIM_TRIANGLES, i;

	gl.inBegin = qfalse;
	if ( n == 0 ) {
		return;
	}

	switch ( gl.primitive ) {
	case GL_TRIANGLES:      numIndexes = ( n / 3 ) * 3; break;
	case GL_TRIANGLE_STRIP:
	case GL_TRIANGLE_FAN:
	case GL_POLYGON:        numIndexes = ( n >= 3 ) ? ( n - 2 ) * 3 : 0; break;
	case GL_QUADS:          numIndexes = ( n / 4 ) * 6; break;
	case GL_LINES:          numIndexes = ( n / 2 ) * 2; prim = MTL_PRIM_LINES; break;
	case GL_LINE_STRIP:     numIndexes = ( n >= 2 ) ? ( n - 1 ) * 2 : 0; prim = MTL_PRIM_LINES; break;
	case GL_LINE_LOOP:      numIndexes = ( n >= 2 ) ? n * 2 : 0; prim = MTL_PRIM_LINES; break;
	case GL_POINTS:         numIndexes = n; prim = MTL_PRIM_POINTS; break;
	default: return;
	}
	if ( numIndexes == 0 ) {
		return;
	}

	verts = MTLR_AllocVertices( n );
	idx = MTLR_AllocIndices( numIndexes );
	if ( !verts || !idx ) {
		return;
	}
	memcpy( verts, gl.imm, n * sizeof( mtl_vertex_t ) );

	switch ( gl.primitive ) {
	case GL_TRIANGLE_STRIP:
		for ( i = 0; i < n - 2; i++ ) {
			// keep the winding of every other triangle consistent
			idx[i * 3 + 0] = ( i & 1 ) ? i + 1 : i;
			idx[i * 3 + 1] = ( i & 1 ) ? i : i + 1;
			idx[i * 3 + 2] = i + 2;
		}
		break;
	case GL_TRIANGLE_FAN:
	case GL_POLYGON:
		for ( i = 0; i < n - 2; i++ ) {
			idx[i * 3 + 0] = 0;
			idx[i * 3 + 1] = i + 1;
			idx[i * 3 + 2] = i + 2;
		}
		break;
	case GL_QUADS:
		for ( i = 0; i < n / 4; i++ ) {
			idx[i * 6 + 0] = i * 4 + 0; idx[i * 6 + 1] = i * 4 + 1; idx[i * 6 + 2] = i * 4 + 2;
			idx[i * 6 + 3] = i * 4 + 0; idx[i * 6 + 4] = i * 4 + 2; idx[i * 6 + 5] = i * 4 + 3;
		}
		break;
	case GL_LINE_STRIP:
		for ( i = 0; i < n - 1; i++ ) { idx[i * 2] = i; idx[i * 2 + 1] = i + 1; }
		break;
	case GL_LINE_LOOP:
		for ( i = 0; i < n; i++ ) { idx[i * 2] = i; idx[i * 2 + 1] = ( i + 1 ) % n; }
		break;
	default:
		for ( i = 0; i < numIndexes; i++ ) idx[i] = i;
		break;
	}

	BuildState( &state );
	MTLR_Draw( &state, prim, n, numIndexes );
}

/*
==============================================================================

FRAMEBUFFER

==============================================================================
*/

static void APIENTRY mtl_Clear( GLbitfield mask ) {
	mtl_state_t state;
	int flags = 0;

	// clears honour the write masks, like OpenGL
	if ( ( mask & GL_COLOR_BUFFER_BIT ) && ( gl.colorMask[0] || gl.colorMask[1] || gl.colorMask[2] || gl.colorMask[3] ) ) {
		flags |= MTL_CLEAR_COLOR;
	}
	if ( ( mask & GL_DEPTH_BUFFER_BIT ) && gl.depthMask ) {
		flags |= MTL_CLEAR_DEPTH;
	}
	if ( ( mask & GL_STENCIL_BUFFER_BIT ) && gl.stencilWriteMask ) {
		flags |= MTL_CLEAR_STENCIL;
	}
	if ( !flags ) {
		return;
	}
	BuildState( &state );
	MTLR_Clear( &state, flags, gl.clearColor, gl.clearDepth, gl.clearStencil );
}

static void APIENTRY mtl_ReadPixels( GLint x, GLint y, GLsizei width, GLsizei height, GLenum format,
                                      GLenum type, GLvoid *pixels ) {
	int fmt;
	switch ( format ) {
	case GL_RGB:             fmt = MTL_READ_RGB; break;
	case GL_RGBA:            fmt = MTL_READ_RGBA; break;
	case GL_DEPTH_COMPONENT: fmt = MTL_READ_DEPTH; break;
	case GL_STENCIL_INDEX:   fmt = MTL_READ_STENCIL; break;
	default: return;
	}
	MTLR_ReadPixels( x, y, width, height, fmt, pixels );
}

/*
==============================================================================

BINDING

==============================================================================
*/

// the qgl* function pointers the renderer uses (declared in renderer/qgl.h)
void ( APIENTRY * qglAlphaFunc )(GLenum func, GLclampf ref);
void ( APIENTRY * qglArrayElement )(GLint i);
void ( APIENTRY * qglBegin )(GLenum mode);
void ( APIENTRY * qglBindTexture )(GLenum target, GLuint texture);
void ( APIENTRY * qglBlendFunc )(GLenum sfactor, GLenum dfactor);
void ( APIENTRY * qglCallList )(GLuint list);
void ( APIENTRY * qglClear )(GLbitfield mask);
void ( APIENTRY * qglClearColor )(GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha);
void ( APIENTRY * qglClearDepth )(GLclampd depth);
void ( APIENTRY * qglClearStencil )(GLint s);
void ( APIENTRY * qglClipPlane )(GLenum plane, const GLdouble *equation);
void ( APIENTRY * qglColor3f )(GLfloat red, GLfloat green, GLfloat blue);
void ( APIENTRY * qglColor4f )(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha);
void ( APIENTRY * qglColor4ubv )(const GLubyte *v);
void ( APIENTRY * qglColorMask )(GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha);
void ( APIENTRY * qglColorPointer )(GLint size, GLenum type, GLsizei stride, const GLvoid *pointer);
void ( APIENTRY * qglCullFace )(GLenum mode);
void ( APIENTRY * qglDeleteTextures )(GLsizei n, const GLuint *textures);
void ( APIENTRY * qglDepthFunc )(GLenum func);
void ( APIENTRY * qglDepthMask )(GLboolean flag);
void ( APIENTRY * qglDepthRange )(GLclampd zNear, GLclampd zFar);
void ( APIENTRY * qglDisable )(GLenum cap);
void ( APIENTRY * qglDisableClientState )(GLenum array);
void ( APIENTRY * qglDrawBuffer )(GLenum mode);
void ( APIENTRY * qglDrawElements )(GLenum mode, GLsizei count, GLenum type, const GLvoid *indices);
void ( APIENTRY * qglEnable )(GLenum cap);
void ( APIENTRY * qglEnableClientState )(GLenum array);
void ( APIENTRY * qglEnd )(void);
void ( APIENTRY * qglFinish )(void);
GLenum ( APIENTRY * qglGetError )(void);
void ( APIENTRY * qglGetIntegerv )(GLenum pname, GLint *params);
const GLubyte * ( APIENTRY * qglGetString )(GLenum name);
void ( APIENTRY * qglLineWidth )(GLfloat width);
void ( APIENTRY * qglLoadIdentity )(void);
void ( APIENTRY * qglLoadMatrixf )(const GLfloat *m);
void ( APIENTRY * qglMatrixMode )(GLenum mode);
void ( APIENTRY * qglOrtho )(GLdouble left, GLdouble right, GLdouble bottom, GLdouble top, GLdouble zNear, GLdouble zFar);
void ( APIENTRY * qglPolygonMode )(GLenum face, GLenum mode);
void ( APIENTRY * qglPolygonOffset )(GLfloat factor, GLfloat units);
void ( APIENTRY * qglPopMatrix )(void);
void ( APIENTRY * qglPushMatrix )(void);
void ( APIENTRY * qglReadPixels )(GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, GLvoid *pixels);
void ( APIENTRY * qglScissor )(GLint x, GLint y, GLsizei width, GLsizei height);
void ( APIENTRY * qglShadeModel )(GLenum mode);
void ( APIENTRY * qglStencilFunc )(GLenum func, GLint ref, GLuint mask);
void ( APIENTRY * qglStencilMask )(GLuint mask);
void ( APIENTRY * qglStencilOp )(GLenum fail, GLenum zfail, GLenum zpass);
void ( APIENTRY * qglTexCoord2f )(GLfloat s, GLfloat t);
void ( APIENTRY * qglTexCoord2fv )(const GLfloat *v);
void ( APIENTRY * qglTexCoordPointer )(GLint size, GLenum type, GLsizei stride, const GLvoid *pointer);
void ( APIENTRY * qglTexEnvf )(GLenum target, GLenum pname, GLfloat param);
void ( APIENTRY * qglTexImage2D )(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const GLvoid *pixels);
void ( APIENTRY * qglTexParameterf )(GLenum target, GLenum pname, GLfloat param);
void ( APIENTRY * qglTexParameterfv )(GLenum target, GLenum pname, const GLfloat *params);
void ( APIENTRY * qglTexSubImage2D )(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const GLvoid *pixels);
void ( APIENTRY * qglTranslatef )(GLfloat x, GLfloat y, GLfloat z);
void ( APIENTRY * qglVertex2f )(GLfloat x, GLfloat y);
void ( APIENTRY * qglVertex3f )(GLfloat x, GLfloat y, GLfloat z);
void ( APIENTRY * qglVertex3fv )(const GLfloat *v);
void ( APIENTRY * qglVertexPointer )(GLint size, GLenum type, GLsizei stride, const GLvoid *pointer);
void ( APIENTRY * qglViewport )(GLint x, GLint y, GLsizei width, GLsizei height);

void ( APIENTRY * qglMultiTexCoord2fARB )( GLenum texture, GLfloat s, GLfloat t );
void ( APIENTRY * qglActiveTextureARB )( GLenum texture );
void ( APIENTRY * qglClientActiveTextureARB )( GLenum texture );
void ( APIENTRY * qglLockArraysEXT )( GLint, GLint );
void ( APIENTRY * qglUnlockArraysEXT )( void );

static void ResetState( void ) {
	int i;

	free( gl.imm );
	free( gl.texParams );
	memset( &gl, 0, sizeof( gl ) );

	Mat_Identity( gl.stack[0][0] );
	Mat_Identity( gl.stack[1][0] );
	gl.blendSrc = GL_ONE;
	gl.blendDst = GL_ZERO;
	gl.depthFunc = GL_LESS;
	gl.alphaFunc = GL_ALWAYS;
	gl.depthMask = GL_TRUE;
	gl.colorMask[0] = gl.colorMask[1] = gl.colorMask[2] = gl.colorMask[3] = GL_TRUE;
	gl.cullFaceMode = GL_BACK;
	gl.polygonMode = GL_FILL;
	gl.stencilFunc = GL_ALWAYS;
	gl.stencilValueMask = gl.stencilWriteMask = 0xff;
	gl.stencilFail = gl.stencilZFail = gl.stencilZPass = GL_KEEP;
	gl.depthNear = 0;
	gl.depthFar = 1;
	gl.clearDepth = 1;
	gl.texEnv[0] = gl.texEnv[1] = GL_MODULATE;
	gl.color[0] = gl.color[1] = gl.color[2] = gl.color[3] = 255;

	gl.texParams = calloc( MAX_TEXNUM, sizeof( texParams_t ) );
	for ( i = 0; i < MAX_TEXNUM; i++ ) {
		gl.texParams[i].minFilter = GL_NEAREST_MIPMAP_LINEAR;
		gl.texParams[i].magFilter = GL_LINEAR;
		gl.texParams[i].wrapS = gl.texParams[i].wrapT = GL_REPEAT;
	}
}

/*
** r_renderer opengl: the same qgl* pointers bound to Apple's OpenGL framework
** (legacy profile). The token values in mac_gl.h are OpenGL's own, so the
** renderer's calls go through unchanged.
*/
#define OPENGL_FRAMEWORK "/System/Library/Frameworks/OpenGL.framework/OpenGL"

static void *glLibrary;

#define GL_BIND( name ) \
	if ( !( *(void **)&qgl##name = dlsym( glLibrary, "gl" #name ) ) ) { \
		ri.Printf( PRINT_ALL, "QGL_Init: OpenGL has no gl" #name "\n" ); \
		return qfalse; \
	}

// extensions: the renderer checks these pointers and does without them
#define GL_BIND_OPTIONAL( name ) \
	*(void **)&qgl##name = dlsym( glLibrary, "gl" #name );

static qboolean QGL_InitOpenGL( void ) {
	glLibrary = dlopen( OPENGL_FRAMEWORK, RTLD_LAZY | RTLD_LOCAL );
	if ( !glLibrary ) {
		ri.Printf( PRINT_ALL, "QGL_Init: could not load " OPENGL_FRAMEWORK "\n" );
		return qfalse;
	}
	GL_BIND( AlphaFunc );
	GL_BIND( ArrayElement );
	GL_BIND( Begin );
	GL_BIND( BindTexture );
	GL_BIND( BlendFunc );
	GL_BIND( CallList );
	GL_BIND( Clear );
	GL_BIND( ClearColor );
	GL_BIND( ClearDepth );
	GL_BIND( ClearStencil );
	GL_BIND( ClipPlane );
	GL_BIND( Color3f );
	GL_BIND( Color4f );
	GL_BIND( Color4ubv );
	GL_BIND( ColorMask );
	GL_BIND( ColorPointer );
	GL_BIND( CullFace );
	GL_BIND( DeleteTextures );
	GL_BIND( DepthFunc );
	GL_BIND( DepthMask );
	GL_BIND( DepthRange );
	GL_BIND( Disable );
	GL_BIND( DisableClientState );
	GL_BIND( DrawBuffer );
	GL_BIND( DrawElements );
	GL_BIND( Enable );
	GL_BIND( EnableClientState );
	GL_BIND( End );
	GL_BIND( Finish );
	GL_BIND( GetError );
	GL_BIND( GetIntegerv );
	GL_BIND( GetString );
	GL_BIND( LineWidth );
	GL_BIND( LoadIdentity );
	GL_BIND( LoadMatrixf );
	GL_BIND( MatrixMode );
	GL_BIND( Ortho );
	GL_BIND( PolygonMode );
	GL_BIND( PolygonOffset );
	GL_BIND( PopMatrix );
	GL_BIND( PushMatrix );
	GL_BIND( ReadPixels );
	GL_BIND( Scissor );
	GL_BIND( ShadeModel );
	GL_BIND( StencilFunc );
	GL_BIND( StencilMask );
	GL_BIND( StencilOp );
	GL_BIND( TexCoord2f );
	GL_BIND( TexCoord2fv );
	GL_BIND( TexCoordPointer );
	GL_BIND( TexEnvf );
	GL_BIND( TexImage2D );
	GL_BIND( TexParameterf );
	GL_BIND( TexParameterfv );
	GL_BIND( TexSubImage2D );
	GL_BIND( Translatef );
	GL_BIND( Vertex2f );
	GL_BIND( Vertex3f );
	GL_BIND( Vertex3fv );
	GL_BIND( VertexPointer );
	GL_BIND( Viewport );
	GL_BIND_OPTIONAL( MultiTexCoord2fARB );
	GL_BIND_OPTIONAL( ActiveTextureARB );
	GL_BIND_OPTIONAL( ClientActiveTextureARB );
	GL_BIND_OPTIONAL( LockArraysEXT );
	GL_BIND_OPTIONAL( UnlockArraysEXT );
	return qtrue;
}

/*
** dllname "opengl": bind to the OpenGL framework; anything else (NULL): bind
** to the Metal translation in this file.
*/
qboolean QGL_Init( const char *dllname ) {
	if ( dllname && !Q_stricmp( dllname, "opengl" ) ) {
		return QGL_InitOpenGL();
	}

	ResetState();

	qglAlphaFunc = mtl_AlphaFunc;
	qglArrayElement = mtl_ArrayElement;
	qglBegin = mtl_Begin;
	qglBindTexture = mtl_BindTexture;
	qglBlendFunc = mtl_BlendFunc;
	qglCallList = mtl_CallList;
	qglClear = mtl_Clear;
	qglClearColor = mtl_ClearColor;
	qglClearDepth = mtl_ClearDepth;
	qglClearStencil = mtl_ClearStencil;
	qglClipPlane = mtl_ClipPlane;
	qglColor3f = mtl_Color3f;
	qglColor4f = mtl_Color4f;
	qglColor4ubv = mtl_Color4ubv;
	qglColorMask = mtl_ColorMask;
	qglColorPointer = mtl_ColorPointer;
	qglCullFace = mtl_CullFace;
	qglDeleteTextures = mtl_DeleteTextures;
	qglDepthFunc = mtl_DepthFunc;
	qglDepthMask = mtl_DepthMask;
	qglDepthRange = mtl_DepthRange;
	qglDisable = mtl_Disable;
	qglDisableClientState = mtl_DisableClientState;
	qglDrawBuffer = mtl_DrawBuffer;
	qglDrawElements = mtl_DrawElements;
	qglEnable = mtl_Enable;
	qglEnableClientState = mtl_EnableClientState;
	qglEnd = mtl_End;
	qglFinish = mtl_Finish;
	qglGetError = mtl_GetError;
	qglGetIntegerv = mtl_GetIntegerv;
	qglGetString = mtl_GetString;
	qglLineWidth = mtl_LineWidth;
	qglLoadIdentity = mtl_LoadIdentity;
	qglLoadMatrixf = mtl_LoadMatrixf;
	qglMatrixMode = mtl_MatrixMode;
	qglOrtho = mtl_Ortho;
	qglPolygonMode = mtl_PolygonMode;
	qglPolygonOffset = mtl_PolygonOffset;
	qglPopMatrix = mtl_PopMatrix;
	qglPushMatrix = mtl_PushMatrix;
	qglReadPixels = mtl_ReadPixels;
	qglScissor = mtl_Scissor;
	qglShadeModel = mtl_ShadeModel;
	qglStencilFunc = mtl_StencilFunc;
	qglStencilMask = mtl_StencilMask;
	qglStencilOp = mtl_StencilOp;
	qglTexCoord2f = mtl_TexCoord2f;
	qglTexCoord2fv = mtl_TexCoord2fv;
	qglTexCoordPointer = mtl_TexCoordPointer;
	qglTexEnvf = mtl_TexEnvf;
	qglTexImage2D = mtl_TexImage2D;
	qglTexParameterf = mtl_TexParameterf;
	qglTexParameterfv = mtl_TexParameterfv;
	qglTexSubImage2D = mtl_TexSubImage2D;
	qglTranslatef = mtl_Translatef;
	qglVertex2f = mtl_Vertex2f;
	qglVertex3f = mtl_Vertex3f;
	qglVertex3fv = mtl_Vertex3fv;
	qglVertexPointer = mtl_VertexPointer;
	qglViewport = mtl_Viewport;

	qglMultiTexCoord2fARB = mtl_MultiTexCoord2fARB;
	qglActiveTextureARB = mtl_ActiveTextureARB;
	qglClientActiveTextureARB = mtl_ClientActiveTextureARB;
	qglLockArraysEXT = mtl_LockArraysEXT;
	qglUnlockArraysEXT = mtl_UnlockArraysEXT;

	return qtrue;
}

void QGL_Shutdown( void ) {
	if ( glLibrary ) {
		dlclose( glLibrary );
		glLibrary = NULL;
	}
	free( gl.imm );
	free( gl.texParams );
	memset( &gl, 0, sizeof( gl ) );

	qglActiveTextureARB = NULL;
	qglClientActiveTextureARB = NULL;
	qglMultiTexCoord2fARB = NULL;
	qglLockArraysEXT = NULL;
	qglUnlockArraysEXT = NULL;
	qglBindTexture = NULL;
}

void QGL_EnableLogging( qboolean enable ) {
}
