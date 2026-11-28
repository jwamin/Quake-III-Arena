/*
===========================================================================
Metal port: OpenGL types and enum values used by the renderer.

The renderer still speaks a small fixed-function OpenGL 1.x subset through
its qgl* function pointers. In the Metal build those pointers are bound to
mtl_qgl.c, which translates the calls into Metal work (MetalRenderer.swift).
No OpenGL library or framework is linked; this header only supplies the
types and the token values the renderer source uses.
===========================================================================
*/
#ifndef __MAC_GL_H__
#define __MAC_GL_H__

typedef unsigned int    GLenum;
typedef unsigned char   GLboolean;
typedef unsigned int    GLbitfield;
typedef void            GLvoid;
typedef signed char     GLbyte;
typedef short           GLshort;
typedef int             GLint;
typedef unsigned char   GLubyte;
typedef unsigned short  GLushort;
typedef unsigned int    GLuint;
typedef int             GLsizei;
typedef float           GLfloat;
typedef float           GLclampf;
typedef double          GLdouble;
typedef double          GLclampd;

#ifndef APIENTRY
#define APIENTRY
#endif

#define GL_FALSE                        0
#define GL_TRUE                         1
#define GL_ZERO                         0
#define GL_ONE                          1

/* primitives */
#define GL_POINTS                       0x0000
#define GL_LINES                        0x0001
#define GL_LINE_LOOP                    0x0002
#define GL_LINE_STRIP                   0x0003
#define GL_TRIANGLES                    0x0004
#define GL_TRIANGLE_STRIP               0x0005
#define GL_TRIANGLE_FAN                 0x0006
#define GL_QUADS                        0x0007
#define GL_QUAD_STRIP                   0x0008
#define GL_POLYGON                      0x0009

/* compare functions */
#define GL_NEVER                        0x0200
#define GL_LESS                         0x0201
#define GL_EQUAL                        0x0202
#define GL_LEQUAL                       0x0203
#define GL_GREATER                      0x0204
#define GL_NOTEQUAL                     0x0205
#define GL_GEQUAL                       0x0206
#define GL_ALWAYS                       0x0207

/* blending */
#define GL_SRC_COLOR                    0x0300
#define GL_ONE_MINUS_SRC_COLOR          0x0301
#define GL_SRC_ALPHA                    0x0302
#define GL_ONE_MINUS_SRC_ALPHA          0x0303
#define GL_DST_ALPHA                    0x0304
#define GL_ONE_MINUS_DST_ALPHA          0x0305
#define GL_DST_COLOR                    0x0306
#define GL_ONE_MINUS_DST_COLOR          0x0307
#define GL_SRC_ALPHA_SATURATE           0x0308

/* buffers / faces */
#define GL_FRONT_LEFT                   0x0400
#define GL_FRONT_RIGHT                  0x0401
#define GL_BACK_LEFT                    0x0402
#define GL_BACK_RIGHT                   0x0403
#define GL_FRONT                        0x0404
#define GL_BACK                         0x0405
#define GL_FRONT_AND_BACK               0x0408

/* errors */
#define GL_NO_ERROR                     0
#define GL_INVALID_ENUM                 0x0500
#define GL_INVALID_VALUE                0x0501
#define GL_INVALID_OPERATION            0x0502
#define GL_STACK_OVERFLOW               0x0503
#define GL_STACK_UNDERFLOW              0x0504
#define GL_OUT_OF_MEMORY                0x0505

/* capabilities */
#define GL_CULL_FACE                    0x0B44
#define GL_DEPTH_TEST                   0x0B71
#define GL_STENCIL_TEST                 0x0B90
#define GL_ALPHA_TEST                   0x0BC0
#define GL_BLEND                        0x0BE2
#define GL_SCISSOR_TEST                 0x0C11
#define GL_TEXTURE_2D                   0x0DE1
#define GL_POLYGON_OFFSET_FILL          0x8037
#define GL_CLIP_PLANE0                  0x3000

/* polygon mode / shading */
#define GL_POINT                        0x1B00
#define GL_LINE                         0x1B01
#define GL_FILL                         0x1B02
#define GL_FLAT                         0x1D00
#define GL_SMOOTH                       0x1D01

/* matrices */
#define GL_MODELVIEW                    0x1700
#define GL_PROJECTION                   0x1701
#define GL_TEXTURE                      0x1702

/* stencil ops */
#define GL_KEEP                         0x1E00
#define GL_REPLACE                      0x1E01
#define GL_INCR                         0x1E02
#define GL_DECR                         0x1E03
#define GL_INVERT                       0x150A

/* strings / queries */
#define GL_VENDOR                       0x1F00
#define GL_MAX_TEXTURE_UNITS_ARB        0x84E2
#define GL_RENDERER                     0x1F01
#define GL_VERSION                      0x1F02
#define GL_EXTENSIONS                   0x1F03
#define GL_MAX_TEXTURE_SIZE             0x0D33

/* data types */
#define GL_BYTE                         0x1400
#define GL_UNSIGNED_BYTE                0x1401
#define GL_SHORT                        0x1402
#define GL_UNSIGNED_SHORT               0x1403
#define GL_INT                          0x1404
#define GL_UNSIGNED_INT                 0x1405
#define GL_FLOAT                        0x1406

/* pixel formats */
#define GL_STENCIL_INDEX                0x1901
#define GL_DEPTH_COMPONENT              0x1902
#define GL_ALPHA                        0x1906
#define GL_RGB                          0x1907
#define GL_RGBA                         0x1908
#define GL_LUMINANCE                    0x1909
#define GL_LUMINANCE_ALPHA              0x190A
#define GL_RGB4                         0x804F
#define GL_RGB5                         0x8050
#define GL_RGB8                         0x8051
#define GL_RGBA4                        0x8056
#define GL_RGBA8                        0x8058
#define GL_LUMINANCE8                   0x8040
#define GL_LUMINANCE8_ALPHA8            0x8045

/* texturing */
#define GL_TEXTURE_ENV                  0x2300
#define GL_TEXTURE_ENV_MODE             0x2200
#define GL_MODULATE                     0x2100
#define GL_DECAL                        0x2101
#define GL_ADD                          0x0104
#define GL_NEAREST                      0x2600
#define GL_LINEAR                       0x2601
#define GL_NEAREST_MIPMAP_NEAREST       0x2700
#define GL_LINEAR_MIPMAP_NEAREST        0x2701
#define GL_NEAREST_MIPMAP_LINEAR        0x2702
#define GL_LINEAR_MIPMAP_LINEAR         0x2703
#define GL_TEXTURE_MAG_FILTER           0x2800
#define GL_TEXTURE_MIN_FILTER           0x2801
#define GL_TEXTURE_WRAP_S               0x2802
#define GL_TEXTURE_WRAP_T               0x2803
#define GL_TEXTURE_BORDER_COLOR         0x1004
#define GL_CLAMP                        0x2900
#define GL_REPEAT                       0x2901
#define GL_CLAMP_TO_EDGE                0x812F

/* client arrays */
#define GL_VERTEX_ARRAY                 0x8074
#define GL_NORMAL_ARRAY                 0x8075
#define GL_COLOR_ARRAY                  0x8076
#define GL_INDEX_ARRAY                  0x8077
#define GL_TEXTURE_COORD_ARRAY          0x8078
#define GL_INDEX_TYPE                   0x0B85

/* clear bits */
#define GL_DEPTH_BUFFER_BIT             0x00000100
#define GL_STENCIL_BUFFER_BIT           0x00000400
#define GL_COLOR_BUFFER_BIT             0x00004000

/* ARB_multitexture */
#define GL_TEXTURE0_ARB                 0x84C0
#define GL_TEXTURE1_ARB                 0x84C1
#define GL_TEXTURE2_ARB                 0x84C2
#define GL_TEXTURE3_ARB                 0x84C3
#define GL_ACTIVE_TEXTURE_ARB           0x84E0
#define GL_CLIENT_ACTIVE_TEXTURE_ARB    0x84E1
#define GL_MAX_ACTIVE_TEXTURES_ARB      0x84E2

#endif /* __MAC_GL_H__ */
