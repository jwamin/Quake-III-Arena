//
// Metal port: shader source, compiled at startup with MTLDevice.makeLibrary(source:).
//
// q3_vs / q3_fs reproduce the fixed-function pipeline the renderer was written
// for: vertex colour, two texture units combined with OpenGL texture
// environments, alpha test, and one user clip plane (portals and mirrors).
// The uniform structs match mtl_vs_uniforms_t / mtl_fs_uniforms_t in mtl_bridge.h.
//

let q3ShaderSource = """
#include <metal_stdlib>
using namespace metal;

struct Vertex {
    packed_float3 xyz;
    packed_float2 st0;
    packed_float2 st1;
    uchar4        color;
};

struct VSUniforms {
    float4x4 mvp;
    float4   clipPlane;
    int      clipEnable;
};

struct FSUniforms {
    int   tex0Enable;
    int   tex1Enable;
    int   env0;
    int   env1;
    int   alphaFunc;
    float alphaRef;
};

struct VOut {
    float4 position [[position]];
    float2 st0;
    float2 st1;
    float4 color;
    float  clip [[clip_distance]] [1];
    float  pointSize [[point_size]];
};

struct FIn {
    float4 position [[position]];
    float2 st0;
    float2 st1;
    float4 color;
};

constant bool hasAlphaTest [[function_constant(0)]];

vertex VOut q3_vs(uint vid [[vertex_id]],
                  const device Vertex *verts [[buffer(0)]],
                  constant VSUniforms &u [[buffer(1)]])
{
    Vertex v = verts[vid];
    float4 p = float4(float3(v.xyz), 1.0);
    float4 c = u.mvp * p;
    c.z = (c.z + c.w) * 0.5;             // OpenGL clip space z [-w, w] -> Metal [0, w]

    VOut o;
    o.position = c;
    o.st0 = float2(v.st0);
    o.st1 = float2(v.st1);
    o.color = float4(v.color) * (1.0 / 255.0);
    o.clip[0] = u.clipEnable ? dot(u.clipPlane, p) : 1.0;
    o.pointSize = 1.0;
    return o;
}

// OpenGL 1.x texture environment
static float4 texEnv(int env, float4 c, float4 t)
{
    switch (env) {
    case 1:  return t;                                         // GL_REPLACE
    case 2:  return float4(mix(c.rgb, t.rgb, t.a), c.a);       // GL_DECAL
    case 3:  return float4(c.rgb + t.rgb, c.a * t.a);          // GL_ADD
    default: return c * t;                                     // GL_MODULATE
    }
}

static bool alphaPasses(int func, float a, float ref)
{
    switch (func) {
    case 0:  return false;
    case 1:  return a <  ref;
    case 2:  return a == ref;
    case 3:  return a <= ref;
    case 4:  return a >  ref;
    case 5:  return a != ref;
    case 6:  return a >= ref;
    default: return true;
    }
}

fragment float4 q3_fs(FIn in [[stage_in]],
                      constant FSUniforms &u [[buffer(0)]],
                      texture2d<float> tex0 [[texture(0)]],
                      texture2d<float> tex1 [[texture(1)]],
                      sampler samp0 [[sampler(0)]],
                      sampler samp1 [[sampler(1)]])
{
    float4 c = in.color;
    if (u.tex0Enable) {
        c = texEnv(u.env0, c, tex0.sample(samp0, in.st0));
    }
    if (u.tex1Enable) {
        c = texEnv(u.env1, c, tex1.sample(samp1, in.st1));
    }
    if (hasAlphaTest && !alphaPasses(u.alphaFunc, c.a, u.alphaRef)) {
        discard_fragment();
    }
    return c;
}

// ---- clears of a scissor rectangle -------------------------------------------------

struct ClearUniforms {
    float4 color;
    float  depth;
};

vertex float4 clear_vs(uint vid [[vertex_id]], constant ClearUniforms &u [[buffer(0)]])
{
    float2 p = float2((vid << 1) & 2, vid & 2) * 2.0 - 1.0;
    return float4(p, u.depth, 1.0);
}

fragment float4 clear_fs(constant ClearUniforms &u [[buffer(0)]])
{
    return u.color;
}

// ---- presentation: scale to the window and apply the gamma ramp -------------------

struct PresentOut {
    float4 position [[position]];
    float2 uv;
};

vertex PresentOut present_vs(uint vid [[vertex_id]])
{
    float2 uv = float2((vid << 1) & 2, vid & 2);
    PresentOut o;
    o.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    o.uv = float2(uv.x, 1.0 - uv.y);
    return o;
}

fragment float4 present_fs(PresentOut in [[stage_in]],
                           texture2d<float> frame [[texture(0)]],
                           texture2d<float> gamma [[texture(1)]])
{
    constexpr sampler frameSampler(filter::linear, address::clamp_to_edge);
    constexpr sampler rampSampler(filter::linear, address::clamp_to_edge);
    float3 c = frame.sample(frameSampler, in.uv).rgb;
    float3 uv = (c * 255.0 + 0.5) / 256.0;
    return float4(gamma.sample(rampSampler, float2(uv.r, 0.5)).r,
                  gamma.sample(rampSampler, float2(uv.g, 0.5)).g,
                  gamma.sample(rampSampler, float2(uv.b, 0.5)).b,
                  1.0);
}
"""
