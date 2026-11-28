//
// Metal port: shaders.
//
// q3_vs / q3_fs reproduce the fixed-function pipeline the renderer was written
// for: vertex colour, two texture units combined with OpenGL texture
// environments, alpha test (a function constant, so pipelines without it keep
// early depth testing), and one user clip plane (portals and mirrors).
// VSUniforms / FSUniforms match mtl_vs_uniforms_t / mtl_fs_uniforms_t in
// mtl_bridge.h. clear_* fill a scissor rectangle; present_* scale the frame
// into the window and apply the gamma ramp.
//
// Xcode compiles this into the app's default.metallib; `make metal` builds
// default.metallib next to quake3-metal.
//

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

// ---------------------------------------------------------------------------
// Ray traced dynamic lights (r_rtShadows; mtl_rt.c, docs/metal-rt.md)
//
// Runs after the opaque surfaces of a world view. For each pixel of the view:
// world position from the depth buffer, normal from the neighbouring depths,
// then for each dynamic light Q3's falloff, N.L and a shadow ray against the
// static world. The light is applied the way Q3's dlight pass blends it
// (dst * light + dst), so the surface colour and lightmap tint it.
// ---------------------------------------------------------------------------

#include <metal_raytracing>
using namespace metal::raytracing;

struct RTLight {
    float4 originRadius;     // matches mtl_rt_light_t
    float4 color;
};

struct RTUniforms {
    float4x4 invViewProj;    // world from OpenGL clip
    float4   viewOrigin;
    int4     viewport;       // x, y, width, height in texture pixels (origin top left)
    int      numLights;
    int      debugView;      // r_rtShadows 2
};

// Q3 squeezes the view weapon into depth [0, 0.3] (RF_DEPTHHACK). No world
// surface is that close to the eye, so such depths are the weapon's: undo it.
static float rtUnhackDepth(float d)
{
    return d < 0.3 ? d / 0.3 : d;
}

static float3 rtWorldPos(int2 p, float d, constant RTUniforms &u)
{
    float2 ndc = (float2(p - u.viewport.xy) + 0.5) / float2(u.viewport.zw) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    float4 w = u.invViewProj * float4(ndc, rtUnhackDepth(d) * 2.0 - 1.0, 1.0);
    return w.xyz / w.w;
}

kernel void rt_dlights(texture2d<float, access::read_write> color [[texture(0)]],
                       depth2d<float, access::read> depth [[texture(1)]],
                       constant RTUniforms &u [[buffer(0)]],
                       constant RTLight *lights [[buffer(1)]],
                       primitive_acceleration_structure world [[buffer(2)]],
                       uint2 tid [[thread_position_in_grid]])
{
    if (int(tid.x) >= u.viewport.z || int(tid.y) >= u.viewport.w) {
        return;
    }
    int2 p = u.viewport.xy + int2(tid);
    float d = depth.read(uint2(p));
    if (d >= 1.0) {
        // nothing drawn (sky is drawn at the far plane too)
        if (u.debugView) {
            color.write(float4(0.0, 0.0, 0.0, 1.0), uint2(p));
        }
        return;
    }
    float3 P = rtWorldPos(p, d, u);

    // normal from the neighbour on each axis whose depth continues the surface best
    int2 lo = u.viewport.xy, hi = u.viewport.xy + u.viewport.zw - 1;
    int2 px0 = clamp(p - int2(1, 0), lo, hi), px1 = clamp(p + int2(1, 0), lo, hi);
    int2 py0 = clamp(p - int2(0, 1), lo, hi), py1 = clamp(p + int2(0, 1), lo, hi);
    float3 X0 = rtWorldPos(px0, depth.read(uint2(px0)), u), X1 = rtWorldPos(px1, depth.read(uint2(px1)), u);
    float3 Y0 = rtWorldPos(py0, depth.read(uint2(py0)), u), Y1 = rtWorldPos(py1, depth.read(uint2(py1)), u);
    float3 dx = length_squared(X1 - P) < length_squared(P - X0) ? X1 - P : P - X0;
    float3 dy = length_squared(Y1 - P) < length_squared(P - Y0) ? Y1 - P : P - Y0;
    float3 N = cross(dx, dy);
    float nlen = length(N);
    if (nlen < 1e-6) {
        if (u.debugView) {
            color.write(float4(0.0, 0.0, 0.0, 1.0), uint2(p));
        }
        return;
    }
    N /= nlen;
    float3 toEye = u.viewOrigin.xyz - P;
    if (dot(N, toEye) < 0.0) {
        N = -N;
    }

    intersector<> shadow;
    shadow.accept_any_intersection(true);
    shadow.assume_geometry_type(geometry_type::triangle);
    shadow.force_opacity(forced_opacity::opaque);

    float3 sum = 0.0;
    float shadowed = 0.0;    // light the shadow rays blocked (debug view)
    for (int i = 0; i < u.numLights; i++) {
        float3 L = lights[i].originRadius.xyz - P;
        float dist = length(L);
        float radius = lights[i].originRadius.w;
        // Q3's dlight shape (tr_shade.c ProjectDlightTexture): horizontally,
        // the dlight image (tr_image.c), 16 texels across +-radius/2 with
        // brightness 4000 / texelDist^2 clamped to 255 and cut to 0 below 75,
        // which is full intensity out to 0.25 radius and gone at 0.456 radius
        // (faded out here instead of ending in a ring); vertically, full
        // strength to radius/2, then linear to 0 at radius.
        float rxy = length(L.xy) / radius;
        float dz = abs(L.z);
        if (rxy >= 0.456 || dz >= radius || dist < 1e-3) {
            continue;
        }
        float intensity = min(1.0, 0.0613 / max(rxy * rxy, 1e-6)) * saturate((0.456 - rxy) / 0.08)
                        * min(1.0, 2.0 * (radius - dz) / radius);
        float3 Ldir = L / dist;
        float ndl = dot(N, Ldir);
        if (ndl <= 0.0) {
            continue;
        }
        // Q3 ignores orientation; half Lambert keeps its brightness on grazing
        // surfaces while still shaping them
        ndl = 0.5 + 0.5 * ndl;
        ray r0;
        r0.origin = P + N * 0.5;
        r0.direction = Ldir;
        r0.min_distance = 0.0;
        r0.max_distance = max(dist - 1.0, 0.0);
        if (shadow.intersect(r0, world).type != intersection_type::none) {
            shadowed += intensity * ndl;
            continue;
        }
        sum += lights[i].color.rgb * (intensity * ndl);
    }
    if (u.debugView) {
        // r_rtShadows 2: only the ray traced light on black; blue where the
        // shadow rays were blocked
        float3 c = sum + float3(0.15, 0.3, 1.0) * saturate(shadowed) * 0.6;
        color.write(float4(saturate(c), 1.0), uint2(p));
        return;
    }
    if (all(sum == 0.0)) {
        return;
    }
    float4 c = color.read(uint2(p));
    color.write(float4(saturate(c.rgb * (1.0 + sum)), c.a), uint2(p));
}
