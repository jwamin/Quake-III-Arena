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

// ---------------------------------------------------------------------------
// Relighting from the map's own lights (r_rtRelight; mtl_rtlights.c,
// docs/metal-rt-relight.md)
//
// For pixels the raster pass marked (stencil bit 0x80), the framebuffer holds
// albedo (lightmaps were drawn white). This evaluates q3map's light formulas
// (q3map/light.c LightingAtSample) for the lights listed in the pixel's grid
// cell, traces shadow rays for the strongest ones, adds the sun and the
// worldspawn ambient, and converts to lightmap texel units the way q3map and
// the renderer do (ColorToBytes, R_ColorShiftLightingBytes).
// ---------------------------------------------------------------------------

struct EnvLight {           // matches mtl_rt_envlight_t
    packed_float3 origin;
    float         photons;
    packed_float3 color;
    float         range;
    packed_float3 normal;
    float         radiusByDist;
    int           type;     // 0 point, 1 spot, 2 area
    int           linear;
    int           twoSided;
    int           pad;
};

struct EnvInfo {            // matches mtl_rt_envinfo_t
    packed_float3 gridOrigin;
    float         cellSize;
    packed_int3   gridDims;
    int           numLights;
    packed_float3 ambient;
    float         lightScale;
    packed_float3 sunColor;
    float         pad0;
    packed_float3 sunDir;
    float         pad1;
};

struct RelightUniforms {
    float4x4 invViewProj;
    float4   viewOrigin;
    int4     viewport;
    int      budget;
    float    relightScale;
    int      sun;
    int      mode;           // r_rtRelight: 2 = no shadow rays (diagnosis)
};

constant int MAX_BUDGET = 32;   // MTL_RT_MAX_BUDGET

static float3 rlWorldPos(int2 p, float d, constant RelightUniforms &u)
{
    float2 ndc = (float2(p - u.viewport.xy) + 0.5) / float2(u.viewport.zw) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    float4 w = u.invViewProj * float4(ndc, rtUnhackDepth(d) * 2.0 - 1.0, 1.0);
    return w.xyz / w.w;
}

// q3map's light added at a sample, before occlusion (0-255 units, colourless)
static float q3mapAdd(const device EnvLight &l, float3 P, float3 N, thread float3 &dir, thread float &dist)
{
    float3 L = float3(l.origin) - P;
    dist = length(L);
    dir = L / max(dist, 1e-4);
    float d = max(dist, 16.0);       // "clamp the distance to prevent super hot spots"
    float angle = dot(N, dir);

    if (l.type == 1) {               // spotlight
        float3 ln = float3(l.normal);
        float distByNormal = -dot(L, ln);
        if (distByNormal < 0.0) {
            return 0.0;
        }
        float3 pointAtDist = float3(l.origin) + distByNormal * ln;
        float radiusAtDist = l.radiusByDist * distByNormal;
        float sampleRadius = length(P - pointAtDist);
        if (sampleRadius >= radiusAtDist) {
            return 0.0;
        }
        float coneScale = sampleRadius <= radiusAtDist - 32.0 ? 1.0 : (radiusAtDist - sampleRadius) / 32.0;
        return l.photons / (d * d) * angle * coneScale;
    }
    if (l.type == 2) {               // area light patch
        if (angle <= 0.0) {
            return 0.0;
        }
        float facing = -dot(float3(l.normal), dir);
        if (l.twoSided) {
            facing = abs(facing);
        }
        angle *= facing;
        if (angle <= 0.0) {
            return 0.0;
        }
    }
    if (l.linear) {
        return max(angle * l.photons * (1.0 / 8000.0) - d, 0.0);
    }
    return l.photons / (d * d) * angle;
}

kernel void rt_relight(texture2d<float, access::read_write> color [[texture(0)]],
                       depth2d<float, access::read> depth [[texture(1)]],
                       texture2d<uint, access::read> stencil [[texture(2)]],
                       constant RelightUniforms &u [[buffer(0)]],
                       constant EnvInfo &info [[buffer(1)]],
                       const device EnvLight *lights [[buffer(2)]],
                       const device uint *cellStart [[buffer(3)]],
                       const device uint *cellLights [[buffer(4)]],
                       primitive_acceleration_structure world [[buffer(5)]],
                       const device float4 *cubes [[buffer(6)]],
                       uint2 tid [[thread_position_in_grid]])
{
    if (int(tid.x) >= u.viewport.z || int(tid.y) >= u.viewport.w) {
        return;
    }
    int2 p = u.viewport.xy + int2(tid);
    if ((stencil.read(uint2(p)).r & 0x80u) == 0) {
        return;                      // not a relit surface
    }
    float d = depth.read(uint2(p));
    if (d >= 1.0) {
        return;
    }
    float3 P = rlWorldPos(p, d, u);

    int2 lo = u.viewport.xy, hi = u.viewport.xy + u.viewport.zw - 1;
    int2 px0 = clamp(p - int2(1, 0), lo, hi), px1 = clamp(p + int2(1, 0), lo, hi);
    int2 py0 = clamp(p - int2(0, 1), lo, hi), py1 = clamp(p + int2(0, 1), lo, hi);
    float3 X0 = rlWorldPos(px0, depth.read(uint2(px0)), u), X1 = rlWorldPos(px1, depth.read(uint2(px1)), u);
    float3 Y0 = rlWorldPos(py0, depth.read(uint2(py0)), u), Y1 = rlWorldPos(py1, depth.read(uint2(py1)), u);
    float3 dx = length_squared(X1 - P) < length_squared(P - X0) ? X1 - P : P - X0;
    float3 dy = length_squared(Y1 - P) < length_squared(P - Y0) ? Y1 - P : P - Y0;
    float3 N = cross(dx, dy);
    float nlen = length(N);
    if (nlen < 1e-6) {
        return;
    }
    N /= nlen;
    if (dot(N, u.viewOrigin.xyz - P) < 0.0) {
        N = -N;
    }

    intersector<> shadow;
    shadow.accept_any_intersection(true);
    shadow.assume_geometry_type(geometry_type::triangle);
    shadow.force_opacity(forced_opacity::opaque);

    // The cell's list holds map lights (point, spot) first, then area
    // patches, each by strength. Half the shadow-ray budget goes to each
    // group; what a group cannot trace is scaled by that group's measured
    // visibility, so sky and area light is shadowed by area rays and lamp
    // light by lamp rays.
    int budget = clamp(u.budget, 2, MAX_BUDGET);
    int budgetPoint = budget / 2, budgetArea = budget - budgetPoint;
    int tracedPoint = 0, tracedArea = 0;
    float3 lit = 0.0, restPoint = 0.0, restArea = 0.0;
    float totalPoint = 0.0, visPoint = 0.0, totalArea = 0.0, visArea = 0.0;

    float3 cellF = (P - float3(info.gridOrigin)) / info.cellSize;
    int3 dims = int3(info.gridDims);
    int3 cell = clamp(int3(floor(cellF)), int3(0), dims - 1);
    uint ci = uint(cell.x + cell.y * dims.x + cell.z * dims.x * dims.y);
    uint first = cellStart[ci], last = cellStart[ci + 1];

    for (uint k = first; k < last; k++) {
        const device EnvLight &l = lights[cellLights[k]];
        bool area = l.type == 2;
        float3 dir;
        float dist;
        float add = q3mapAdd(l, P, N, dir, dist);
        // q3map's cutoff for point and spot lights; area lights (exact form
        // factor in q3map) have none, only the grid's range limit
        if (add <= (area ? 0.0 : 1.0)) {
            continue;
        }
        if (area ? tracedArea >= budgetArea : tracedPoint >= budgetPoint) {
            if (area) {
                restArea += l.color * add;
            } else {
                restPoint += l.color * add;
            }
            continue;
        }
        ray r0;
        r0.origin = P + N * 0.5;
        r0.direction = dir;
        r0.min_distance = 0.0;
        r0.max_distance = max(dist - 1.0, 0.0);
        bool visible = u.mode == 2 || shadow.intersect(r0, world).type == intersection_type::none;
        if (area) {
            tracedArea++;
            totalArea += add;
            visArea += visible ? add : 0.0;
        } else {
            tracedPoint++;
            totalPoint += add;
            visPoint += visible ? add : 0.0;
        }
        if (visible) {
            lit += l.color * add;
        }
    }

    // the many weak lights beyond the cell's list (almost all area patches):
    // its ambient cube, interpolated between the eight nearest cell centres
    // and weighted by the normal (+x -x +y -y +z -z)
    {
        float3 f = (P - float3(info.gridOrigin)) / info.cellSize - 0.5;
        int3 i0 = clamp(int3(floor(f)), int3(0), dims - 1);
        int3 i1 = min(i0 + 1, dims - 1);
        float3 t = saturate(f - float3(i0));
        float3 w2 = N * N;
        int3 side = int3(N.x < 0.0 ? 1 : 0, N.y < 0.0 ? 3 : 2, N.z < 0.0 ? 5 : 4);
        for (int c = 0; c < 8; c++) {
            int3 cc = int3((c & 1) ? i1.x : i0.x, (c & 2) ? i1.y : i0.y, (c & 4) ? i1.z : i0.z);
            float3 tw = float3((c & 1) ? t.x : 1.0 - t.x, (c & 2) ? t.y : 1.0 - t.y, (c & 4) ? t.z : 1.0 - t.z);
            float weight = tw.x * tw.y * tw.z;
            if (weight <= 0.0) {
                continue;
            }
            uint base = uint(cc.x + cc.y * dims.x + cc.z * dims.x * dims.y) * 6;
            restArea += weight * (w2.x * cubes[base + side.x].rgb + w2.y * cubes[base + side.y].rgb
                                  + w2.z * cubes[base + side.z].rgb);
        }
    }
    // untraced light, scaled by its group's visibility (as is where nothing
    // of the group was traced)
    lit += restPoint * (totalPoint > 0.0 ? visPoint / totalPoint : 1.0);
    lit += restArea * (totalArea > 0.0 ? visArea / totalArea : 1.0);

    // sun: lit when the ray toward it leaves the world (the sky is not in it)
    float3 sunColor = float3(info.sunColor);
    if (u.sun && any(sunColor > 0.0)) {
        float3 sd = normalize(float3(info.sunDir));
        float angle = dot(N, sd);
        if (angle > 0.0) {
            ray rs;
            rs.origin = P + N * 0.5;
            rs.direction = sd;
            rs.min_distance = 0.0;
            rs.max_distance = 65536.0;
            if (u.mode == 2 || shadow.intersect(rs, world).type == intersection_type::none) {
                lit += sunColor * angle;
            }
        }
    }

    // to lightmap texel units: q3map clamps by colour normalization, the
    // renderer shifts by the overbright difference and normalizes again
    float3 light = (float3(info.ambient) + lit) * info.lightScale * u.relightScale;
    float m = max(light.r, max(light.g, light.b));
    if (m > 1.0) {
        light /= m;
    }
    float4 c = color.read(uint2(p));
    color.write(float4(c.rgb * light, c.a), uint2(p));
}
