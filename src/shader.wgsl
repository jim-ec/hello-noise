struct Parameters {
    panning: vec2<f32>,
    aspect_ratio: f32,
    time: f32,
    zoom: f32,
    mode: u32,
}

var<push_constant> parameters: Parameters;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

const MODE_UV: u32 = 0;
const MODE_VALUE: u32 = 1;
const MODE_PERLIN: u32 = 2;
const MODE_SIMPLEX: u32 = 3;

const TAU: f32 = 6.2831853072;

@vertex
fn vertex(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Bit patterns of vertices:
    // A - B
    // | /
    // C
    // Tri: A C B
    // X: 0 1 0 => 0b010 => 0x2
    // Y: 0 0 1 => 0b001 => 0x1
    let id = (vec2(0x2u, 0x1u) >> vec2(vertex_index)) & vec2(1u); // [0, 1]^2
    let uv = vec2<i32>(id << vec2(2u)) - 1; // [-1, 3]^2
    var out: VertexOutput;
    out.uv = vec2<f32>(parameters.panning + exp(parameters.zoom) * (vec2<f32>(uv))) * vec2(parameters.aspect_ratio, 1.0);
    out.position = vec4(vec2<f32>(uv), 0.0, 1.0);
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let noise = noise(parameters.mode, uv);
    let color = color(parameters.mode, noise, uv);
    return vec4(pow(color, vec3(2.2)), 1.0);
}

fn color(mode: u32, noise: f32, uv: vec2<f32>) -> vec3<f32> {
    if (MODE_UV == parameters.mode) {
        return vec3(fract(uv), 0.0);
    }
    else {
        return vec3(noise) * 0.5 + 0.5;
    }
}

fn noise(mode: u32, uv: vec2<f32>) -> f32 {
    if (MODE_VALUE == parameters.mode) {
        return value_noise(uv);
    }
    else if (MODE_PERLIN == parameters.mode) {
        return perlin_noise(uv);
    }
    else if (MODE_SIMPLEX == parameters.mode) {
        return simplex_noise(uv);
    }
    else {
        return 0.0;
    }
}

fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);

    let n00 = rand2(i + vec2(0, 0));
    let n01 = rand2(i + vec2(0, 1));
    let n10 = rand2(i + vec2(1, 0));
    let n11 = rand2(i + vec2(1, 1));

    let nx0 = mix(n00, n10, k(f.x));
    let nx1 = mix(n01, n11, k(f.x));
    let nxy = mix(nx0, nx1, k(f.y));

    return nxy;
}

fn perlin_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);

    let g00 = unit_vector(rand2(i + vec2(0, 0)) * TAU);
    let g01 = unit_vector(rand2(i + vec2(0, 1)) * TAU);
    let g10 = unit_vector(rand2(i + vec2(1, 0)) * TAU);
    let g11 = unit_vector(rand2(i + vec2(1, 1)) * TAU);

    let n00 = dot(g00, f - vec2(0, 0));
    let n01 = dot(g01, f - vec2(0, 1));
    let n10 = dot(g10, f - vec2(1, 0));
    let n11 = dot(g11, f - vec2(1, 1));

    let nx0 = mix(n00, n10, k(f.x));
    let nx1 = mix(n01, n11, k(f.x));
    let nxy = mix(nx0, nx1, k(f.y));

    return nxy;
}

fn simplex_noise(p: vec2<f32>) -> f32 {
    const F2 = 0.5 * (sqrt(3.0) - 1.0);
    const G2 = (3.0 - sqrt(3.0)) / 6.0;

    // The simplex origin in skewed space
    let s = vec2(floor(p + (p.x + p.y) * F2));

    // The simplex origin in world space
    let i = s - (s.x + s.y) * G2;

    // Offset to the simplex origin in world space
    let f0 = p - i;

    // The intermediately traversed vertex relative to the simplex origin
    let v1: vec2<f32> = select(vec2(0.0, 1.0), vec2(1.0, 0.0), f0.x > f0.y);

    // Offsets to the other two simplex vertices in world space
    let f1 = f0 - v1 + G2;
    let f2 = f0 - 1.0 + 2.0 * G2;

    let r = vec3(dot(f0, f0), dot(f1, f1), dot(f2, f2));
    let m = max(vec3(0.0), 0.5 - r);

    // Generate normalized gradient vectors at each simplex vertex
    let g0 = unit_vector(rand2(s) * TAU);
    let g1 = unit_vector(rand2(s + v1) * TAU);
    let g2 = unit_vector(rand2(s + 1.0) * TAU);

    let n = dot(m * m * m * m, vec3(dot(g0, f0), dot(g1, f1), dot(g2, f2)));

    return 70.0 * n;
}

fn unit_vector(angle: f32) -> vec2<f32> {
    return vec2(cos(angle), sin(angle));
}

fn k(t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t * t2;
    let t4 = t * t2;
    let t5 = t * t2;

    // return 6 * t5 - 15 * t4 + 10 * t3;
    return 3 * t2 - 2 * t3;
}

fn rand2(v: vec2<f32>) -> f32 {
    return rand3(vec3(v, 0.0));
}

fn rand3(v: vec3<f32>) -> f32 {
    let x = bitcast<u32>(v.x);
    let y = bitcast<u32>(v.y);
    let z = bitcast<u32>(v.z);
    let seed = x ^ (y * 0x9E3779B9u) ^ (z * 0x1B873593u);
    let hash = hash_murmur(seed);
    return u32_to_unit_f32(hash) * 2.0 - 1.0;
}

fn u32_to_unit_f32(x: u32) -> f32 {
    let mantissa = x & 0x007FFFFFu;
    let one_point = mantissa | 0x3F800000u;
    return bitcast<f32>(one_point) - 1.0;
}

fn hash_murmur(x: u32) -> u32 {
    var h = x;
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}
