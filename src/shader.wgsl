struct Parameters {
    panning: vec2<f32>,
    aspect_ratio: f32,
    time: f32,
    zoom: f32,
    mode: u32,
    dim: u32,
    warp: u32,
    warp_strength: f32,
    octaves: u32,
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
    let noise = warped_noise(uv);
    let color = color(noise, uv);
    return vec4(pow(color, vec3(2.2)), 1.0);
}

fn color(noise: f32, uv: vec2<f32>) -> vec3<f32> {
    if (MODE_UV == parameters.mode) {
        return vec3(fract(uv), 0.0);
    }
    else {
        return vec3(noise) * 0.5 + 0.5;
    }
}

fn warped_noise(p: vec2<f32>) -> f32 {
    var u = vec2(0.0);
    for (var i = 0u; i < parameters.warp; i++) {
        u = vec2(
            fbm(p + parameters.warp_strength * u + vec2(27.0, 7.0)),
            fbm(p + parameters.warp_strength * u + vec2(42.0, 31.0)),
        );
    }
    return fbm(p + parameters.warp_strength * u);
}

fn fbm(p: vec2<f32>) -> f32 {
    const LACUNARITY: f32 = 2.0;
    const PERSISTENCE: f32 = 0.5;

    var n = 0.0;

    var amplitude = 0.5;
    var frequency = 1.0;

    var q = p;

    for (var i = 0u; i < parameters.octaves; i++) {
        n += amplitude * noise(q);
        q *= LACUNARITY;
        amplitude *= PERSISTENCE;
        frequency *= LACUNARITY;
    }

    return n;
}

fn noise(p: vec2<f32>) -> f32 {
    if (parameters.dim == 2) {
        return noise_2d(p);
    }
    else if (parameters.dim == 3) {
        return noise_3d(vec3(p, parameters.time));
    }
    else {
        return 0.0;
    }
}

fn noise_2d(x: vec2<f32>) -> f32 {
    if (MODE_VALUE == parameters.mode) {
        return value_noise_2d(x);
    }
    else if (MODE_PERLIN == parameters.mode) {
        return perlin_noise_2d(x);
    }
    else if (MODE_SIMPLEX == parameters.mode) {
        // return simplex_noise(uv);
        return simplex_noise_2d(x);
    }
    else {
        return 0.0;
    }
}

fn noise_3d(p: vec3<f32>) -> f32 {
    if (MODE_VALUE == parameters.mode) {
        return value_noise_3d(p);
    }
    else if (MODE_PERLIN == parameters.mode) {
        return perlin_noise_3d(p);
    }
    else if (MODE_SIMPLEX == parameters.mode) {
        return simplex_noise_3d(p);
    }
    else {
        return 0.0;
    }
}

fn value_noise_2d(p: vec2<f32>) -> f32 {
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

fn value_noise_3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);

    let u = k(f.x);
    let v = k(f.y);
    let w = k(f.z);

    let n000 = rand3(i + vec3(0.0, 0.0, 0.0));
    let n100 = rand3(i + vec3(1.0, 0.0, 0.0));
    let n010 = rand3(i + vec3(0.0, 1.0, 0.0));
    let n110 = rand3(i + vec3(1.0, 1.0, 0.0));
    let n001 = rand3(i + vec3(0.0, 0.0, 1.0));
    let n101 = rand3(i + vec3(1.0, 0.0, 1.0));
    let n011 = rand3(i + vec3(0.0, 1.0, 1.0));
    let n111 = rand3(i + vec3(1.0, 1.0, 1.0));

    let nx00 = mix(n000, n100, u);
    let nx10 = mix(n010, n110, u);
    let nx01 = mix(n001, n101, u);
    let nx11 = mix(n011, n111, u);
    let nxy0 = mix(nx00, nx10, v);
    let nxy1 = mix(nx01, nx11, v);
    let nxyz = mix(nxy0, nxy1, w);

    return nxyz;
}

fn perlin_noise_2d(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);

    let g00 = rand_vec2(i + vec2(0, 0));
    let g01 = rand_vec2(i + vec2(0, 1));
    let g10 = rand_vec2(i + vec2(1, 0));
    let g11 = rand_vec2(i + vec2(1, 1));

    let n00 = dot(g00, f - vec2(0, 0));
    let n01 = dot(g01, f - vec2(0, 1));
    let n10 = dot(g10, f - vec2(1, 0));
    let n11 = dot(g11, f - vec2(1, 1));

    let nx0 = mix(n00, n10, k(f.x));
    let nx1 = mix(n01, n11, k(f.x));
    let nxy = mix(nx0, nx1, k(f.y));

    return nxy;
}

fn perlin_noise_3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);

    let g000 = rand_vec3(i + vec3(0.0, 0.0, 0.0));
    let g100 = rand_vec3(i + vec3(1.0, 0.0, 0.0));
    let g010 = rand_vec3(i + vec3(0.0, 1.0, 0.0));
    let g110 = rand_vec3(i + vec3(1.0, 1.0, 0.0));
    let g001 = rand_vec3(i + vec3(0.0, 0.0, 1.0));
    let g101 = rand_vec3(i + vec3(1.0, 0.0, 1.0));
    let g011 = rand_vec3(i + vec3(0.0, 1.0, 1.0));
    let g111 = rand_vec3(i + vec3(1.0, 1.0, 1.0));

    let n000 = dot(g000, f - vec3(0.0, 0.0, 0.0));
    let n100 = dot(g100, f - vec3(1.0, 0.0, 0.0));
    let n010 = dot(g010, f - vec3(0.0, 1.0, 0.0));
    let n110 = dot(g110, f - vec3(1.0, 1.0, 0.0));
    let n001 = dot(g001, f - vec3(0.0, 0.0, 1.0));
    let n101 = dot(g101, f - vec3(1.0, 0.0, 1.0));
    let n011 = dot(g011, f - vec3(0.0, 1.0, 1.0));
    let n111 = dot(g111, f - vec3(1.0, 1.0, 1.0));

    let u = k(f.x);
    let v = k(f.y);
    let w = k(f.z);

    let nx00 = mix(n000, n100, u);
    let nx10 = mix(n010, n110, u);
    let nx01 = mix(n001, n101, u);
    let nx11 = mix(n011, n111, u);
    let nxy0 = mix(nx00, nx10, v);
    let nxy1 = mix(nx01, nx11, v);
    let nxyz = mix(nxy0, nxy1, w);

    return nxyz;
}

fn simplex_noise_2d(p: vec2<f32>) -> f32 {
    const F = 0.5 * (sqrt(3.0) - 1.0);
    const G = (3.0 - sqrt(3.0)) / 6.0;

    // The simplex origin in skewed space
    let s = vec2(floor(p + (p.x + p.y) * F));

    // The simplex origin in world space
    let i = s - (s.x + s.y) * G;

    // Unskew the simplex origin back to world space
    let f0 = p - i;

    // The intermediately traversed vertex relative to the simplex origin
    let v1: vec2<f32> = select(vec2(0.0, 1.0), vec2(1.0, 0.0), f0.x > f0.y);

    // Offsets to the other two simplex vertices in world space
    let f1 = f0 - v1 + G;
    let f2 = f0 - 1.0 + 2.0 * G;

    // Generate normalized gradient vectors at each simplex vertex
    let g0 = rand_vec2(s);
    let g1 = rand_vec2(s + v1);
    let g2 = rand_vec2(s + 1.0);

    let r = vec3(dot(f0, f0), dot(f1, f1), dot(f2, f2));
    let m = max(vec3(0.0), 0.5 - r);

    let m2 = m * m;
    let m4 = m2 * m2;

    let grad_dots = vec3(
        dot(g0, f0),
        dot(g1, f1),
        dot(g2, f2),
    );

    let n = dot(m4, grad_dots);

    return 70.0 * n;
}

fn simplex_noise_3d(p: vec3<f32>) -> f32 {
    const F = 1.0 / 3.0;
    const G = 1.0 / 6.0;

    // The simplex origin in skewed space
    let s = floor(p + (p.x + p.y + p.z) * F);

    // Unskew the simplex origin back to world space
    let i = s - (s.x + s.y + s.z) * G;
    let f0 = p - i; // The x,y,z distances from the cell origin

    // Compare components: x>=y, y>=z, z>=x
    let e = step(vec3(0.0), f0 - f0.yzx);

    // The intermediately traversed vertices relative to the simplex origin
    let v1 = e * (1.0 - e.zxy);
    let v2 = 1.0 - e.zxy * (1.0 - e);

    // Offsets to the other three simplex vertices in world space
    let f1 = f0 - v1 + G;
    let f2 = f0 - v2 + 2.0 * G;
    let f3 = f0 - 1.0 + 3.0 * G;

    // Generate normalized gradient vectors at each simplex vertex
    let g0 = rand_vec3(s);
    let g1 = rand_vec3(s + v1);
    let g2 = rand_vec3(s + v2);
    let g3 = rand_vec3(s + 1.0);

    let r = vec4(dot(f0, f0), dot(f1, f1), dot(f2, f2), dot(f3, f3));
    let m = max(vec4(0.0), 0.5 - r);

    let m2 = m * m;
    let m4 = m2 * m2;

    let grad_dots = vec4(
        dot(g0, f0),
        dot(g1, f1),
        dot(g2, f2),
        dot(g3, f3)
    );

    let n = dot(m4, grad_dots);

    return 32.0 * n;
}

fn rand_vec2(p: vec2<f32>) -> vec2<f32> {
    let x = rand2(p);
    let y = rand2(p + vec2(42.0, 0.0));
    let z = rand2(p + vec2(0.0, 42.0));
    return normalize(vec2(x, y));
}

fn rand_vec3(p: vec3<f32>) -> vec3<f32> {
    let x = rand3(p);
    let y = rand3(p + vec3(42.0, 0.0, 0.0));
    let z = rand3(p + vec3(0.0, 42.0, 0.0));
    return normalize(vec3(x, y, z));
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
