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
    lacunarity: f32,
    persistence: f32,
    levels: u32,
    saturation: f32,
    dither: u32,
    output: u32,
}

var<push_constant> parameters: Parameters;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) uv_screen_space: vec2<f32>,
}

const MODE_VALUE: u32 = 1;
const MODE_PERLIN: u32 = 2;
const MODE_SIMPLEX: u32 = 3;
const MODE_WORLEY: u32 = 4;

const OUTPUT_VALUE: u32 = 0;
const OUTPUT_GRADIENT: u32 = 1;
const OUTPUT_SPLIT: u32 = 2;

const TAU: f32 = 6.2831853072;

struct Noise {
    f: f32,
    df: vec3<f32>,
}

@vertex
fn vertex(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Bit patterns of vertices:
    // C
    // | \
    // A - B
    // X: 0 1 0 => 0b010 => 0x2
    // Y: 0 0 1 => 0b001 => 0x1
    let id = (vec2(0x2u, 0x1u) >> vec2(vertex_index)) & vec2(1u); // [0, 1]^2
    let uv = vec2<i32>(id << vec2(2u)) - 1; // [-1, 3]^2
    var out: VertexOutput;
    out.uv = vec2<f32>(parameters.panning + exp(parameters.zoom) * (vec2<f32>(uv))) * vec2(parameters.aspect_ratio, 1.0);
    out.uv_screen_space = vec2<f32>(uv);
    out.position = vec4(vec2<f32>(uv), 0.0, 1.0);
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let noise = warp_noise(embed(uv));

    var color = select(
        vec3(noise.f) * 0.5 + 0.5,
        vec3(normalize(noise.df) * 0.5 + 0.5),
        vec3<bool>(select(
            bool(parameters.output) && in.uv_screen_space.x + -0.2 * in.uv_screen_space.y < 0.0,
            bool(parameters.output),
            parameters.output != OUTPUT_SPLIT
        )),
    );

    color = quantize_color(color, in.position.xy);
    color = saturate_color(color);

    return vec4(pow(color, vec3(2.2)), 1.0);
}

fn quantize_color(color: vec3<f32>, screen_space_position: vec2<f32>) -> vec3<f32> {
    let step_size = 1.0 / f32(parameters.levels);
    let dither_noise = rand2(screen_space_position) - 0.5;
    var dither = select(vec3(0.0), vec3(dither_noise * step_size), vec3<bool>(parameters.dither));
    return select(
        color,
        trunc((color + dither) * f32(parameters.levels)) / f32(parameters.levels),
        parameters.levels > 0,
    );
}

fn saturate_color(color: vec3<f32>) -> vec3<f32> {
    let k = parameters.saturation;
    let lower = 0.5 * pow(2.0 * color, vec3(k));
    let upper = 1.0 - 0.5 * pow(2.0 * (1.0 - color), vec3(k));
    return select(upper, lower, color < vec3(0.5));
}

fn embed(p: vec2<f32>) -> vec3<f32> {
    if (parameters.dim == 2) {
        return vec3(p, 0.0);
    }
    else if (parameters.dim == 3) {
        return vec3(p, parameters.time);
    }
    else {
        return vec3();
    }
}

fn warp_noise(p: vec3<f32>) -> Noise {
    let s = parameters.warp_strength;
    let I = mat3x3<f32>(
        vec3(1.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0),
        vec3(0.0, 0.0, 1.0)
    );

    // u: The displacement vector
    var u = vec3(0.0);

    // Ju: The Jacobian (derivative) of u with respect to p
    // Initially zero because u is constant (0,0)
    var Ju = mat3x3<f32>(vec3(0.0), vec3(0.0), vec3(0.0));

    for (var i = 0u; i < parameters.warp; i++) {
        // Current coordinate `q`
        let q = p + s * u;

        // Jacobian of `q` with respect to `p`
        // Jq = d(p)/dp + d(s * u)/dp = I + s * Ju
        let Jq = I + s * Ju;

        // Sample noise
        let nx = fbm(q + vec3(0.0, 0.0, 0.0));
        let ny = fbm(q + vec3(27.0, 7.0, 34.0));
        let nz = fbm(q + vec3(42.0, 31.0, 13.0));

        u = vec3(nx.f, ny.f, nz.f);

        // Update Ju, we need the gradient of the new u with respect to p.
        // Chain Rule: grad_p = grad_q * Jq
        let du_dx = nx.df * Jq;
        let du_dy = ny.df * Jq;
        let du_dz = nz.df * Jq;

        // Construct new Jacobian matrix for u
        // Columns are partials wrt X
        Ju = mat3x3<f32>(
            vec3(du_dx.x, du_dy.x, du_dz.x),
            vec3(du_dx.y, du_dy.y, du_dz.y),
            vec3(du_dx.z, du_dy.z, du_dz.z),
        );
    }

    let q = p + s * u;
    let Jq = I + s * Ju;

    let n = fbm(q);

    return Noise(n.f, n.df * Jq);
}

fn fbm(p: vec3<f32>) -> Noise {
    var out = Noise();

    var amplitude = 0.5;
    var frequency = 1.0;

    var q = p;

    for (var i = 0u; i < parameters.octaves; i++) {
        let n = noise(q);
        out.f += amplitude * n.f;
        out.df += amplitude * frequency * n.df;
        q *= parameters.lacunarity;
        amplitude *= parameters.persistence;
        frequency *= parameters.lacunarity;
    }

    return out;
}

fn noise(p: vec3<f32>) -> Noise {
    if (parameters.dim == 2) {
        return noise_2d(p.xy);
    }
    else if (parameters.dim == 3) {
        return noise_3d(p);
    }
    else {
        return Noise();
    }
}

fn noise_2d(p: vec2<f32>) -> Noise {
    if (MODE_VALUE == parameters.mode) {
        return value_noise_2d(p);
    }
    else if (MODE_PERLIN == parameters.mode) {
        return perlin_noise_2d(p);
    }
    else if (MODE_SIMPLEX == parameters.mode) {
        return simplex_noise_2d(p);
    }
    else if (MODE_WORLEY == parameters.mode) {
        return worley_noise_2d(p);
    }
    else {
        return Noise();
    }
}

fn noise_3d(p: vec3<f32>) -> Noise {
    if (MODE_VALUE == parameters.mode) {
        return value_noise_3d(p);
    }
    else if (MODE_PERLIN == parameters.mode) {
        return perlin_noise_3d(p);
    }
    else if (MODE_SIMPLEX == parameters.mode) {
        return simplex_noise_3d(p);
    }
    else if (MODE_WORLEY == parameters.mode) {
        return worley_noise_3d(p);
    }
    else {
        return Noise();
    }
}

fn value_noise_2d(p: vec2<f32>) -> Noise {
    let i = floor(p);
    let f = fract(p);

    let n00 = rand2(i + vec2(0, 0));
    let n10 = rand2(i + vec2(1, 0));
    let n01 = rand2(i + vec2(0, 1));
    let n11 = rand2(i + vec2(1, 1));

    let k = k2(f);
    let dk = dkdt2(f);

    let nx0 = mix(n00, n10, k.x);
    let nx1 = mix(n01, n11, k.x);

    var out: Noise;
    out.f = mix(nx0, nx1, k.y);
    out.df.x = mix(n10 - n00, n11 - n01, k.y) * dk.x;
    out.df.y = mix(n01 - n00, n11 - n10, k.x) * dk.y;
    return out;
}

fn value_noise_3d(p: vec3<f32>) -> Noise {
    let i = floor(p);
    let f = fract(p);

    let k = k3(f);

    let n000 = rand3(i + vec3(0.0, 0.0, 0.0));
    let n100 = rand3(i + vec3(1.0, 0.0, 0.0));
    let n010 = rand3(i + vec3(0.0, 1.0, 0.0));
    let n110 = rand3(i + vec3(1.0, 1.0, 0.0));
    let n001 = rand3(i + vec3(0.0, 0.0, 1.0));
    let n101 = rand3(i + vec3(1.0, 0.0, 1.0));
    let n011 = rand3(i + vec3(0.0, 1.0, 1.0));
    let n111 = rand3(i + vec3(1.0, 1.0, 1.0));

    let nx00 = mix(n000, n100, k.x);
    let nx10 = mix(n010, n110, k.x);
    let nx01 = mix(n001, n101, k.x);
    let nx11 = mix(n011, n111, k.x);
    let nxy0 = mix(nx00, nx10, k.y);
    let nxy1 = mix(nx01, nx11, k.y);
    let nxyz = mix(nxy0, nxy1, k.z);

    var out: Noise;
    out.f = nxyz;

    let dx = mix(
        vec2(n100 - n000, n101 - n001),
        vec2(n110 - n010, n111 - n011),
        k.y
    );
    let dy = mix(
        vec2(n010 - n000, n011 - n001),
        vec2(n110 - n100, n111 - n101),
        k.x
    );
    let dz = mix(
        vec2(n001 - n000, n011 - n010),
        vec2(n101 - n100, n111 - n110),
        k.x
    );

    out.df = dkdt3(f) * mix(
        vec3(dx.x, dy.x, dz.x),
        vec3(dx.y, dy.y, dz.y),
        vec3(k.z, k.z, k.y),
    );

    return out;
}

fn perlin_noise_2d(p: vec2<f32>) -> Noise {
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

    let k = k2(f);

    let nx0 = mix(n00, n10, k.x);
    let nx1 = mix(n01, n11, k.x);
    let nxy = mix(nx0, nx1, k.y);

    // Gradient:
    // Linearly interpolate the random vectors themselves
    // This represents the direction of the flow coming from the corners
    let g_avg = mix(mix(g00, g10, k.x), mix(g01, g11, k.x), k.y);

    // Add the contribution from the curve slopes
    // The change in value due to the easing function sliding between values
    let slope = dkdt2(f) * vec2(mix(n10 - n00, n11 - n01, k.y), (nx1 - nx0));

    var out: Noise;
    out.f = nxy;
    out.df = vec3(g_avg + slope, 0.0);
    return out;
}

fn perlin_noise_3d(p: vec3<f32>) -> Noise {
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

    let k = k3(f);

    let nx00 = mix(n000, n100, k.x);
    let nx10 = mix(n010, n110, k.x);
    let nx01 = mix(n001, n101, k.x);
    let nx11 = mix(n011, n111, k.x);
    let nxy0 = mix(nx00, nx10, k.y);
    let nxy1 = mix(nx01, nx11, k.y);
    let nxyz = mix(nxy0, nxy1, k.z);

    var out: Noise;
    out.f = nxyz;
    return out;
}

fn simplex_noise_2d(p: vec2<f32>) -> Noise {
    const F = 0.5 * (sqrt(3.0) - 1.0);
    const G = (3.0 - sqrt(3.0)) / 6.0;

    let s = vec2(floor(p + (p.x + p.y) * F));
    let i = s - (s.x + s.y) * G;
    let f0 = p - i;

    let v1: vec2<f32> = select(vec2(0.0, 1.0), vec2(1.0, 0.0), f0.x > f0.y);

    let f1 = f0 - v1 + G;
    let f2 = f0 - 1.0 + 2.0 * G;

    let g0 = rand_vec2(s);
    let g1 = rand_vec2(s + v1);
    let g2 = rand_vec2(s + 1.0);

    let r = vec3(dot(f0, f0), dot(f1, f1), dot(f2, f2));
    let m = max(vec3(0.0), 0.5 - r);

    let m2 = m * m;
    let m3 = m2 * m;
    let m4 = m3 * m;

    let grad_dots = vec3(
        dot(g0, f0),
        dot(g1, f1),
        dot(g2, f2),
    );

    const NORMALIZATION_FACTOR = 70.0;
    var out: Noise;

    out.f = NORMALIZATION_FACTOR * dot(m4, grad_dots);

    // gradient = m^4 * g - 8 * m^3 * (g . f) * f

    // Term 1: m^4 * g (linear contribution from the gradient vector)
    let term1 = m4.x * g0 + m4.y * g1 + m4.z * g2;

    // Term 2: 8 * m^3 * (g . f) * f (radial falloff contribution)
    let t2_factors = 8.0 * m3 * grad_dots;
    let term2 = t2_factors.x * f0 + t2_factors.y * f1 + t2_factors.z * f2;

    out.df = vec3(vec2(NORMALIZATION_FACTOR * (term1 - term2)), 0.0);

    return out;
}

fn simplex_noise_3d(p: vec3<f32>) -> Noise {
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
    let m3 = m2 * m;
    let m4 = m3 * m;

    let grad_dots = vec4(
        dot(g0, f0),
        dot(g1, f1),
        dot(g2, f2),
        dot(g3, f3)
    );

    const NORMALIZATION_FACTOR = 32.0;
    var out: Noise;

    out.f = NORMALIZATION_FACTOR * dot(m4, grad_dots);

    // Gradient: m^4 * g - 8 * m^3 * (g . f) * f
    // Term 1: m^4 * g (Linear contribution)
    let term1 = m4.x * g0 + m4.y * g1 + m4.z * g2 + m4.w * g3;

    // Term 2: 8 * m^3 * (g . f) * f (Radial falloff contribution)
    let t2_factors = 8.0 * m3 * grad_dots;
    let term2 = t2_factors.x * f0 + t2_factors.y * f1 + t2_factors.z * f2 + t2_factors.w * f3;

    out.df = NORMALIZATION_FACTOR * (term1 - term2);

    return out;
}

fn worley_noise_2d(p: vec2<f32>) -> Noise {
    let i = floor(p);
    let f = fract(p);

    let d_max = 2.0;
    var d_min = d_max;

    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let neighbor = vec2<f32>(f32(x), f32(y));
            let cell_id = i + neighbor;
            let point_local = rand_vec2(cell_id) * 0.5 + 0.5;
            let diff = neighbor + point_local - f;
            d_min = min(d_min, dot(diff, diff));
        }
    }

    var out: Noise;
    out.f = sqrt(d_min) / sqrt(d_max);
    return out;
}

fn worley_noise_3d(p: vec3<f32>) -> Noise {
    let i = floor(p);
    let f = fract(p);

    let d_max = 3.0;
    var d_min = d_max;

    for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let neighbor = vec3<f32>(f32(x), f32(y), f32(z));
                let cell_id = i + neighbor;
                let point_local = rand_vec3(cell_id) * 0.5 + 0.5;
                let diff = neighbor + point_local - f;
                d_min = min(d_min, dot(diff, diff));
            }
        }
    }

    var out: Noise;
    out.f = sqrt(d_min) / sqrt(d_max);
    return out;
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

fn k2(t: vec2<f32>) -> vec2<f32> {
    let t2 = t * t;
    let t3 = t * t2;
    return 3 * t2 - 2 * t3;
}

fn k3(t: vec3<f32>) -> vec3<f32> {
    let t2 = t * t;
    let t3 = t * t2;
    return 3 * t2 - 2 * t3;
}

fn dkdt2(t: vec2<f32>) -> vec2<f32> {
    // d/dt (3t^2 - 2t^3) = 6t(1-t)
    return 6.0 * t * (1.0 - t);
}

fn dkdt3(t: vec3<f32>) -> vec3<f32> {
    // d/dt (3t^2 - 2t^3) = 6t(1-t)
    return 6.0 * t * (1.0 - t);
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
