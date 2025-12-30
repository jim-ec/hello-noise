struct Uniforms {
    aspect_ratio: f32,
    time: f32,
    zoom: f32,
    mode: u32,
}

var<push_constant> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

const MODE_UV: u32 = 0;
const MODE_VALUE: u32 = 1;
const MODE_PERLIN: u32 = 2;

const TAU: f32 = 6.2831853072;

@vertex
fn vertex(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Bit patterns of quad vertices:
    // A - B
    // | / |
    // C - D
    // Tri #0: A C B => X: 0 0 1 | 1 0 1 => 0b101100 => 0x2c
    // Tri #1: B C D => Y: 0 1 0 | 0 1 1 => 0b110010 => 0x32
    let id = (vec2(0x2cu, 0x32u) >> vec2(vertex_index % 6u)) & vec2(1u);
    let uv = vec2<f32>(vec2<i32>(id << vec2(1)) - 1);

    var out: VertexOutput;
    out.uv = vec2<f32>(exp(uniforms.zoom) * uv) * vec2(uniforms.aspect_ratio, 1.0);
    out.position = vec4(uv, 0.0, 1.0);
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let noise = noise(uniforms.mode, uv);
    let color = color(uniforms.mode, noise, uv);
    return vec4(pow(color, vec3(2.2)), 1.0);
}

fn color(mode: u32, noise: f32, uv: vec2<f32>) -> vec3<f32> {
    if (MODE_UV == uniforms.mode) {
        return vec3(fract(uv), 0.0);
    }
    else {
        return vec3(noise) * 0.5 + 0.5;
    }
}

fn noise(mode: u32, uv: vec2<f32>) -> f32 {
    if (MODE_VALUE == uniforms.mode) {
        return value_noise(uv);
    }
    else if (MODE_PERLIN == uniforms.mode) {
        return perlin_noise(uv);
    }
    else {
        return 0.0;
    }
}

fn perlin_noise(p: vec2<f32>) -> f32 {
    let x0 = floor(p.x);
    let y0 = floor(p.y);
    let x1 = x0 + 1.0;
    let y1 = y0 + 1.0;
    let u = p.x - x0;
    let v = p.y - y0;

    let g00 = unit_vector(rand(vec2(x0, y0)) * TAU);
    let g01 = unit_vector(rand(vec2(x0, y1)) * TAU);
    let g10 = unit_vector(rand(vec2(x1, y0)) * TAU);
    let g11 = unit_vector(rand(vec2(x1, y1)) * TAU);

    let n00 = dot(g00, vec2(u, v));
    let n01 = dot(g01, vec2(u, v - 1));
    let n10 = dot(g10, vec2(u - 1, v));
    let n11 = dot(g11, vec2(u - 1, v - 1));

    let nx0 = mix(n00, n10, f(u));
    let nx1 = mix(n01, n11, f(u));
    let nxy = mix(nx0, nx1, f(v));

    return nxy;
}

fn unit_vector(angle: f32) -> vec2<f32> {
    return vec2(cos(angle), sin(angle));
}

fn f(t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t * t2;
    let t4 = t * t2;
    let t5 = t * t2;

    // return 6 * t5 - 15 * t4 + 10 * t3;
    return 3 * t2 - 2 * t3;
}

fn value_noise(p: vec2<f32>) -> f32 {
    let x0 = floor(p.x);
    let y0 = floor(p.y);
    let x1 = x0 + 1.0;
    let y1 = y0 + 1.0;
    let u = p.x - x0;
    let v = p.y - y0;

    let n00 = rand(vec2(x0, y0));
    let n01 = rand(vec2(x0, y1));
    let n10 = rand(vec2(x1, y0));
    let n11 = rand(vec2(x1, y1));

    let nx0 = mix(n00, n10, f(u));
    let nx1 = mix(n01, n11, f(u));
    let nxy = mix(nx0, nx1, f(v));

    return nxy;
}

/// Returns random values in [-1, 1).
fn rand(p: vec2<f32>) -> f32 {
    const SEED: f32 = 42.0;
    let input = p + vec2(SEED);
    let unit = fract(sin(dot(input, vec2(12.9898, 78.233))) * 43758.5453123);
    return unit * 2.0 - 1.0;
}
