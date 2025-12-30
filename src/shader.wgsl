struct Uniforms {
    aspect_ratio: f32,
    time: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vertex(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Bit patterns of quad vertices:
    // A - B
    // | / |
    // C - D
    // Tri #0: A C B => X: 0 0 1 | 1 0 1 => 0b101100 => 0x2c
    // Tri #1: B C D => Y: 0 1 0 | 0 1 1 => 0b110010 => 0x32
    let id = (vec2(0x2cu, 0x32u) >> vec2(vertex_index % 6u)) & vec2(1u);

    var out: VertexOutput;
    out.uv = vec2<f32>(id) * vec2(uniforms.aspect_ratio, 1.0);
    out.position = vec4(vec2<f32>(vec2<i32>(id << vec2(1)) - 1), 0.0, 1.0);
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    var color: vec3<f32>;

    color.r = in.uv.x;
    color.g = in.uv.y;
    color.b = cos(uniforms.time) * 0.5 + 0.5;

    const ZOOM: f32 = 50.0;
    color = vec3(value_noise(ZOOM * in.uv));

    return vec4(pow(color, vec3(2.2)), 1.0);
}

fn value_noise(p: vec2<f32>) -> f32 {
    let x0 = floor(p.x);
    let y0 = floor(p.y);
    let x1 = x0 + 1.0;
    let y1 = y0 + 1.0;

    let r00 = rand(vec2(x0, y0));
    let r01 = rand(vec2(x0, y1));
    let r10 = rand(vec2(x1, y0));
    let r11 = rand(vec2(x1, y1));

    let kx = p.x - x0;
    let ky = p.y - y0;
    let ry0 = mix(r00, r10, kx);
    let ry1 = mix(r01, r11, kx);
    let rx = mix(ry0, ry1, ky);

    return rx;
}

fn rand(p: vec2<f32>) -> f32 {
    const SEED: f32 = 42.0;
    let input = p + vec2(SEED);
    return fract(sin(dot(input, vec2(12.9898, 78.233))) * 43758.5453123);
}
