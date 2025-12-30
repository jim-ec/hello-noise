struct Uniforms {
    t: f32,
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
    out.uv = vec2<f32>(id);
    out.position = vec4(vec2<f32>(vec2<i32>(id << vec2(1)) - 1), 0.0, 1.0);
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    var color: vec3<f32>;

    color.r = in.uv.x;
    color.g = in.uv.y;
    color.b = cos(uniforms.t) * 0.5 + 0.5;

    return vec4(pow(color, vec3(2.2)), 1.0);
}
