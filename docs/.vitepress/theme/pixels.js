// A pixel field over the hero sky: the sky is redrawn as a grid of tiny cells, and a height-field wave simulation,
// one texel per cell, shifts, flickers and posterizes the cells the pointer disturbs. At rest the grid matches the
// CSS background, so the swap is invisible; the CSS grain stays on top.
// Without WebGL2 float render targets, or with reduced motion, the static CSS hero is left untouched.

const QUAD = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);

const VERT = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main() {
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

// one step of the wave equation: r = height, g = velocity
const STEP = `#version 300 es
precision highp float;
uniform sampler2D u_state;
uniform vec2 u_texel;
uniform float u_damping;
in vec2 v_uv;
out vec4 o;
void main() {
  vec4 s = texture(u_state, v_uv);
  float avg = (
    texture(u_state, v_uv + vec2(u_texel.x, 0.0)).r +
    texture(u_state, v_uv - vec2(u_texel.x, 0.0)).r +
    texture(u_state, v_uv + vec2(0.0, u_texel.y)).r +
    texture(u_state, v_uv - vec2(0.0, u_texel.y)).r
  ) * 0.25;
  s.g += (avg - s.r) * 2.0;
  s.g *= u_damping;
  s.r += s.g;
  s.r *= 0.9995;
  o = s;
}`;

// a smooth cosine-shaped drop
const DROP = `#version 300 es
precision highp float;
uniform sampler2D u_state;
uniform vec2 u_center;
uniform float u_radius;
uniform float u_strength;
uniform float u_aspect;
in vec2 v_uv;
out vec4 o;
void main() {
  vec4 s = texture(u_state, v_uv);
  vec2 d = (v_uv - u_center) * vec2(u_aspect, 1.0);
  float k = max(0.0, 1.0 - length(d) / u_radius);
  s.r += (0.5 - cos(k * 3.14159265) * 0.5) * u_strength;
  o = s;
}`;

// the sky of .tw-hero, sampled once per cell and disturbed by the wave height
const RENDER = `#version 300 es
precision highp float;
uniform sampler2D u_state;
uniform vec2 u_texel;
uniform vec2 u_size;
uniform float u_cell;
uniform float u_time;
in vec2 v_uv;
out vec4 o;

vec3 hex(float r, float g, float b) { return vec3(r, g, b) / 255.0; }

vec3 sky(vec2 p) {
  // linear-gradient(172deg, ...) in CSS pixels, y pointing down
  float a = radians(172.0);
  vec2 dir = vec2(sin(a), -cos(a));
  float len = abs(u_size.x * dir.x) + abs(u_size.y * dir.y);
  float t = clamp(dot(p - 0.5 * u_size, dir) / len + 0.5, 0.0, 1.0);
  vec3 c0 = hex(11.0, 26.0, 51.0), c1 = hex(27.0, 58.0, 107.0), c2 = hex(63.0, 124.0, 196.0);
  vec3 c3 = hex(212.0, 87.0, 124.0), c4 = hex(242.0, 124.0, 95.0), c5 = hex(248.0, 181.0, 108.0);
  vec3 c6 = hex(253.0, 228.0, 195.0);
  vec3 col = mix(c0, c1, clamp(t / 0.22, 0.0, 1.0));
  col = mix(col, c2, clamp((t - 0.22) / 0.22, 0.0, 1.0));
  col = mix(col, c3, clamp((t - 0.44) / 0.20, 0.0, 1.0));
  col = mix(col, c4, clamp((t - 0.64) / 0.16, 0.0, 1.0));
  col = mix(col, c5, clamp((t - 0.80) / 0.13, 0.0, 1.0));
  col = mix(col, c6, clamp((t - 0.93) / 0.07, 0.0, 1.0));

  // the setting sun: .tw-hero::before, a radial gradient in a circle below the bottom edge
  float D = 0.62 * u_size.x;
  vec2 center = vec2(0.5 * u_size.x, 1.62 * u_size.y - 0.5 * D);
  float d = length(p - center) / (0.5 * D * 1.41421356);
  vec4 s0 = vec4(hex(255.0, 233.0, 200.0) * 0.8, 0.8);
  vec4 s1 = vec4(hex(255.0, 201.0, 133.0) * 0.4, 0.4);
  vec4 s2 = vec4(hex(242.0, 124.0, 95.0) * 0.149, 0.149);
  vec4 sun = mix(s0, s1, clamp(d / 0.24, 0.0, 1.0));
  sun = mix(sun, s2, clamp((d - 0.24) / 0.22, 0.0, 1.0));
  sun = mix(sun, vec4(0.0), clamp((d - 0.46) / 0.22, 0.0, 1.0));
  return sun.rgb + col * (1.0 - sun.a);
}

float height(vec2 uv) { return texture(u_state, uv).r; }

float hash(vec2 q) { return fract(sin(dot(q, vec2(127.1, 311.7))) * 43758.5453); }

void main() {
  vec2 p = vec2(v_uv.x, 1.0 - v_uv.y) * u_size;          // CSS pixels, y down
  vec2 cell = floor(p / u_cell);
  vec2 c = (cell + 0.5) * u_cell;
  vec2 uv = vec2(c.x / u_size.x, 1.0 - c.y / u_size.y);
  float h = height(uv);
  float hx = height(uv + vec2(u_texel.x, 0.0)) - height(uv - vec2(u_texel.x, 0.0));
  float hy = height(uv + vec2(0.0, u_texel.y)) - height(uv - vec2(0.0, u_texel.y));
  float e = clamp(abs(h) * 14.0 + length(vec2(hx, hy)) * 10.0, 0.0, 1.0);

  // the slope moves each cell by whole cells, so the grid tears instead of bending
  vec2 shift = floor(vec2(-hx, hy) * 70.0 + 0.5) * u_cell;
  vec3 col = sky(c + shift);
  col *= 1.0 + h * 5.0;                                   // crests lighter, troughs darker

  // disturbed cells drop to a few color levels and flicker
  vec3 poster = floor(col * 6.0 + 0.5) / 6.0;
  col = mix(col, poster, smoothstep(0.15, 0.6, e));
  col += (hash(cell + floor(u_time * 14.0)) - 0.5) * 0.22 * e;

  // a faint seam between disturbed cells, and a still dither everywhere
  vec2 f = fract(p / u_cell);
  float seam = step(f.x, 1.0 / u_cell) + step(f.y, 1.0 / u_cell);
  col *= 1.0 - min(seam, 1.0) * 0.12 * e;
  col += (hash(cell) - 0.5) * 0.018;
  o = vec4(col, 1.0);
}`;

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}

function program(gl, frag) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, VERT));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, frag));
  gl.bindAttribLocation(p, 0, "a_pos");
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const uniforms = {};
  for (let i = 0; i < gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i++) {
    const name = gl.getActiveUniform(p, i).name;
    uniforms[name] = gl.getUniformLocation(p, name);
  }
  return { p, u: uniforms };
}

function pixels(hero) {
  if (hero.querySelector(".tw-pixels")) return;
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;

  const canvas = document.createElement("canvas");
  canvas.className = "tw-pixels";
  canvas.setAttribute("aria-hidden", "true");
  const gl = canvas.getContext("webgl2", { alpha: false, antialias: false, premultipliedAlpha: false });
  if (!gl || !gl.getExtension("EXT_color_buffer_float")) return;

  let step, drop, render;
  try {
    step = program(gl, STEP);
    drop = program(gl, DROP);
    render = program(gl, RENDER);
  } catch (e) {
    return;
  }

  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, QUAD, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

  let W = 0, H = 0, simW = 0, simH = 0, cell = 3;
  let targets = [];

  function target(w, h) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    return { tex, fb };
  }

  function resize() {
    const rect = hero.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = rect.width;
    H = rect.height;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    cell = Math.max(3, Math.ceil(W / 520));                // one simulation texel per cell
    const w = Math.max(32, Math.ceil(W / cell)), h = Math.max(32, Math.ceil(H / cell));
    if (w !== simW || h !== simH) {
      targets.forEach(({ tex, fb }) => (gl.deleteTexture(tex), gl.deleteFramebuffer(fb)));
      simW = w;
      simH = h;
      targets = [target(w, h), target(w, h)];
    }
    draw();
  }

  function pass(prog, setup) {
    const [src, dst] = targets;
    gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fb);
    gl.viewport(0, 0, simW, simH);
    gl.useProgram(prog.p);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, src.tex);
    gl.uniform1i(prog.u.u_state, 0);
    setup(prog.u);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    targets = [dst, src];
  }

  function draw() {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.useProgram(render.p);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, targets[0].tex);
    gl.uniform1i(render.u.u_state, 0);
    gl.uniform2f(render.u.u_texel, 1 / simW, 1 / simH);
    gl.uniform2f(render.u.u_size, W, H);
    gl.uniform1f(render.u.u_cell, cell);
    gl.uniform1f(render.u.u_time, performance.now() / 1000);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  const drops = [];
  let last = null, frame = 0, quietUntil = 0;

  function addDrop(x, y, strength) {
    drops.push([x / W, 1 - y / H, strength]);
    quietUntil = performance.now() + 7000;
    if (!frame) frame = requestAnimationFrame(tick);
  }

  function tick(now) {
    frame = 0;
    if (!hero.isConnected) return;
    const radius = 22 / H;
    for (const [x, y, s] of drops.splice(0)) {
      pass(drop, (u) => {
        gl.uniform2f(u.u_center, x, y);
        gl.uniform1f(u.u_radius, radius);
        gl.uniform1f(u.u_strength, s);
        gl.uniform1f(u.u_aspect, W / H);
      });
    }
    for (let i = 0; i < 2; i++) {
      pass(step, (u) => {
        gl.uniform2f(u.u_texel, 1 / simW, 1 / simH);
        gl.uniform1f(u.u_damping, 0.988);
      });
    }
    draw();
    if (now < quietUntil) frame = requestAnimationFrame(tick);
  }

  hero.addEventListener("pointermove", (e) => {
    const rect = hero.getBoundingClientRect();
    const x = e.clientX - rect.left, y = e.clientY - rect.top;
    if (last) {
      // a trail of drops along the path, stronger when the pointer moves faster
      const dx = x - last[0], dy = y - last[1], dist = Math.hypot(dx, dy);
      const n = Math.min(12, Math.ceil(dist / 10));
      const strength = Math.min(0.045, 0.012 + dist * 0.0012);
      for (let i = 1; i <= n; i++) addDrop(last[0] + (dx * i) / n, last[1] + (dy * i) / n, strength / Math.sqrt(n));
    } else {
      addDrop(x, y, 0.03);
    }
    last = [x, y];
  });
  hero.addEventListener("pointerleave", () => (last = null));
  hero.addEventListener("pointerdown", (e) => {
    const rect = hero.getBoundingClientRect();
    addDrop(e.clientX - rect.left, e.clientY - rect.top, 0.14);
  });

  hero.prepend(canvas);
  new ResizeObserver(resize).observe(hero);
  resize();
  hero.classList.add("tw-hero--pixels");
}

export function initPixels() {
  document.querySelectorAll(".tw-hero").forEach(pixels);
}
