#!/usr/bin/env python3
"""Ray-trace the prototype's scene with Blender Cycles from the JSON written by export.js.

    blender -b -P dev/raytrace/render.py -- scene.json out_dir [options]
    python3 dev/raytrace/render.py scene.json out_dir [options]      (with the `bpy` module installed)

Options:
    --device gpu|cpu     gpu uses OptiX (hardware ray tracing) and falls back to CUDA, then CPU. Default gpu.
    --samples N          paths per pixel (default 192)
    --size WxH           picture size (default 1080x1920)
    --views a,b,...      which cameras: game (the game's own camera), hero, street, quay, aerial, top (default game,hero)
    --exposure E         stops added to every picture (default 0)
    --sun-from deg       compass direction the sun shines FROM (0 = north, 90 = east); default: as exported
    --sun-height deg     sun height above the horizon; default: as exported
    --frames N           with one view: render N frames of a slow move toward that view, as frame_0001.png ...

This is an OFFLINE render of the same geometry and colours the game draws: seconds a picture on a
graphics card, minutes on a CPU, not the game running. Nothing here is used by the game.
The light rig below is this script's own: it takes the sun's direction and daytime colour from the
export and sets every strength itself, so a change to the game's lighting.js does not show here.
"""
import base64
import json
import math
import os
import sys
import tempfile
import time

import bpy
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]


def opt(name, default):
    return argv[argv.index(name) + 1] if name in argv else default


SCENE_JSON, OUT = argv[0], argv[1]
DEVICE = opt("--device", "gpu").lower()
SAMPLES = int(opt("--samples", "192"))
W, H = (int(v) for v in opt("--size", "1080x1920").split("x"))
VIEWS = opt("--views", "game,hero").split(",")
EXPOSURE = float(opt("--exposure", "0"))
FRAMES = int(opt("--frames", "0"))
os.makedirs(OUT, exist_ok=True)
t0 = time.time()


def log(*a):
    print("[render]", *a, flush=True)


def f32(s, width):
    return np.frombuffer(base64.b64decode(s), dtype=np.float32).reshape(-1, width)


data = json.load(open(SCENE_JSON))
log("scene", SCENE_JSON, "meshes", len(data["meshes"]), "dusk", round(data["dusk"], 2), "blender", bpy.app.version_string)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
tmp = tempfile.mkdtemp(prefix="nh-rt-")

# ---- textures -------------------------------------------------------------
images = []
for i, t in enumerate(data["textures"]):
    p = os.path.join(tmp, f"tex{i}.png")
    open(p, "wb").write(base64.b64decode(t["png"]))
    img = bpy.data.images.load(p)
    img.colorspace_settings.name = "sRGB" if t["srgb"] else "Non-Color"
    images.append(img)


# ---- materials ------------------------------------------------------------
def sock(node, *names):
    for n in names:
        if n in node.inputs:
            return node.inputs[n]
    raise KeyError(names)


def make_material(i, m):
    mat = bpy.data.materials.new(f"m{i}_{m['name']}")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])

    def image_node(index):
        t = data["textures"][index]
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = images[index]
        tex.extension = "REPEAT" if t["wrap"] else "EXTEND"
        uv = nt.nodes.new("ShaderNodeUVMap")
        mapping = nt.nodes.new("ShaderNodeMapping")
        mapping.inputs["Scale"].default_value = (t["repeat"][0], t["repeat"][1], 1)
        mapping.inputs["Location"].default_value = (t["offset"][0], t["offset"][1], 0)
        nt.links.new(uv.outputs[0], mapping.inputs["Vector"])
        nt.links.new(mapping.outputs[0], tex.inputs["Vector"])
        return tex

    def multiply(a_socket, b_socket):
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        mix.inputs[0].default_value = 1.0
        nt.links.new(a_socket, mix.inputs[6])
        nt.links.new(b_socket, mix.inputs[7])
        return mix.outputs[2]

    rgb = nt.nodes.new("ShaderNodeRGB")
    rgb.outputs[0].default_value = (*m["color"], 1)
    color = rgb.outputs[0]
    alpha = None
    if m["map"] >= 0:
        tex = image_node(m["map"])
        color = multiply(color, tex.outputs["Color"])
        if m["transparent"] or m["alphaTest"] > 0:
            alpha = tex.outputs["Alpha"]
    if m["vertexColors"]:
        vc = nt.nodes.new("ShaderNodeVertexColor")
        vc.layer_name = "Col"
        color = multiply(color, vc.outputs["Color"])
    if m["unlit"]:  # drawn without lighting in the game (signs, markers): make it its own light source
        nt.links.new(color, sock(bsdf, "Emission Color", "Emission"))
        sock(bsdf, "Emission Strength").default_value = 1.0
        sock(bsdf, "Base Color").default_value = (0, 0, 0, 1)
    else:
        nt.links.new(color, sock(bsdf, "Base Color"))
        e = m["emissive"]
        if max(e) > 1e-4:
            ergb = nt.nodes.new("ShaderNodeRGB")
            ergb.outputs[0].default_value = (*[min(1.0, c / max(max(e), 1e-6)) for c in e], 1)
            ecol = ergb.outputs[0]
            if m["emissiveMap"] >= 0:
                ecol = multiply(ecol, image_node(m["emissiveMap"]).outputs["Color"])
            nt.links.new(ecol, sock(bsdf, "Emission Color", "Emission"))
            sock(bsdf, "Emission Strength").default_value = max(e) * 2.0
    sock(bsdf, "Roughness").default_value = float(m["rough"])
    sock(bsdf, "Metallic").default_value = float(m["metal"])
    if m["roughnessMap"] >= 0:
        sep = nt.nodes.new("ShaderNodeSeparateColor")
        nt.links.new(image_node(m["roughnessMap"]).outputs["Color"], sep.inputs[0])
        nt.links.new(sep.outputs[1], sock(bsdf, "Roughness"))
    if m["bumpMap"] >= 0 and m["bumpScale"]:
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = min(1.0, abs(m["bumpScale"]) * 4)
        nt.links.new(image_node(m["bumpMap"]).outputs["Color"], bump.inputs["Height"])
        nt.links.new(bump.outputs[0], sock(bsdf, "Normal"))
    if m["transparent"]:
        if alpha is not None and m["opacity"] < 0.999:
            mul = nt.nodes.new("ShaderNodeMath")
            mul.operation = "MULTIPLY"
            mul.inputs[1].default_value = m["opacity"]
            nt.links.new(alpha, mul.inputs[0])
            alpha = mul.outputs[0]
        if alpha is not None:
            nt.links.new(alpha, sock(bsdf, "Alpha"))
        else:
            sock(bsdf, "Alpha").default_value = m["opacity"]
    elif alpha is not None:
        nt.links.new(alpha, sock(bsdf, "Alpha"))
    mat.use_backface_culling = False
    return mat


materials = [make_material(i, m) for i, m in enumerate(data["materials"])]

# ---- meshes ---------------------------------------------------------------
tris = 0
for i, g in enumerate(data["meshes"]):
    pos, nor, uv = f32(g["pos"], 3), f32(g["nor"], 3), f32(g["uv"], 2)
    idx = np.frombuffer(base64.b64decode(g["idx"]), dtype=np.uint32).astype(np.int32)
    n_tri = len(idx) // 3
    me = bpy.data.meshes.new(f"g{i}")
    me.vertices.add(len(pos))
    me.vertices.foreach_set("co", pos.ravel())
    me.loops.add(len(idx))
    me.loops.foreach_set("vertex_index", idx)
    me.polygons.add(n_tri)
    me.polygons.foreach_set("loop_start", np.arange(0, len(idx), 3, dtype=np.int32))
    me.polygons.foreach_set("loop_total", np.full(n_tri, 3, dtype=np.int32))
    me.uv_layers.new(name="UVMap").data.foreach_set("uv", uv[idx].ravel())
    if g["col"]:
        col = f32(g["col"], 3)
        rgba = np.concatenate([col[idx], np.ones((len(idx), 1), dtype=np.float32)], axis=1)
        me.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="CORNER").data.foreach_set("color", rgba.ravel())
    me.update(calc_edges=True)
    me.validate(clean_customdata=False)
    try:  # keep the game's own shading normals; a mesh that refuses them is simply shaded flat
        if not data["materials"][g["material"]]["flat"] and len(me.loops) == len(idx):
            me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), dtype=bool))
            me.normals_split_custom_set(nor[idx].tolist())
    except Exception as exc:  # noqa: BLE001
        log("normals skipped for mesh", i, type(exc).__name__, exc)
    me.materials.append(materials[g["material"]])
    scene.collection.objects.link(bpy.data.objects.new(f"g{i}", me))
    tris += n_tri
log("built", tris, "triangles in", round(time.time() - t0, 1), "s")

# ---- light ----------------------------------------------------------------
tod = data["tod"]
sun_l = next((l for l in data["lights"] if l["type"] == "DirectionalLight"), None)
hemi = next((l for l in data["lights"] if l["type"] == "HemisphereLight"), None)
toward = Vector(sun_l["toward"]) if sun_l else Vector((-0.5, -0.5, 0.7))
if "--sun-from" in argv or "--sun-height" in argv:
    az = math.radians(float(opt("--sun-from", "235")))
    el = math.radians(float(opt("--sun-height", "30")))
    toward = Vector((math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)))
toward.normalize()
night, warm = tod["night"], tod["warm"]
day_color = sun_l["color"] if sun_l else (1, 0.9, 0.8)
sun = bpy.data.lights.new("Sun", "SUN")
sun.energy = 0.10 + 6.4 * tod["sun"]             # watts per square metre: strong by day, a moon's worth at night
sun.color = [d * (1 - night) + m * night for d, m in zip(day_color, (0.55, 0.66, 1.0))]   # sunlight by day, moonlight at night
sun.angle = math.radians(1.6)                     # a slightly soft sun gives readable penumbras
sun_o = bpy.data.objects.new("Sun", sun)
sun_o.rotation_euler = toward.to_track_quat("Z", "Y").to_euler()
scene.collection.objects.link(sun_o)

world = bpy.data.worlds.new("Sky")
world.use_nodes = True
scene.world = world
wn = world.node_tree
wn.nodes.clear()
w_out = wn.nodes.new("ShaderNodeOutputWorld")
bg = wn.nodes.new("ShaderNodeBackground")
grad = wn.nodes.new("ShaderNodeValToRGB")       # horizon colour blending up to the zenith colour
tc = wn.nodes.new("ShaderNodeTexCoord")
sepz = wn.nodes.new("ShaderNodeSeparateXYZ")
wn.links.new(tc.outputs["Generated"], sepz.inputs[0])
wn.links.new(sepz.outputs["Z"], grad.inputs["Fac"])
day_zenith = (0.20, 0.42, 0.90)
horizon = [(1 - warm) * h0 + warm * h1 for h0, h1 in zip((0.72, 0.84, 1.0), (1.0, 0.66, 0.38))]
zenith = [(1 - 0.35 * warm) * c for c in day_zenith]
dark = 1 - 0.965 * night                          # the night sky keeps a little deep blue, never black
grad.color_ramp.elements[0].position = 0.0        # the horizon
grad.color_ramp.elements[0].color = (*[c * dark + n * night for c, n in zip(horizon, (0.010, 0.016, 0.040))], 1)
grad.color_ramp.elements[1].position = 0.55       # well up the sky
grad.color_ramp.elements[1].color = (*[c * dark + n * night for c, n in zip(zenith, (0.004, 0.008, 0.026))], 1)
bg.inputs["Strength"].default_value = 0.75
wn.links.new(grad.outputs["Color"], bg.inputs["Color"])
wn.links.new(bg.outputs[0], w_out.inputs["Surface"])

if night > 0.02:  # street lamps become real lights
    for k, p in enumerate(data["lamps"]):
        lamp = bpy.data.lights.new(f"Lamp{k}", "POINT")
        lamp.energy = 950 * night
        lamp.color = (1.0, 0.62, 0.30)
        lamp.shadow_soft_size = 0.18
        lo = bpy.data.objects.new(f"Lamp{k}", lamp)
        lo.location = (p[0], p[1], p[2] - 0.35)
        scene.collection.objects.link(lo)

# ---- cameras --------------------------------------------------------------
focus = Vector(data["focus"])


def look_at(name, eye, target, lens=35, fstop=None):
    cam = bpy.data.cameras.new(name)
    cam.lens = lens
    cam.clip_start, cam.clip_end = 0.2, 2000
    if fstop:
        cam.dof.use_dof = True
        cam.dof.focus_distance = (Vector(target) - Vector(eye)).length
        cam.dof.aperture_fstop = fstop
    o = bpy.data.objects.new(name, cam)
    o.location = eye
    o.rotation_euler = (Vector(target) - Vector(eye)).to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(o)
    return o


def game_camera():
    c = data["camera"]
    cam = bpy.data.cameras.new("game")
    cam.sensor_fit = "VERTICAL"
    cam.angle_y = math.radians(c["fovY"])
    cam.clip_start, cam.clip_end = 0.5, 3000
    o = bpy.data.objects.new("game", cam)
    m = c["matrix"]
    o.matrix_world = Matrix([[m[0], m[4], m[8], m[12]], [m[1], m[5], m[9], m[13]], [m[2], m[6], m[10], m[14]], [0, 0, 0, 1]])
    scene.collection.objects.link(o)
    return o


drop, pick, water_y = Vector(data["drop"]), Vector(data["pick"]), data["block"]["waterY"]
CAMERAS = {
    "game": game_camera,
    # eye level, a few steps south-east of whoever the game camera follows, looking back at them
    "hero": lambda: look_at("hero", focus + Vector((5.5, -6.0, 0.7)), focus + Vector((-1.0, 2.0, -0.3)), lens=35, fstop=2.8),
    # eye level along an east-west street: export the scene with the player standing on one
    "street": lambda: look_at("street", focus + Vector((-10.0, -0.8, 0.55)), focus + Vector((8.0, 0.4, 0.1)), lens=30, fstop=4.0),
    # from a boat's height on the harbour, looking north at the quay and the boat shed
    "quay": lambda: look_at("quay", Vector((pick.x + 7.0, water_y - 11.0, 2.4)), Vector((pick.x + 1.0, water_y + 6.0, 1.6)), lens=32, fstop=5.6),
    # the whole block from over the harbour
    "aerial": lambda: look_at("aerial", Vector((46.0, water_y - 42.0, 40.0)), Vector((2.0, 4.0, 0.0)), lens=32),
    # straight down, for checking the export against the game's own map
    "top": lambda: look_at("top", focus + Vector((0, -0.01, 85)), focus, lens=50),
}

# ---- render settings ------------------------------------------------------
scene.render.engine = "CYCLES"
cy = scene.cycles
cy.samples = SAMPLES
cy.use_adaptive_sampling = True
cy.adaptive_threshold = 0.02
cy.use_denoising = True
cy.max_bounces = 8
used = "CPU"
if DEVICE == "gpu":
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for kind in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = kind
                prefs.get_devices()
                gpus = [d for d in prefs.devices if d.type == kind]
                if gpus:
                    for d in prefs.devices:
                        d.use = d.type == kind
                    cy.device = "GPU"
                    used = kind + ": " + ", ".join(d.name for d in gpus)
                    break
            except Exception as exc:  # noqa: BLE001
                log(kind, "not usable:", type(exc).__name__, exc)
    except Exception as exc:  # noqa: BLE001
        log("no GPU preferences:", type(exc).__name__, exc)
if used == "CPU":
    cy.device = "CPU"
log("rendering on", used)
scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = W, H, 100
scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent = False
try:
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
except Exception as exc:  # noqa: BLE001
    log("view transform left at default:", exc)
scene.view_settings.exposure = EXPOSURE + (1.1 * night)   # open up the night a little: readable deep blue, as the game aims for


def stats(path):  # numbers to judge exposure by, without having to look at the picture
    try:
        img = bpy.data.images.load(path)
        px = np.array(img.pixels[:], dtype=np.float32).reshape(-1, 4)[:, :3]
        lum = px @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        q = np.quantile(lum, [0.05, 0.5, 0.95])
        return {"mean": round(float(lum.mean()), 3), "p05": round(float(q[0]), 3), "p50": round(float(q[1]), 3), "p95": round(float(q[2]), 3),
                "black": round(float((lum < 0.02).mean()), 3), "white": round(float((lum > 0.98).mean()), 3)}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


report = []
for name in VIEWS:
    if name not in CAMERAS:
        log("unknown view", name)
        continue
    cam = CAMERAS[name]()
    scene.camera = cam
    if FRAMES > 1:  # a slow push in: start 18 % further back and a little higher, end at the view itself
        end = cam.location.copy()
        target = focus if name != "quay" else pick
        for f in range(FRAMES):
            k = f / (FRAMES - 1)
            ease = k * k * (3 - 2 * k)
            cam.location = end + (end - target) * (0.18 * (1 - ease)) + Vector((0, 0, 1.5 * (1 - ease)))
            if name != "game":
                cam.rotation_euler = (target + Vector((0, 0, 0.6)) - cam.location).to_track_quat("-Z", "Y").to_euler()
            scene.render.filepath = os.path.join(OUT, f"{name}_{f + 1:04d}.png")
            t1 = time.time()
            bpy.ops.render.render(write_still=True)
            if f % 10 == 0 or f == FRAMES - 1:
                log("frame", f + 1, "of", FRAMES, round(time.time() - t1, 1), "s")
        report.append({"view": name, "frames": FRAMES})
    else:
        scene.render.filepath = os.path.join(OUT, f"{name}.png")
        t1 = time.time()
        bpy.ops.render.render(write_still=True)
        r = {"view": name, "file": scene.render.filepath, "seconds": round(time.time() - t1, 1)}
        r.update(stats(scene.render.filepath))
        report.append(r)
        log(json.dumps(r))
json.dump({"device": used, "samples": SAMPLES, "size": [W, H], "dusk": data["dusk"], "views": report, "total_seconds": round(time.time() - t0, 1)},
          open(os.path.join(OUT, "report.json"), "w"), indent=1)
log("done in", round(time.time() - t0, 1), "s")
