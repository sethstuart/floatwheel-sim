"""Build and render a Floatwheel from the sim's own geometry - run inside Blender.

Usage (from a cmd/PowerShell shell, Blender on the Windows side):
    blender -b --factory-startup --python tools/blender_scene.py -- ^
        --geometry out/geometry.json --out out/render_01.png ^
        [--view beauty|side|pushed] [--pitch 6] [--samples 96] [--res 1600x900]

The physics model is deliberately primitive (boxes and a cylinder: it is a balance
sim, not a CAD file). This script builds the *visual* board from the same numbers,
which is normal practice - visual mesh and collision shape are allowed to differ, and
sim/spec.py stays the single source of truth for both.

Notes learned the hard way, kept here so the next person does not re-learn them:
  * MuJoCo's EGL offscreen renderer is broken on this box (eglQueryString is None),
    so renders go through headless Blender.
  * Blender's mesh primitives take *diameter* for `radius1`/`depth`, while bmesh ops
    take *radius*. Mixing them up shows up as a tyre twice as fat as the spec.
  * Parenting with parent_inverse keeps world transforms while reparenting; forgetting
    it leaves the board embedded in the floor.
  * Principled BSDF input names move between Blender versions, so material writes go
    through set_in() and try the aliases in turn.
"""

import math
import os
import sys

import bpy


def arg(name, default=None):
    if "--" in sys.argv:
        a = sys.argv[sys.argv.index("--") + 1 :]
        if name in a:
            return a[a.index(name) + 1]
    return default


def set_in(node, names, value):
    if node is None:
        return False
    for n in names:
        if n in node.inputs:
            node.inputs[n].default_value = value
            return True
    return False


def material(name, colour, rough=0.5, metal=0.0, emit=None, emit_strength=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    set_in(bsdf, ["Base Color", "Color"], colour)
    set_in(bsdf, ["Roughness", "Specular"], rough)
    set_in(bsdf, ["Metallic"], metal)
    if emit is not None:
        set_in(bsdf, ["Emission Color", "Emission"], emit)
        set_in(bsdf, ["Emission Strength", "Emit"], emit_strength)
    return mat


def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def add_cyl(name, r, depth, loc, rot, matside, matcap, verts=96):
    """Primitive cylinders take DIAMETER for radius1 and depth."""
    bpy.ops.mesh.primitive_cylinder_add(
        radius=r, depth=depth, location=loc, rotation=rot, vertices=verts
    )
    o = bpy.context.active_object
    o.name = name
    o.data.materials.append(matside)
    return o


def add_torus(name, R, minor, loc, rot, mat, minor_seg=18, major_seg=96):
    """primitive_torus_add takes RADIUS for both radii (unlike the cylinder
    primitive, which takes diameter for depth and radius for radius1)."""
    bpy.ops.mesh.primitive_torus_add(
        major_radius=R,
        minor_radius=minor,
        major_segments=major_seg,
        minor_segments=minor_seg,
        location=loc,
        rotation=rot,
    )
    o = bpy.context.active_object
    o.name = name
    o.data.materials.append(mat)
    return o


def add_box(name, dims, loc, mat, bevel=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
    o = bpy.context.active_object
    o.name = name
    o.scale = (dims[0], dims[1], dims[2])
    o.location = loc
    o.data.materials.append(mat)
    if bevel > 0:
        m = o.modifiers.new("bevel", "BEVEL")
        m.width = min(bevel, min(dims) * 0.4)
        m.segments = 4
    return o


def main():
    gp = arg("--geometry", os.path.join("out", "geometry.json"))
    out = arg("--out", os.path.join("out", "render.png"))
    view = arg("--view", "beauty")
    pitch = float(arg("--pitch", "0"))
    samples = int(arg("--samples", "96"))
    res = arg("--res", "1400x875")
    engine = arg("--engine", "auto")
    # Blender's own cwd is C:\ when launched with -b from a shell, so a relative --out
    # wrote the first render to C:\out\render_beauty.png. Resolve everything against
    # the repo root, which is where the sim side wrote geometry.json.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gp = gp if os.path.isabs(gp) else os.path.join(root, gp)
    out = out if os.path.isabs(out) else os.path.join(root, out)
    import json

    g = json.load(open(gp, "r", encoding="utf-8"))

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for block in list(coll):
            if block.users == 0:
                coll.remove(block)

    rubber = material("tyre", (0.017, 0.017, 0.019, 1), 0.88)
    alloy = material("hub", (0.42, 0.43, 0.46, 1), 0.28, 1.0)
    shell = material("deck", (0.018, 0.019, 0.026, 1), 0.42)
    grip = material("grip", (0.020, 0.020, 0.022, 1), 0.96)
    ledm = material("led", (0.05, 0.05, 0.05, 1), 0.3, 0.0, g["led"]["colour"], 4.0)
    floor_m = material("floor", (0.16, 0.16, 0.17, 1), 0.45)

    R = g["wheel"]["radius"]
    W = g["wheel"]["width"]
    D = g["deck"]

    rig = bpy.data.objects.new("rig", None)
    link(rig)
    rig.location = (0, 0, R)
    axle = bpy.data.objects.new("axle", None)
    link(axle)
    axle.parent = rig

    # wheel: it rolls along x with its axis on y, so the wheel PLANE is x-z. The first
    # version placed spokes at (cos a, sin a, 0) and rotated them about z - the x-y
    # plane - which stuck them out of the sides of the wheel like a cross, and rendered
    # exactly that way. Anything about this wheel has to live in x-z.
    tyre = add_torus("tyre", R - W * 0.30, W * 0.30, (0, 0, 0), (math.pi / 2, 0, 0), rubber)
    tyre.parent = axle
    for side in (-1, 1):  # sidewall reflector stripes, the one bright detail a tyre has
        sw = add_torus(
            "sidewall%d" % side,
            R - W * 0.34,
            0.0035,
            (0, side * (W / 2 - 0.004), 0),
            (math.pi / 2, 0, 0),
            alloy,
        )
        sw.parent = axle
    rim = add_torus("rim", g["wheel"]["rim_radius"], 0.009, (0, 0, 0), (math.pi / 2, 0, 0), alloy)
    rim.parent = axle
    # A hub motor is a wide drum filling most of the wheel's width, not a small can:
    # the first render's 0.42R x 0.86W read as a paint roller.
    hub = add_cyl("hub", 0.62 * R, W * 0.62, (0, 0, 0), (math.pi / 2, 0, 0), alloy, alloy)
    hub.parent = axle
    cap = add_cyl("hubcap", 0.30 * R, W * 0.66, (0, 0, 0), (math.pi / 2, 0, 0), shell, shell)
    cap.parent = axle
    for i in range(g["wheel"]["spokes"]):
        a = 2 * math.pi * i / g["wheel"]["spokes"]
        rmid = (0.62 * R + g["wheel"]["rim_radius"]) * 0.5
        s = add_box(
            "spoke%d" % i,
            (g["wheel"]["rim_radius"] - 0.62 * R, 0.010, 0.014),
            (math.cos(a) * rmid, 0.0, math.sin(a) * rmid),
            alloy,
        )
        s.rotation_euler = (0.0, -a, 0.0)  # about y, because the wheel plane is x-z
        s.parent = axle

    # deck: the platform sits above the tyre line (see sim/model.py deck_z for why -
    # the tyre has to clear it, which the physics model needed too).
    deck = add_box(
        "deck",
        (D["length"], D["width"], D["thickness"]),
        (0, 0, D["z_above_axle"]),
        shell,
        bevel=0.010,
    )
    deck.parent = rig
    # Nose and tail: a plain slab reads as a prototype. Two thinner end pieces hung
    # below the full-length platform read as a deck that thins toward the ends. They go
    # *under* the slab rather than replacing it, so the footpads (which the footpad ADC
    # gating cares about) still sit on a surface, and the total length stays DECK_LEN so
    # the picture agrees with the collision box in sim/model.py.
    for side in (-1, 1):
        nose = add_box(
            "nose%d" % side,
            (D["length"] * 0.17, D["width"] * 0.92, D["thickness"] * 0.62),
            (
                side * D["length"] * 0.42,
                0,
                D["z_above_axle"] + D["thickness"] / 2 - 0.002 - D["thickness"] * 0.31,
            ),
            shell,
            bevel=0.007,
        )
        nose.parent = rig
    spine = add_box(
        "spine",
        (D["length"] * 0.34, D["width"] * 0.78, 0.020),
        (0, 0, D["z_above_axle"] + D["thickness"] / 2 + 0.008),
        shell,
        bevel=0.006,
    )
    spine.parent = rig
    for i, x in enumerate(g["footpads"]["x_offsets"]):
        pad = add_box(
            "pad%d" % i,
            (g["footpads"]["length"], g["footpads"]["width"], 0.006),
            (x, 0, D["z_above_axle"] + D["thickness"] / 2 + 0.003),
            grip,
            bevel=0.002,
        )
        pad.parent = rig
    for ysign in (-1, 1):
        strip = add_box(
            "led%d" % ysign,
            (D["length"] * 0.66, g["led"]["width"], 0.004),
            (0, ysign * (D["width"] / 2 - 0.002), D["z_above_axle"] - D["thickness"] / 2 + 0.002),
            ledm,
        )
        strip.parent = rig

    # ground
    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "ground"
    ground.data.materials.append(floor_m)

    lean = math.radians(pitch)
    rig.rotation_euler = (0, -lean, 0)

    target = bpy.data.objects.new("target", None)
    link(target)
    target.location = (0, 0, 0.20)

    if view == "side":
        cam_loc = (2.05, -0.05, 0.19)
        lens = 68
    elif view == "pushed":
        cam_loc = (1.34, -1.00, 0.44)
        lens = 46
    else:
        cam_loc = (1.24, -0.96, 0.34)
        lens = 50

    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = lens
    cam = bpy.data.objects.new("cam", cam_data)
    link(cam)
    cam.location = cam_loc
    cam.constraints.new("TRACK_TO")
    cam.constraints[0].target = target
    bpy.context.scene.camera = cam

    sun_d = bpy.data.lights.new("sun", "SUN")
    sun_d.energy = 2.2
    sun_d.angle = math.radians(2.0)
    sun_d.color = (1.0, 0.95, 0.88)
    sun = bpy.data.objects.new("sun", sun_d)
    link(sun)
    sun.rotation_euler = (math.radians(52), 0, math.radians(-36))

    key_d = bpy.data.lights.new("key", "AREA")
    key_d.energy = 110
    key_d.size = 1.8
    key_d.color = (1.0, 0.97, 0.93)
    key = bpy.data.objects.new("key", key_d)
    link(key)
    key.location = (1.5, -1.7, 2.3)
    key.rotation_euler = (math.radians(52), 0, math.radians(39))

    fill_d = bpy.data.lights.new("fill", "AREA")
    fill_d.energy = 40
    fill_d.size = 2.8
    fill_d.color = (0.82, 0.87, 1.0)
    fill = bpy.data.objects.new("fill", fill_d)
    link(fill)
    fill.location = (-1.9, -1.2, 1.5)
    fill.rotation_euler = (math.radians(60), 0, math.radians(-56))

    world = bpy.data.worlds.get("world") or bpy.data.worlds.new("world")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.35, 0.38, 0.44, 1)
        bg.inputs[1].default_value = 0.4

    w, h = (int(v) for v in res.lower().split("x"))
    sc = bpy.context.scene
    sc.render.resolution_x = w
    sc.render.resolution_y = h
    sc.render.film_transparent = False
    try:
        sc.view_settings.view_transform = "AgX"
        sc.view_settings.look = "AgX - Base Contrast"
    except TypeError:
        sc.view_settings.view_transform = "Filmic"

    def gpu_ready():
        """Blender 4.x API is prefs.compute_device_type + prefs.devices[].use. The
        prefs.backends loop I wrote first is a 5.x shape and raised
        'CyclesPreferences' object has no attribute 'backends' on 4.5."""
        prefs = context_preferences()
        if prefs is None:
            return False
        for kind in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
            try:
                prefs.compute_device_type = kind
                prefs.get_devices()
                used = 0
                for d in prefs.devices:
                    if d.type == kind:
                        d.use = True
                        used += 1
                if used:
                    return True
            except Exception:
                continue
        return False

    gpu = False
    try:
        gpu = gpu_ready()
    except Exception as e:
        print("gpu config failed: %s" % e)

    want = engine
    if want in ("auto", "cycles"):
        try:
            sc.render.engine = "CYCLES"
            sc.cycles.samples = samples
            sc.cycles.use_denoising = True
            sc.cycles.device = "GPU" if gpu else "CPU"
        except Exception as e:
            print("cycles unusable (%s), using EEVEE" % e)
            want = "eevee"
    if want == "eevee":
        try:
            sc.render.engine = "BLENDER_EEVEE_NEXT"
        except Exception:
            sc.render.engine = "BLENDER_EEVEE"
        sc.eevee.taa_render_samples = min(samples, 64)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    sc.render.filepath = out
    bpy.ops.render.render(write_still=True)
    dev = getattr(sc, "cycles", None) and getattr(sc.cycles, "device", "-")
    print(
        "RENDERED %s  view=%s pitch=%.1f engine=%s device=%s samples=%s"
        % (out, view, pitch, sc.render.engine, dev, samples)
    )


def context_preferences():
    try:
        return bpy.context.preferences.addons["cycles"].preferences
    except Exception:
        return None


if __name__ == "__main__":
    main()
