import bpy
import numpy as np
import json
import os
from mathutils import Vector, Quaternion

# =========================================================
# CONFIG
# =========================================================
DATA_DIR = r"D:\git-projects\cardillo\cardillo\visualization\export"
JSON_FILES = ["rod_A.json"]


DATA_DIR = r"D:\git-projects\cardillo\examples\Herrmann2025_mixed_Cosserat_rod\blender\slen_1e+01\helix_nel16"
JSON_FILES = ["Rod.json"]

ARMATURE_SUFFIX = "_arm"
QUAD_SIZE = 0.2


# =========================================================
# QUAD MESH
# =========================================================
def create_rectange(n, wy, wz):
    verts = []
    for i in range(n):
        v0 = (0.0, -wy / 2, -wz / 2)
        v1 = (0.0, wy / 2, -wz / 2)
        v2 = (0.0, wy / 2, wz / 2)
        v3 = (0.0, -wy / 2, wz / 2)

        verts.extend([v0, v1, v2, v3])

    return verts, 4


def create_circle(n, r):
    # TODO: maybe change number of vertices per cross-section
    n_cs = 32
    verts = []
    for i in range(n):
        for j in range(n_cs):
            angle = 2 * np.pi * j / n_cs
            y = r * np.cos(angle)
            z = r * np.sin(angle)
            verts.append((0.0, y, z))

    return verts, n_cs


# =========================================================
# ARMATURE
# =========================================================
def create_armature(name, n):

    arm_name = name + ARMATURE_SUFFIX

    if arm_name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[arm_name], do_unlink=True)

    arm_data = bpy.data.armatures.new(arm_name)
    arm_obj = bpy.data.objects.new(arm_name, arm_data)
    bpy.context.collection.objects.link(arm_obj)

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")

    bones = []

    for i in range(n):

        bone = arm_data.edit_bones.new("Bone_{}".format(i))
        bone.head = (0.0, 0.0, 0.0)
        bone.tail = (0.0, 0.1, 0.0)  # must be like this!

        bones.append(bone)

    bpy.ops.object.mode_set(mode="OBJECT")

    return arm_obj


# =========================================================
# VERTEX GROUPS
# =========================================================
def assign_vertex_groups(obj, n, n_cs):

    for i in range(n):
        vg = obj.vertex_groups.new(name=f"Bone_{i}")
        vg.add([int(x + n_cs * i) for x in np.arange(n_cs)], 1.0, "REPLACE")


# =========================================================
# ARMATURE MODIFIER
# =========================================================
def add_armature_modifier(obj, arm):

    mod = obj.modifiers.get("Armature")

    if mod is None:
        mod = obj.modifiers.new("Armature", "ARMATURE")

    mod.object = arm

    # edge split
    mod = obj.modifiers.new("EdgeSplit", "EDGE_SPLIT")
    mod.split_angle = np.deg2rad(30)
    mod.use_edge_angle = True
    mod.use_edge_sharp = True


# =========================================================
# OBJECT GET
# =========================================================
def get_object(name, n, cross_section):

    if name in bpy.data.objects:
        return bpy.data.objects[name]

    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    # create mesh
    mesh_ = obj.data
    mesh_.clear_geometry()

    # vertices
    if cross_section["type"] == "rectangle":
        verts, n_cs = create_rectange(
            n, cross_section["width"], cross_section["height"]
        )
    elif cross_section["type"] == "circle":
        verts, n_cs = create_circle(n, cross_section["radius"])

    # faces
    f0 = np.arange(n_cs, 0, -1) - 1
    fn = np.arange(n_cs) + (n - 1) * n_cs
    faces = [f0, fn]
    for i in range(n - 1):
        base = i * n_cs
        for j in range(n_cs):
            v1 = base + j
            v2 = base + (j + 1) % n_cs
            v3 = base + (j + 1) % n_cs + n_cs
            v4 = base + j + n_cs
            faces.append((v1, v2, v3, v4))

    # update mesh
    mesh_.from_pydata(verts, [], faces)
    mesh_.update()

    # somehow important (ChatGPT)
    bpy.context.view_layer.update()

    # set all polygons in smooth
    for p in mesh_.polygons:
        p.use_smooth = True

    return obj, n_cs


# =========================================================
# LOAD DATA
# =========================================================
def load_json(file):

    path = os.path.join(DATA_DIR, file)

    with open(path, "r") as f:
        meta = json.load(f)

    frames = []
    for fname in meta["frames"]:
        frames.append(np.load(os.path.join(DATA_DIR, fname)))

    m = len(frames)
    n = frames[0].shape[0]

    name = meta["name"]
    cross_section = meta["cross_section"]
    obj, n_cs = get_object(name, n, cross_section)

    # =====================================================
    # ARMATURE
    # =====================================================
    arm = create_armature(name, n)
    add_armature_modifier(obj, arm)
    assign_vertex_groups(obj, n, n_cs)

    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")

    # =====================================================
    # ANIMATION
    # =====================================================
    for fi, frame in enumerate(frames):

        bpy.context.scene.frame_set(fi)

        for i in range(n):

            bone = arm.pose.bones[f"Bone_{i}"]

            pos = frame[i, 0:3]
            q = Quaternion(frame[i, 3:7])

            bone.location = Vector(pos)
            bone.rotation_mode = "QUATERNION"
            bone.rotation_quaternion = q

            bone.keyframe_insert(data_path="location")
            bone.keyframe_insert(data_path="rotation_quaternion")

    bpy.ops.object.mode_set(mode="OBJECT")

    print("Loaded:", name, "n:", n, "m:", m)


# =========================================================
# RUN
# =========================================================
for jf in JSON_FILES:
    load_json(jf)

print("DONE")
