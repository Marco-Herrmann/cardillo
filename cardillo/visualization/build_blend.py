import bpy
from pathlib import Path
import sys
import numpy as np

argv = sys.argv
argv = argv[argv.index("--") + 1 :]

output_path = argv[0]
gltf_files = sorted(Path(argv[1]).glob("*.glb"))

bpy.ops.wm.read_factory_settings(use_empty=True)

# gather everything the gltf importer creates into one dedicated collection,
# instead of relying on whatever the default active collection happens to be
collection = bpy.data.collections.new(Path(output_path).stem)
bpy.context.scene.collection.children.link(collection)
bpy.context.view_layer.active_layer_collection = (
    bpy.context.view_layer.layer_collection.children[collection.name]
)

for f in gltf_files:
    bpy.ops.import_scene.gltf(filepath=str(f), bone_heuristic="TEMPERANCE")

# auto smooth objects
# TODO: it is not the best way to go via selection and active objects
bpy.ops.object.select_all(action="DESELECT")
for obj in bpy.context.scene.objects:
    if obj.type == "MESH":
        obj.select_set(True)

if len(bpy.context.selected_objects) > 0:
    bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
    bpy.ops.object.shade_auto_smooth(angle=np.deg2rad(80))

# give each mesh its own material, so colors can be tweaked per body later
for obj in bpy.context.scene.objects:
    if obj.type != "MESH" or obj.data.materials:
        continue

    mat = bpy.data.materials.new(name=obj.data.name)
    mat.use_nodes = True
    obj.data.materials.append(mat)

# handling of empties
for obj in bpy.context.scene.objects:
    if "__invisible" in obj.name:
        obj.hide_viewport = True
        # obj.hide_render = True
        obj.name = obj.name.replace("__invisible", "")

    if obj.name.endswith("_root"):
        obj.hide_viewport = True

    if obj.type == "EMPTY":
        if obj.name.endswith("_obj"):
            obj.empty_display_type = "ARROWS"
        elif obj.name.endswith("_v"):
            obj.empty_display_type = "SINGLE_ARROW"
        elif obj.name.endswith("_Omega"):
            obj.empty_display_type = "SINGLE_ARROW"
        elif obj.name.endswith("_vec"):
            obj.empty_display_type = "SINGLE_ARROW"
        elif obj.name.endswith("_block"):
            obj.empty_display_type = "CUBE"
        obj.empty_display_size = 1

# adjust animation frames
max_frame = 0
for obj in bpy.data.objects:
    ad = obj.animation_data
    if not ad or not ad.action:
        continue

    # TODO: this is really ugly. See if we can check for only one, as they should have the same number of frames
    action = ad.action
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for fc in bag.fcurves:
                    for kp in fc.keyframe_points:
                        max_frame = max(max_frame, kp.co.x)

bpy.context.scene.frame_current = 0
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = int(np.ceil(max_frame))

# deselect all objects and save
bpy.ops.object.select_all(action="DESELECT")
bpy.context.preferences.filepaths.save_version = 0
bpy.ops.wm.save_as_mainfile(filepath=output_path)
