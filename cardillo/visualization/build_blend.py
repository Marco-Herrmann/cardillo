import bpy
import sys
import numpy as np

argv = sys.argv
argv = argv[argv.index("--") + 1 :]

output_path = argv[0]
gltf_files = argv[1:]

bpy.ops.wm.read_factory_settings(use_empty=True)

for f in gltf_files:
    bpy.ops.import_scene.gltf(filepath=f)

# auto smooth objects
# TODO: it is not the best way to go via selection and active objects
bpy.ops.object.select_all(action="DESELECT")
for obj in bpy.context.scene.objects:
    if obj.type == "MESH":
        obj.select_set(True)

if len(bpy.context.selected_objects) > 0:
    bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
    bpy.ops.object.shade_auto_smooth(angle=np.deg2rad(80))

# handling of empties
for obj in bpy.context.scene.objects:
    if obj.type == "EMPTY":
        if obj.name.endswith("_obj"):
            obj.empty_display_type = "ARROWS"
        elif obj.name.endswith("_v"):
            obj.empty_display_type = "SINGLE_ARROW"
        elif obj.name.endswith("_Omega"):
            obj.empty_display_type = "SINGLE_ARROW"
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
bpy.ops.wm.save_as_mainfile(filepath=output_path)
