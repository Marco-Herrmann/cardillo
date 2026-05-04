import bpy
import sys

argv = sys.argv
argv = argv[argv.index("--") + 1 :]

output_path = argv[0]
gltf_files = argv[1:]

bpy.ops.wm.read_factory_settings(use_empty=True)

for f in gltf_files:
    bpy.ops.import_scene.gltf(filepath=f)


for obj in bpy.context.scene.objects:
    if obj.type == "EMPTY":
        if obj.name.endswith("_obj"):
            obj.empty_display_type = "ARROWS"
        elif obj.name.endswith("_v"):
            obj.empty_display_type = "SINGLE_ARROW"
        elif obj.name.endswith("_Omega"):
            obj.empty_display_type = "SINGLE_ARROW"
        obj.empty_display_size = 1

bpy.ops.wm.save_as_mainfile(filepath=output_path)
