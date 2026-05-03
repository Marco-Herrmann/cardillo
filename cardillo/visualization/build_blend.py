import bpy
import sys

argv = sys.argv
argv = argv[argv.index("--") + 1 :]

output_path = argv[0]
gltf_files = argv[1:]

bpy.ops.wm.read_factory_settings(use_empty=True)

for f in gltf_files:
    bpy.ops.import_scene.gltf(filepath=f)

bpy.ops.wm.save_as_mainfile(filepath=output_path)
