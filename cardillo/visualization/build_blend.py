import bpy
from pathlib import Path
import sys
import numpy as np

argv = sys.argv
argv = argv[argv.index("--") + 1 :]

source_path = argv[0]
link_path = argv[1]
gltf_files = sorted(Path(argv[2]).glob("*.glb"))

bpy.ops.wm.read_factory_settings(use_empty=True)

# gather everything the gltf importer creates into one dedicated collection,
# instead of relying on whatever the default active collection happens to be
collection_name = "System"
collection = bpy.data.collections.new(collection_name)
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

# give every mesh the same shared material, instead of one material per mesh.
# The slot is object-linked, so a later per-object override (materials.blend)
# can assign a different material to each object independently.
material = bpy.data.materials.new(name="Material")
for obj in bpy.context.scene.objects:
    if obj.type != "MESH" or obj.data.materials:
        continue

    obj.data.materials.append(material)
    obj.material_slots[0].link = "OBJECT"
    obj.material_slots[0].material = material

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

# give each "_block" arrow empty a default cylinder as its visible shape.
# Iterate over a snapshot list since this loop adds new objects to the scene.
for obj in list(bpy.context.scene.objects):
    if obj.type != "EMPTY" or not obj.name.endswith("_block"):
        continue

    bpy.ops.mesh.primitive_cylinder_add(location=(0, 0, 0), rotation=(0, 0, 0))
    cylinder = bpy.context.object
    cylinder.name = f"{obj.name}_cylinder"
    cylinder.data.materials.append(material)
    cylinder.material_slots[0].link = "OBJECT"
    cylinder.material_slots[0].material = material
    bpy.ops.object.shade_auto_smooth(angle=np.deg2rad(80))
    # the empty's own non-uniform scale (thin, thin, long) does the stretching
    cylinder.parent = obj

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

# deselect all objects and save the file that owns all the imported data
bpy.ops.object.select_all(action="DESELECT")
bpy.context.preferences.filepaths.save_version = 0
bpy.ops.wm.save_as_mainfile(filepath=source_path)

# start a fresh file that only links the collection from source_path, so it
# stays small and updates automatically whenever source_path is rebuilt.
# Only create it once: it is meant for local additions (camera, lights, ...)
# and must not be clobbered by later reruns of this script.
if not Path(link_path).exists():
    bpy.ops.wm.read_factory_settings(use_empty=True)

    with bpy.data.libraries.load(source_path, link=True) as (data_from, data_to):
        data_to.collections = [collection_name]

    linked_collection = data_to.collections[0]
    bpy.context.scene.collection.children.link(linked_collection)

    # equivalent to Outliner > Library Override > "Selected and Content":
    # gives every object in the collection its own local, editable override
    # (transform, visibility, modifiers, material slots, ...) while mesh and
    # action data stay linked to system.blend. This is the non-UI API
    # because background mode (blender -b) has no Outliner to run that
    # operator from. do_fully_editable=True is needed, otherwise most
    # properties are locked "system overrides" instead of freely editable.
    linked_collection.override_hierarchy_create(
        bpy.context.scene, bpy.context.view_layer, do_fully_editable=True
    )

    # override_hierarchy_create adds the override collection to the scene but
    # does not remove the plain link we made above, so drop it ourselves
    bpy.context.scene.collection.children.unlink(linked_collection)

    bpy.context.scene.frame_current = 0
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = int(np.ceil(max_frame))

    bpy.ops.wm.save_as_mainfile(filepath=link_path)
