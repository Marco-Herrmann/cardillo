bl_info = {
    "name": "Animate Modes",
    "author": "Marco Herrmann",
    "version": (1, 1),
    "blender": (3, 0, 0),
    "category": "Object",
}

import bpy
from mathutils import Vector, Quaternion
import numpy as np


def get_reference_root():
    for obj in bpy.data.objects:
        if obj.name.endswith("_root"):
            return obj

    return None


def get_omegas():
    root = get_reference_root()
    if root is None:
        return []

    if "omegas" not in root:
        return []

    return root["omegas"]


def omega_items(self, context):
    omegas = get_omegas()

    items = []

    for i, omega in enumerate(omegas):
        items.append((str(i), f"Mode {i}", f"omega = {omega:.6f} 1/s"))

    if not items:
        items.append(("0", "No Modes", ""))

    return items


####################
# create animation #
####################
def get_q_align(v):
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-12:
        return Quaternion((1, 0, 0, 0)), 0

    u = v / v_norm
    dots = [u.dot(e) for e in np.eye(3)]
    idx = int(np.argmax([abs(d) for d in dots]))

    e_best = np.zeros(3, dtype=float)
    e_best[idx] = 1.0
    sign = np.sign(dots[idx])
    e_target = sign * e_best

    # TODO: we have to take the one where the cross product is largest
    axis = np.cross(e_target, u)
    angle = np.acos(u @ e_target)

    axis = axis / np.linalg.norm(axis)

    #    print(dots, axis, angle, idx)
    return Quaternion(axis, angle), idx


def bake_animation(idx, amplitude, play_time):
    scene = bpy.context.scene

    if play_time < 1e-12:
        play_time = 1.0

    fps = scene.render.fps
    scene.frame_start = 0
    scene.frame_end = int(fps * play_time)

    frame_start = scene.frame_start
    frame_end = scene.frame_end

    def time_factor(frame):
        t = frame / fps
        return np.sin(2.0 * np.pi * t / play_time)

    for root in bpy.data.objects:

        if not root.name.endswith("_root"):
            continue

        if "Delta_r" not in root or "B_Delta_phi" not in root:
            continue

        if not root.children:
            continue

        child = root.children[0]

        # eq position
        r_OP0 = np.array(root["r_OP0"], dtype=float)
        P_IB0 = Quaternion(np.array(root["P_IB0"], dtype=float))

        # displacements
        Delta_r = np.array(root["Delta_r"][idx], dtype=float)
        B_Delta_phi = np.array(root["B_Delta_phi"][idx], dtype=float)

        # clear animation
        root.animation_data_clear()
        child.animation_data_clear()

        # update for scale
        q_align, align_idx = get_q_align(B_Delta_phi)

        # go trhough frames
        for frame in range(frame_start, frame_end + 1):
            A_t = amplitude * time_factor(frame)

            # translation
            d_r = A_t * Delta_r
            root.location = Vector(r_OP0 + d_r)
            root.keyframe_insert(data_path="location", frame=frame)

            # rotation
            dphi_vec = A_t * B_Delta_phi
            dphi = np.linalg.norm(dphi_vec)

            if dphi < 1e-12:
                dq = Quaternion((1, 0, 0, 0))
            else:
                axis = Vector(dphi_vec / dphi)
                angle = np.arctan(dphi)
                dq = Quaternion(axis, angle)

            root.rotation_mode = "QUATERNION"
            root.rotation_quaternion = P_IB0 @ q_align
            root.keyframe_insert(data_path="rotation_quaternion", frame=frame)

            s = np.sqrt(1 + dphi**2) * np.ones(3)
            s[align_idx] = 1.0
            root.scale = s
            root.keyframe_insert(data_path="scale", frame=frame)

            # child rotation
            child.rotation_mode = "QUATERNION"
            child.rotation_quaternion = q_align.inverted() @ dq
            child.keyframe_insert(data_path="rotation_quaternion", frame=frame)


class AnimateModesProperties(bpy.types.PropertyGroup):
    select_mode: bpy.props.EnumProperty(name="Mode", items=omega_items)
    current_omega: bpy.props.StringProperty(name="Omega", default="---")
    amplitude: bpy.props.FloatProperty(name="Amplitude", default=1.0)
    play_time: bpy.props.FloatProperty(name="Play Time (s)", default=2.0, min=0.01)


class ANIMATEMODES_OT_update(bpy.types.Operator):
    bl_idname = "animatemodes.update"
    bl_label = "Update"

    def execute(self, context):
        props = context.scene.animate_modes_props
        omegas = get_omegas()

        if not omegas:
            self.report({"WARNING"}, "No omegas found")
            return {"CANCELLED"}

        idx = int(props.select_mode)
        idx = min(idx, len(omegas) - 1)

        props.current_omega = f"{omegas[idx]:.6f} 1/s"

        bake_animation(idx, props.amplitude, props.play_time)
        return {"FINISHED"}


class ANIMATEMODES_PT_panel(bpy.types.Panel):
    bl_label = "Animate Modes"
    bl_idname = "ANIMATEMODES_PT_panel"

    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Animate Modes"

    def draw(self, context):
        layout = self.layout
        props = context.scene.animate_modes_props

        layout.prop(props, "select_mode")
        layout.label(text=f"Omega: {props.current_omega}")
        layout.separator()

        layout.prop(props, "amplitude")
        layout.prop(props, "play_time")
        layout.separator()

        layout.operator("animatemodes.update")


classes = (
    AnimateModesProperties,
    ANIMATEMODES_OT_update,
    ANIMATEMODES_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.animate_modes_props = bpy.props.PointerProperty(
        type=AnimateModesProperties
    )


def unregister():
    del bpy.types.Scene.animate_modes_props

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
