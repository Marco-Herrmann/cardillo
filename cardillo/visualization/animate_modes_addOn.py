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


_omega_items_cache = []


def omega_items(self, context):
    global _omega_items_cache

    omegas = get_omegas()

    items = []

    for i, omega in enumerate(omegas):
        items.append((str(i), f"Mode {i}", f"omega = {omega:.6f} 1/s"))

    if not items:
        items.append(("0", "No Modes", ""))

    # Blender does not keep its own reference to the strings in dynamic
    # enum items, so a module-level cache is needed to prevent the
    # returned strings from being garbage-collected while still in use.
    _omega_items_cache = items

    return _omega_items_cache


####################
# create animation #
####################
def decomposition(P_IB0, B_Delta_phi):
    """find P_IJ, P_JB, s such that
        A_IJ @ S0 @ A_JB = A_IB0 @ (I + scale * ax2skew(B_Delta_phi))
    , with S = diag(s) and A_IB0 = A(P_IB0), A_IJ = A(P_IJ), A_JB = A(P_JB) in SO(3)"""

    Delta_phi = np.linalg.norm(B_Delta_phi)
    if Delta_phi < 1e-12:
        s = np.ones(3)

        def fun(scale):
            q_scalar = 1 - Delta_phi**2 * scale**2 / 8
            q_vec = scale / 2 * B_Delta_phi
            return Quaternion([q_scalar, *q_vec]), s

        P_IJ = P_IB0

    else:
        i = np.argmin(np.abs(B_Delta_phi))
        B_target = np.zeros(3, dtype=float)
        B_target[i] = 1.0

        B_n = B_Delta_phi / Delta_phi

        B_axis = np.cross(B_target, B_n)
        B_axis /= np.linalg.norm(B_axis)
        angle = np.arccos(np.clip(B_n @ B_target, -1.0, 1.0))

        P_B0J = Quaternion(B_axis, angle)
        P_IJ = P_IB0 @ P_B0J

        s0 = np.ones(3)

        def fun(scale):
            s = s0 * np.sqrt(1 + scale**2 * Delta_phi**2)
            s[i] = 1.0

            angle_ = np.arctan(scale * Delta_phi)
            P_JB = P_B0J.inverted() @ Quaternion(B_n, angle_)

            return P_JB, s

    return P_IJ, fun


def smallest_rotation_quaternion(v, i=None):
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-12:
        if i is None:
            i = 0
        return np.array([1, 0, 0, 0], dtype=float), i

    u = v / v_norm

    ei = np.zeros(3, dtype=float)
    if i is None:
        dots = [u @ e for e in np.eye(3)]
        i = int(np.argmax([abs(d) for d in dots]))
        ei[i] = 1.0 * np.sign(dots[i])
    else:
        ei[i] = 1.0

    inner = u @ ei
    # Handle the case where u is parallel to ei
    if abs(inner) > 1.0 - 1e-12:
        if inner > 0:
            # already aligned
            return np.array([1, 0, 0, 0], dtype=float), i
        else:
            # 180° rotation
            q = np.zeros(4, dtype=float)
            q[1 + (i + 1) % 3] = 1.0
            return q, i

    angle = np.arccos(np.clip(u @ ei, -1.0, 1.0))
    axis = np.cross(ei, u)
    axis = axis / np.linalg.norm(axis)
    p, p0 = np.sin(angle / 2) * axis, np.cos(angle / 2)
    return np.array([p0, *p]), i


def bake_arrow_root(
    root, r_OP0, r_OP1, Delta_r_P0, Delta_r_P1, scale_perp, frames, A_t_all
):
    """Keyframe a line/arrow root: it points from a moving P0 to a moving P1."""
    root.rotation_mode = "QUATERNION"
    scale_dir = np.array([scale_perp, scale_perp, 1.0])

    # translation, direction and length for every frame at once
    r0_all = r_OP0 + A_t_all[:, None] * Delta_r_P0
    r1_all = r_OP1 + A_t_all[:, None] * Delta_r_P1
    a_all = r1_all - r0_all
    length_all = np.linalg.norm(a_all, axis=1)

    for k, frame in enumerate(frames):
        frame = int(frame)

        root.location = Vector(r0_all[k])
        root.keyframe_insert(data_path="location", frame=frame)

        look_quat, _ = smallest_rotation_quaternion(a_all[k], 2)
        root.rotation_quaternion = look_quat
        root.keyframe_insert(data_path="rotation_quaternion", frame=frame)

        root.scale = Vector(scale_dir * length_all[k])
        root.keyframe_insert(data_path="scale", frame=frame)


def bake_discrete_root(
    root, child, r_OP0, P_IB0, Delta_r, B_Delta_phi, frames, A_t_all
):
    """Keyframe a root/child pair: root translates, child carries the residual
    rotation of the linearized (non-orthogonal) mode rotation, see decomposition()."""
    root.rotation_mode = "QUATERNION"
    child.rotation_mode = "QUATERNION"

    # translation for every frame at once
    d_r_all = A_t_all[:, None] * Delta_r

    # root rotation is constant per mode
    P_IJ, fun = decomposition(P_IB0, B_Delta_phi)
    root.rotation_quaternion = P_IJ

    for k, frame in enumerate(frames):
        frame = int(frame)

        root.location = Vector(r_OP0 + d_r_all[k])
        root.keyframe_insert(data_path="location", frame=frame)

        P_JB, s = fun(A_t_all[k])

        root.scale = s
        root.keyframe_insert(data_path="scale", frame=frame)

        child.rotation_quaternion = P_JB
        child.keyframe_insert(data_path="rotation_quaternion", frame=frame)


def bake_animation(idx, amplitude, play_time):
    scene = bpy.context.scene

    if play_time < 1e-12:
        play_time = 1.0

    fps = scene.render.fps
    scene.frame_start = 0
    scene.frame_end = int(fps * play_time)

    # frame range and oscillation amplitude are shared by every root, so
    # compute them once instead of per root/frame
    frames = np.arange(scene.frame_start, scene.frame_end + 1)
    t = frames / fps
    A_t_all = amplitude * np.sin(2.0 * np.pi * t / play_time)

    # arrow/line
    for root in bpy.data.objects:
        if not root.name.endswith("_root"):
            continue

        if (
            "r_OP0" not in root
            or "r_OP1" not in root
            or "Delta_r_P0" not in root
            or "Delta_r_P1" not in root
        ):
            continue

        r_OP0 = np.array(root["r_OP0"], dtype=float)
        r_OP1 = np.array(root["r_OP1"], dtype=float)
        Delta_r_P0 = np.array(root["Delta_r_P0"][idx], dtype=float)
        Delta_r_P1 = np.array(root["Delta_r_P1"][idx], dtype=float)
        scale_perp = root["scale_perp"]

        root.animation_data_clear()
        bake_arrow_root(
            root, r_OP0, r_OP1, Delta_r_P0, Delta_r_P1, scale_perp, frames, A_t_all
        )

    # discrete
    for root in bpy.data.objects:
        if not root.name.endswith("_root"):
            continue

        if "Delta_r" not in root or "B_Delta_phi" not in root:
            continue

        if not root.children:
            continue

        child = root.children[0]

        r_OP0 = np.array(root["r_OP0"], dtype=float)
        P_IB0 = Quaternion(np.array(root["P_IB0"], dtype=float))
        Delta_r = np.array(root["Delta_r"][idx], dtype=float)
        B_Delta_phi = np.array(root["B_Delta_phi"][idx], dtype=float)

        root.animation_data_clear()
        child.animation_data_clear()
        bake_discrete_root(
            root, child, r_OP0, P_IB0, Delta_r, B_Delta_phi, frames, A_t_all
        )

    # armatures
    for arm_obj in bpy.data.objects:
        if arm_obj.type != "ARMATURE":
            continue

        pose_bones = arm_obj.pose.bones
        arm_obj.animation_data_clear()

        for root in pose_bones:
            if not root.name.endswith("_root"):
                continue

            if "Delta_r" not in root or "B_Delta_phi" not in root:
                continue

            if not root.children:
                continue

            child = root.children[0]

            r_OP0 = np.array(root["r_OP0"], dtype=float)
            P_IB0 = Quaternion(np.array(root["P_IB0"], dtype=float))
            Delta_r = np.array(root["Delta_r"][idx], dtype=float)
            B_Delta_phi = np.array(root["B_Delta_phi"][idx], dtype=float)

            bake_discrete_root(
                root, child, r_OP0, P_IB0, Delta_r, B_Delta_phi, frames, A_t_all
            )


class AnimateModesProperties(bpy.types.PropertyGroup):
    select_mode: bpy.props.EnumProperty(name="Mode", items=omega_items)
    current_omega: bpy.props.StringProperty(name="Omega", default="---")
    amplitude: bpy.props.FloatProperty(name="Amplitude", default=1.0)
    play_time: bpy.props.FloatProperty(name="Play Time (s)", default=2.0, min=0.01)


def update_animation(context):
    props = context.scene.animate_modes_props
    omegas = get_omegas()

    if not omegas:
        return False

    idx = int(props.select_mode)
    idx = min(idx, len(omegas) - 1)

    props.current_omega = f"{omegas[idx]:.6f} 1/s"

    bake_animation(idx, props.amplitude, props.play_time)
    return True


class ANIMATEMODES_OT_update(bpy.types.Operator):
    bl_idname = "animatemodes.update"
    bl_label = "Update"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        if not update_animation(context):
            self.report({"WARNING"}, "No omegas found")
            return {"CANCELLED"}

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

    context = bpy.context
    if context.scene is not None:
        context.scene.animate_modes_props.select_mode = "0"
        update_animation(context)


def unregister():
    del bpy.types.Scene.animate_modes_props

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
