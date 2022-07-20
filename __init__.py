from .helperFunctions import find_by_name
import bpy
import mathutils

from bpy.props import (
    BoolProperty,
    FloatVectorProperty
)

bl_info = {
    'name': 'Custom Blender Addon',
    'author': 'Marius Unger',
    'version': (0, 0, 1),
    'blender': (3, 0, 0),
    'location': 'Property Panel',
    'description': 'Helper Functions to rig feet.',
    'warning': '',
    'wiki_url': '',
    'tracker_url': '',
    'category': 'Mesh'
}


class bone_property_group(bpy.types.PropertyGroup):
    # UI
    expanded: BoolProperty(default=False)
    # 3D Cursor
    center_cursor: BoolProperty(default=False)
    cursor_offset: FloatVectorProperty(default=(0.0, 0.0, 0.0))
    # Joints
    joint_1: bpy.props.FloatVectorProperty(default=(0.0, 0.0, 0.0))
    joint_2: bpy.props.FloatVectorProperty(default=(0.0, 0.0, 0.0))
    joint_3: bpy.props.FloatVectorProperty(default=(0.0, 0.0, 0.0))
    joint_4: bpy.props.FloatVectorProperty(default=(0.0, 0.0, 0.0))
    # Gizmos
    show_gizmos: BoolProperty(default=True)


PropList = [
    "center_cursor",
    "cursor_offset",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "show_gizmos"
]


class custom_panel(bpy.types.Panel):
    bl_idname = "OBJECT_PT_custom_panel"
    bl_region_type = "WINDOW"
    bl_space_type = "PROPERTIES"
    bl_label = "Addon Rigging Tools"

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        row = box.row()
        row.prop(context.scene.bone_props, "expanded",
                icon="TRIA_DOWN" if context.scene.bone_props.expanded else "TRIA_RIGHT",
                icon_only=True, emboss=False)
        row.label(text="All Props")

        if context.scene.bone_props.expanded:
            for prop_names in PropList:
                row = box.row()
                row.prop(context.scene.bone_props, prop_names)


class UI_bone_settings(bpy.types.AddonPreferences):
    bl_idname = __name__

    test_bool: BoolProperty(
        default=False,
        name="Test Prop",
        description="Test Description"
    )

    def draw(self, context):
        layout = self.layout
        col = layout.column()
        col.prop(self, "test_bool")


def handler_function(scene):
    print("Handler Update")


classes = (
    bone_property_group,
    custom_panel,
    UI_bone_settings,
)


def register():
    from bpy.utils import register_class

    for cls in classes:
        register_class(cls)

    bpy.types.Scene.bone_props = bpy.props.PointerProperty(type=bone_property_group)
    bpy.app.handlers.depsgraph_update_post.append(handler_function)


def unregister():
    from bpy.utils import unregister_class

    del bpy.types.Scene.bone_props
    for cls in classes:
        unregister_class(cls)

    bpy.app.handlers.depsgraph_update_post.remove(handler_function)


if __name__ == "__main__":
    register()
