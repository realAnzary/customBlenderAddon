import bmesh
import bpy
import mathutils


bl_info = {
    "name": "Custom Addon",
    "description": "Custom Toolkit zum Riggen von Füßen",
    "author": "Marius Unger",
    "doc_url": "https://github.com/realAnzary/customBlenderAddon",
    "category": "Object",
    "location": "Porperties Panel",
    "warning": "",
    "version": (1, 0, 1),
    "blender": (3, 0, 0)
}


def find_by_name(scene, name):
    returnedObjects = []
    for objects in scene.objects:
        if name in objects.name:
            returnedObjects.append(objects)
    return returnedObjects


class CustomPropertyGroup(bpy.types.PropertyGroup):
    # Props für 3D-Cursor Position
    follow_bool: bpy.props.BoolProperty(name="center_3d_cursor")
    cursor_offset: bpy.props.FloatVectorProperty(name="offset")
    # Punkte im Fuß
    cruris_vec: bpy.props.FloatVectorProperty(name="joint_top")
    talus_vec: bpy.props.FloatVectorProperty(name="joint_middle")
    antetarsus_vec: bpy.props.FloatVectorProperty(name="joint_front")
    calcaneus_vec: bpy.props.FloatVectorProperty(name="joint_back")
    # Placeholder Skalierung
    joint_size: bpy.props.FloatProperty(name="joint_scale", default=1)


class CustomAddonPanel(bpy.types.Panel):
    bl_idname = "OBJECT_PT_custom_panel"
    bl_region_type = 'WINDOW'
    bl_space_type = 'PROPERTIES'
    bl_label = 'Addon Rigging Tools'

    def draw(self, context):
        layout = self.layout
        layout.label(text="Allgemeine Tools")
        layout.operator('custom.center_selected', text="Objekt zentrieren")
        layout.label(text="3D Cursor Tools")
        layout.operator('custom.spawn_anchor', text="Ankerpunkte setzen")
        layout.prop(context.scene.custom_props, "follow_bool", text="3D Cursor zentrieren")
        layout.prop(context.scene.custom_props, "cursor_offset", text="Cursor Offset")
        layout.label(text="Rigging Tools")
        layout.operator('custom.spawn_bones', text="Armature & Knochen hinzufügen")
        layout.operator('custom.visualize_joints', text="Joints anzeigen")
        layout.prop(context.scene.custom_props, "joint_size", text="Placeholder Größe")
        layout.label(text="Joint Positionen setzen")
        layout.operator('custom.set_value_joint1', text="Cruris / Joint1 Position setzen")
        layout.operator('custom.set_value_joint2', text="Talus / Joint2 Position setzen")
        layout.operator('custom.set_value_joint3', text="Antetarsus / Joint3 Position setzen")
        layout.operator('custom.set_value_joint4', text="Calcaneus / Joint4 Position setzen")
        layout.label(text="Joint Informationen")
        layout.prop(context.scene.custom_props, "cruris_vec")
        layout.prop(context.scene.custom_props, "talus_vec")
        layout.prop(context.scene.custom_props, "antetarsus_vec")
        layout.prop(context.scene.custom_props, "calcaneus_vec")


class CenterSelected(bpy.types.Operator):
    """Zentriert ein ausgewähltes Objekt und den 3D Cursor"""
    bl_idname = "custom.center_selected"
    bl_label = "Center selected Object"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT"

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def execute(self, context):
        bpy.context.area.type = 'VIEW_3D'
        bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
        bpy.ops.view3d.snap_selected_to_cursor()
        bpy.context.area.ui_type = 'PROPERTIES'
        return {'FINISHED'}


class SpawnAnchorPoints(bpy.types.Operator):
    """Fügt der Szene Objekte hinzu; Dienen als Anhaltspunkte um 3D Cursor zu zentrieren;
Muss im Edit-Mode benutzt werden und platziert für jeden asugewählten Vertex ein Ankerpunkt"""
    bl_idname = 'custom.spawn_anchor'
    bl_label = "Adds Anchor-Points to the Scene"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH"

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def execute(self, context):
        name_list = ["AnchorPoint.{:03d}".format(c + 1) for c in range(0, 10)]
        selected_obj = context.object.data
        mesh = bmesh.new()  # Vlt Unused Api Documentation hat das aber drinnen
        mesh = bmesh.from_edit_mesh(selected_obj)
        selectedVerts = [verts for verts in mesh.verts if verts.select]
        for selected in range(0, len(selectedVerts)):
            object = bpy.data.objects.new(name_list[selected], object_data=None)
            bpy.context.scene.collection.objects.link(object)

            object.location = selectedVerts[selected].co
            object.empty_display_size = 2
            object.empty_display_type = "PLAIN_AXES"
            object.select_set(True)

        return {"FINISHED"}


class SpawnBones(bpy.types.Operator):
    """Vebindet alle Joints in der Szene mit Knochen"""
    bl_idname = 'custom.spawn_bones'
    bl_label = "Spawn Bones in corresponding Position"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        bpy.ops.object.armature_add(enter_editmode=False, align="WORLD", location=(0, 0, 0), scale=(1, 1, 1))
        arm_obj = bpy.data.objects["Armature"]
        bpy.ops.object.select_all(action='DESELECT')
        arm_obj.select_set(True)
        bpy.ops.object.editmode_toggle()

        edit_bones = arm_obj.data.edit_bones

        for bones in arm_obj.data.edit_bones:
            if bones.name == "Bone":
                arm_obj.data.edit_bones.remove(bones)

        bone = edit_bones.new('Bone1')
        bone.head = context.scene.custom_props.cruris_vec
        bone.tail = context.scene.custom_props.talus_vec
        bone.select_tail = True

        bone = edit_bones.new('Bone2')
        bone.head = context.scene.custom_props.talus_vec
        bone.tail = context.scene.custom_props.antetarsus_vec
        bone.parent = edit_bones['Bone1']

        bone = edit_bones.new('Bone3')
        bone.head = context.scene.custom_props.talus_vec
        bone.tail = context.scene.custom_props.calcaneus_vec
        bone.parent = edit_bones['Bone1']

        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')

        self.report({'INFO'}, f"Parent setzen/ Knochen und Mesh verbinden!")
        return {"FINISHED"}


class VisualizeJoints(bpy.types.Operator):
    """Setzt an jede Joint Position in der Szene eine Sphere und ein Gizmo zum visualisieren"""
    bl_idname = "custom.visualize_joints"
    bl_label = "Adds Placeholders to show the Joints"

    # noinspection PyMethodMayBeStatic
    def execute(self, context):
        old_placeholder = find_by_name(context.scene, "Placeholder")
        bpy.ops.object.select_all(action="DESELECT")
        for obj in old_placeholder:
            obj.select_set(True)
        bpy.ops.object.delete()

        point_list = [context.scene.custom_props.cruris_vec, context.scene.custom_props.talus_vec,
                      context.scene.custom_props.calcaneus_vec, context.scene.custom_props.antetarsus_vec]

        scale_vec = mathutils.Vector((context.scene.custom_props.joint_size,
                                      context.scene.custom_props.joint_size,
                                      context.scene.custom_props.joint_size))

        for joints in point_list:
            bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=1.0, calc_uvs=True,
                                                 enter_editmode=False, align='WORLD', location=joints,
                                                 rotation=(0.0, 0.0, 0.0), scale=scale_vec)
            bpy.context.active_object.name = "Placeholder"
        return {"FINISHED"}


class jointGizmos(bpy.types.GizmoGroup):
    bl_label = "Test Gizmo"
    bl_idname = "OBJECT_GGT_gizmo_test"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_options = {'3D', 'PERSISTENT'}

    def __init__(self):
        self.giz1 = None
        self.giz2 = None
        self.giz3 = None
        self.giz4 = None

    @classmethod
    def poll(cls, context):
        ob = context.object
        return ob and ob.type == 'MESH' and "Placeholder" in ob.name

    # noinspection PyUnusedLocal
    def setup(self, context):
        giz_type_1 = self.gizmos.new("GIZMO_GT_cage_3d")
        giz_type_2 = self.gizmos.new("GIZMO_GT_cage_3d")
        giz_type_3 = self.gizmos.new("GIZMO_GT_cage_3d")
        giz_type_4 = self.gizmos.new("GIZMO_GT_cage_3d")

        giz_type_1.matrix_basis = bpy.data.objects["Placeholder"].matrix_world.normalized()
        giz_type_2.matrix_basis = bpy.data.objects["Placeholder.001"].matrix_world.normalized()
        giz_type_2.matrix_basis = bpy.data.objects["Placeholder.002"].matrix_world.normalized()
        giz_type_2.matrix_basis = bpy.data.objects["Placeholder.003"].matrix_world.normalized()

        self.giz1 = giz_type_1
        self.giz2 = giz_type_2
        self.giz3 = giz_type_3
        self.giz4 = giz_type_4

    def refresh(self, context):
        # Joint 1
        vec = mathutils.Vector((context.scene.custom_props.cruris_vec[0],
                                context.scene.custom_props.cruris_vec[1],
                                context.scene.custom_props.cruris_vec[2]))
        newMat = mathutils.Matrix()

        newMat[0][3] = vec[0]
        newMat[1][3] = vec[1]
        newMat[2][3] = vec[2]

        self.giz1.matrix_basis = newMat.normalized()

        # Joint 2
        vec = mathutils.Vector((context.scene.custom_props.talus_vec[0],
                                context.scene.custom_props.talus_vec[1],
                                context.scene.custom_props.talus_vec[2]))
        newMat[0][3] = vec[0]
        newMat[1][3] = vec[1]
        newMat[2][3] = vec[2]

        self.giz2.matrix_basis = newMat.normalized()

        # Joint 3
        vec = mathutils.Vector((context.scene.custom_props.antetarsus_vec[0],
                                context.scene.custom_props.antetarsus_vec[1],
                                context.scene.custom_props.antetarsus_vec[2]))
        newMat[0][3] = vec[0]
        newMat[1][3] = vec[1]
        newMat[2][3] = vec[2]

        self.giz3.matrix_basis = newMat.normalized()

        # Joint 4
        vec = mathutils.Vector((context.scene.custom_props.calcaneus_vec[0],
                                context.scene.custom_props.calcaneus_vec[1],
                                context.scene.custom_props.calcaneus_vec[2]))
        newMat[0][3] = vec[0]
        newMat[1][3] = vec[1]
        newMat[2][3] = vec[2]

        self.giz4.matrix_basis = newMat.normalized()


class SetPositionJoint1(bpy.types.Operator):
    bl_idname = 'custom.set_value_joint1'
    bl_label = "Sets the Vector Value for Joint 1"

    # noinspection PyMethodMayBeStatic
    def execute(self, context):
        context.scene.custom_props.cruris_vec = bpy.context.scene.cursor.location
        return {"FINISHED"}


class SetPositionJoint2(bpy.types.Operator):
    bl_idname = 'custom.set_value_joint2'
    bl_label = "Sets the Vector Value for Joint 2"

    # noinspection PyMethodMayBeStatic
    def execute(self, context):
        context.scene.custom_props.talus_vec = bpy.context.scene.cursor.location
        return {"FINISHED"}


class SetPositionJoint3(bpy.types.Operator):
    bl_idname = 'custom.set_value_joint3'
    bl_label = "Sets the Vector Value for Joint 3"

    # noinspection PyMethodMayBeStatic
    def execute(self, context):
        context.scene.custom_props.antetarsus_vec = bpy.context.scene.cursor.location
        return {"FINISHED"}


class SetPositionJoint4(bpy.types.Operator):
    bl_idname = 'custom.set_value_joint4'
    bl_label = "Sets the Vector Value for Joint 4"

    # noinspection PyMethodMayBeStatic
    def execute(self, context):
        context.scene.custom_props.calcaneus_vec = bpy.context.scene.cursor.location
        return {"FINISHED"}


def handlerFunc(scene):
    if scene.custom_props.follow_bool is True:
        targetPos = mathutils.Vector((0, 0, 0))
        offsetProp = scene.custom_props.cursor_offset
        offset = mathutils.Vector((offsetProp[0], offsetProp[1], offsetProp[2]))
        anchorPoints = find_by_name(scene, "AnchorPoint")
        for points in anchorPoints:
            targetPos += points.location
        bpy.context.scene.cursor.location = targetPos / len(anchorPoints) + offset


classes = (
    SpawnAnchorPoints,
    SpawnBones,
    CenterSelected,
    VisualizeJoints,
    jointGizmos,
    SetPositionJoint1,
    SetPositionJoint2,
    SetPositionJoint3,
    SetPositionJoint4,
    CustomPropertyGroup,
    CustomAddonPanel
)


def register():
    from bpy.utils import register_class

    for cls in classes:
        register_class(cls)

    bpy.types.Scene.custom_props = bpy.props.PointerProperty(type=CustomPropertyGroup)
    bpy.app.handlers.depsgraph_update_post.append(handlerFunc)


def unregister():
    from bpy.utils import unregister_class

    del bpy.types.Scene.custom_props

    for cls in classes:
        unregister_class(cls)

    bpy.app.handlers.depsgraph_update_post.remove(handlerFunc)


if __name__ == '__main__':
    register()
