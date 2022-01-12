import bmesh
import bpy
import mathutils


class CustomPropertyGroup(bpy.types.PropertyGroup):
    # Bool für 3D-Crusor Position
    follow_bool: bpy.props.BoolProperty(name="center_3d_cursor")
    # Punkte im Fuß
    cruris_vec: bpy.props.FloatVectorProperty(name="joint_top")
    talus_vec: bpy.props.FloatVectorProperty(name="joint_middle")
    antetarsus_vec: bpy.props.FloatVectorProperty(name="joint_front")
    calcaneus_vec: bpy.props.FloatVectorProperty(name="joint_back")


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
        layout.label(text="Rigging Tools")
        layout.operator('custom.spawn_bones', text="Armature & Knochen hinzufügen")
        layout.operator('custom.visualize_joints', text="Joints anzeigen")
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


class SpawnAnchorPoints(bpy.types.Operator):
    """Fügt der Szene Objekte hinzu; Dienen als Anhaltspunkte um 3D Cursor zu zentrieren;
Muss im Edit-Mode benutzt werden und platziert für jeden asugewählten Vertex ein Ankerpunkt"""
    bl_idname = 'custom.spawn_anchor'
    bl_label = "Adds Anchor Object to the Scene"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object.select_get() and context.mode == "EDIT_MESH"

    def execute(self, context):
        name_list = ["AnchorPoint.{:03d}".format(c + 1) for c in range(0, 10)]

        selected_obj = context.object.data
        mesh = bmesh.new()
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
    bl_label = "Spawn Bones in corresbonding Position"
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


class SetPositionJoint1(bpy.types.Operator):
    bl_idname = 'custom.set_value_joint1'
    bl_label = "Sets the Vector Value for Joint 1"

    def execute(self, context):
        context.scene.custom_props.cruris_vec = bpy.context.scene.cursor.location
        return {"FINISHED"}


class SetPositionJoint2(bpy.types.Operator):
    bl_idname = 'custom.set_value_joint2'
    bl_label = "Sets the Vector Value for Joint 2"

    def execute(self, context):
        context.scene.custom_props.talus_vec = bpy.context.scene.cursor.location
        return {"FINISHED"}


class SetPositionJoint3(bpy.types.Operator):
    bl_idname = 'custom.set_value_joint3'
    bl_label = "Sets the Vector Value for Joint 3"

    def execute(self, context):
        context.scene.custom_props.antetarsus_vec = bpy.context.scene.cursor.location
        return {"FINISHED"}


class SetPositionJoint4(bpy.types.Operator):
    bl_idname = 'custom.set_value_joint4'
    bl_label = "Sets the Vector Value for Joint 4"

    def execute(self, context):
        context.scene.custom_props.calcaneus_vec = bpy.context.scene.cursor.location
        return {"FINISHED"}


class CenterSelected(bpy.types.Operator):
    """Zentriert ein ausgewähltes Objekt und den 3D Cursor"""
    bl_idname = "custom.center_selected"  # UID/Methodname
    bl_label = "Center selected Object"  # Name in UI
    bl_options = {'REGISTER', 'UNDO'}  # Enable undo

    def execute(self, context):
        bpy.context.area.type = 'VIEW_3D'
        bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
        bpy.ops.view3d.snap_selected_to_cursor()
        bpy.context.area.ui_type = 'PROPERTIES'
        return {'FINISHED'}


class VisualizeJoints(bpy.types.Operator):
    """Setzt an jede Joint Position in der Szene eine Sphere und ein Gizmo zum visualisieren"""
    bl_idname = "custom.visualize_joints"
    bl_label = "Adds Placeholders to show the Joints"

    def execute(self, context):
        old_placeholder = find_multiple_by_name(context.scene, "Placeholder")
        bpy.ops.object.select_all(action="DESELECT")
        for obj in old_placeholder:
            obj.select_set(True)
        bpy.ops.object.delete()

        point_list = [context.scene.custom_props.cruris_vec, context.scene.custom_props.talus_vec,
                      context.scene.custom_props.calcaneus_vec, context.scene.custom_props.antetarsus_vec]

        for joints in point_list:
            bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1.0, calc_uvs=True,
                                                 enter_editmode=False, align='WORLD', location=joints,
                                                 rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0))
            bpy.context.active_object.name = "Placeholder"
        return {"FINISHED"}


class jointGizmos(bpy.types.GizmoGroup):
    bl_label = "Test Gizmo"
    bl_idname = "OBJECT_GGT_gizmo_test"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_options = {'3D', 'PERSISTENT'}

    @classmethod
    def poll(cls, context):
        ob = context.object
        return ob and ob.type == 'MESH' and ("Placeholder" in ob.name)

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
        newMat = mathutils.Matrix()

        newMat[0][3] = vec[0]
        newMat[1][3] = vec[1]
        newMat[2][3] = vec[2]

        self.giz2.matrix_basis = newMat.normalized()

        # Joint 3
        vec = mathutils.Vector((context.scene.custom_props.antetarsus_vec[0],
                                context.scene.custom_props.antetarsus_vec[1],
                                context.scene.custom_props.antetarsus_vec[2]))
        newMat = mathutils.Matrix()

        newMat[0][3] = vec[0]
        newMat[1][3] = vec[1]
        newMat[2][3] = vec[2]

        self.giz3.matrix_basis = newMat.normalized()

        # Joint 4
        # Joint 3
        vec = mathutils.Vector((context.scene.custom_props.calcaneus_vec[0],
                                context.scene.custom_props.calcaneus_vec[1],
                                context.scene.custom_props.calcaneus_vec[2]))
        newMat = mathutils.Matrix()

        newMat[0][3] = vec[0]
        newMat[1][3] = vec[1]
        newMat[2][3] = vec[2]

        self.giz4.matrix_basis = newMat.normalized()


bl_info = {
    "name": "Custom Addon",
    "description": "Custom Toolkit zum Riggen",
    "author": "Marius Unger",
    "category": "Object",
    "location": "Porperties Panel",
    "warning": "",
    "version": (1, 0, 0),
    "blender": (3, 0, 0)
}


def find_multiple_by_name(scene, name):
    returned_objects = []
    for objects in scene.objects:
        if objects.name.isalpha():
            truncated_name = objects.name
        else:
            length = len(objects.name)
            truncated_name = objects.name[0:length - 4]
        if truncated_name == name:
            returned_objects.append(objects)
    return returned_objects


def handlerFunc(scene):
    if scene.custom_props.follow_bool is True:
        targetPos = mathutils.Vector((0, 0, 0))
        anchorPoints = find_multiple_by_name(scene, "AnchorPoint")
        for points in anchorPoints:
            targetPos += points.location
        bpy.context.scene.cursor.location = targetPos / len(anchorPoints)


def register():
    bpy.utils.register_class(SpawnAnchorPoints)
    bpy.utils.register_class(SpawnBones)
    bpy.utils.register_class(CenterSelected)
    bpy.utils.register_class(VisualizeJoints)
    bpy.utils.register_class(jointGizmos)

    bpy.utils.register_class(SetPositionJoint1)
    bpy.utils.register_class(SetPositionJoint2)
    bpy.utils.register_class(SetPositionJoint3)
    bpy.utils.register_class(SetPositionJoint4)

    bpy.utils.register_class(CustomPropertyGroup)
    bpy.types.Scene.custom_props = bpy.props.PointerProperty(type=CustomPropertyGroup)

    bpy.utils.register_class(CustomAddonPanel)

    bpy.app.handlers.depsgraph_update_post.append(handlerFunc)


def unregister():
    bpy.utils.unregister_class(SpawnAnchorPoints)
    bpy.utils.unregister_class(SpawnBones)
    bpy.utils.unregister_class(CenterSelected)
    bpy.utils.unregister_class(VisualizeJoints)
    bpy.utils.unregister_class(jointGizmos)

    bpy.utils.unregister_class(SetPositionJoint1)
    bpy.utils.unregister_class(SetPositionJoint2)
    bpy.utils.unregister_class(SetPositionJoint3)
    bpy.utils.unregister_class(SetPositionJoint4)

    del bpy.types.Scene.custom_props
    bpy.utils.unregister_class(CustomPropertyGroup)

    bpy.utils.unregister_class(CustomAddonPanel)

    bpy.app.handlers.depsgraph_update_post.remove(handlerFunc)


if __name__ == '__main__':
    register()
