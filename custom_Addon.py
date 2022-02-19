import bmesh
import bpy
import mathutils
import bpy_extras


bl_info = {
    "name": "Custom Addon",
    "description": "Custom Toolkit zum Riggen von Füßen",
    "author": "Marius Unger",
    "doc_url": "https://github.com/realAnzary/customBlenderAddon",
    "category": "Object",
    "location": "Porperties Panel",
    "warning": "",
    "version": (1, 0, 2),
    "blender": (3, 0, 0)
}


def find_by_name(scene, name):
    returnedObjects = []
    for objects in scene.objects:
        if name in objects.name:
            returnedObjects.append(objects)
    return returnedObjects


def add_angle_object(self, context):
    from mathutils import Vector

    scale_x = self.scale.x
    scale_y = self.scale.y
    scale_z = self.scale.z

    scale = 5

    verts = [Vector((0 * scale_x, .5 * scale_y, 0 * scale_z)),
             Vector((0 * scale_x, -.5 * scale_y, 0 * scale_z)),
             Vector((scale * .5 * scale_x, .5 * scale_y, 0 * scale_z)),
             Vector((scale * .5 * scale_x, -.5 * scale_y, 0 * scale_z)),
             Vector((0 * scale_x, .5 * scale_y, scale * 1 * scale_z)),
             Vector((0 * scale_x, -.5 * scale_y, scale * 1 * scale_z)),

             Vector((scale * .5 * scale_x, .5 * scale_y, .5 * scale_z)),
             Vector((scale * .5 * scale_x, -.5 * scale_y, .5 * scale_z)),
             Vector((.5 * scale_x, .5 * scale_y, scale * 1 * scale_z)),
             Vector((.5 * scale_x, -.5 * scale_y, scale * 1 * scale_z)),
             Vector((.5 * scale_x, .5 * scale_y, .5 * scale_z)),
             Vector((.5 * scale_x, -.5 * scale_y, .5 * scale_z))]

    edges = [[0, 1], [1, 3], [3, 2], [2, 0],
             [4, 5], [5, 9], [9, 8], [8, 4],
             [10, 11], [11, 7], [7, 6], [6, 10],
             [0, 4], [10, 8], [2, 6],
             [1, 5], [11, 9], [3, 7]]

    faces = [[1, 3, 7, 11, 9, 5],
             [0, 2, 6, 10, 8, 4],
             [0, 1, 3, 2],
             [10, 11, 7, 6],
             [4, 5, 9, 8],
             [6, 7, 3, 2],
             [8, 9, 11, 10],
             [0, 4, 5, 1]]

    mesh = bpy.data.meshes.new(name="New Object")
    mesh.from_pydata(verts, edges, faces)
    bpy_extras.object_utils.object_data_add(context, mesh, operator=self)


def add_object_button(self, context):
    self.layout.operator(
        AddAngleObject.bl_idname,
        text="Add Angle Object",
        icon="PLUGIN"
    )


class CustomPropertyGroup(bpy.types.PropertyGroup):
    # UI-Panel
    expanded: bpy.props.BoolProperty(default=False)
    # Props für 3D-Cursor Position
    follow_bool: bpy.props.BoolProperty()
    cursor_offset: bpy.props.FloatVectorProperty()
    # Punkte im Fuß
    cruris_vec: bpy.props.FloatVectorProperty()
    talus_vec: bpy.props.FloatVectorProperty()
    antetarsus_vec: bpy.props.FloatVectorProperty()
    calcaneus_vec: bpy.props.FloatVectorProperty()

    gizmo_visibility: bpy.props.BoolProperty(default=False)


class CustomAddonPanel(bpy.types.Panel):
    bl_idname = "OBJECT_PT_custom_panel"
    bl_region_type = 'WINDOW'
    bl_space_type = 'PROPERTIES'
    bl_label = 'Addon Rigging Tools'

    def draw(self, context):
        layout = self.layout
        layout.label(text="Allgemeine Tools")
        layout.operator('custom.center_selected', text="Objekt zentrieren")
        layout.operator('custom.add_angle_object', text="90° Objekt")
        layout.label(text="3D Cursor Tools")
        layout.operator('custom.spawn_anchor', text="Ankerpunkte setzen")
        layout.prop(context.scene.custom_props, "follow_bool", text="3D Cursor zentrieren")
        layout.prop(context.scene.custom_props, "cursor_offset", text="Cursor Offset")
        layout.label(text="Rigging Tools")
        layout.operator('custom.spawn_bones', text="Armature & Knochen hinzufügen")
        layout.prop(context.scene.custom_props, "gizmo_visibility", text="Gizmos anzeigen")
        layout.label(text="Joint Positionen setzen")
        layout.operator('custom.set_value_joint1', text="Cruris / Joint1 Position setzen")
        layout.operator('custom.set_value_joint2', text="Talus / Joint2 Position setzen")
        layout.operator('custom.set_value_joint3', text="Antetarsus / Joint3 Position setzen")
        layout.operator('custom.set_value_joint4', text="Calcaneus / Joint4 Position setzen")

        box = layout.box()
        row = box.row()
        row.prop(context.scene.custom_props, "expanded",
                 icon="TRIA_DOWN" if context.scene.custom_props.expanded else "TRIA_RIGHT",
                 icon_only=True, emboss=False
                 )
        row.label(text="Joint Informationen")

        if context.scene.custom_props.expanded:
            row = box.row()
            row.prop(context.scene.custom_props, "cruris_vec")
            row = box.row()
            row.prop(context.scene.custom_props, "talus_vec")
            row = box.row()
            row.prop(context.scene.custom_props, "antetarsus_vec")
            row = box.row()
            row.prop(context.scene.custom_props, "calcaneus_vec")
        # col = layout.column(heading="Header")


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


class AddAngleObject(bpy.types.Operator, bpy_extras.object_utils.AddObjectHelper):
    """Create a new Mesh Object"""
    bl_idname = "custom.add_angle_object"
    bl_label = "Add Mesh Object"
    bl_options = {'REGISTER', 'UNDO'}

    scale: bpy.props.FloatVectorProperty(
        name="scale",
        default=(1.0, 1.0, 1.0),
        subtype='TRANSLATION',
        description="scaling",
    )

    def execute(self, context):
        add_angle_object(self, context)

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
        mesh = bmesh.new()
        mesh = bmesh.from_edit_mesh(selected_obj)
        selectedVerts = [verts for verts in mesh.verts if verts.select]
        for selected in range(0, len(selectedVerts)):
            object = bpy.data.objects.new(name_list[selected], object_data=None)
            bpy.context.scene.collection.objects.link(object)

            object.location = selectedVerts[selected].co
            object.empty_display_size = 2
            object.empty_display_type = "PLAIN_AXES"

        return {"FINISHED"}


class SpawnBones(bpy.types.Operator):
    """Vebindet alle Joints in der Szene mit Knochen"""
    bl_idname = 'custom.spawn_bones'
    bl_label = "Spawn Bones in corresponding Position"
    bl_options = {'REGISTER', 'UNDO'}

    # noinspection PyMethodMayBeStatic
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
        return {"FINISHED"}


shape_Line_Cube = ((0.5, 0.5, -0.5), (-0.5, 0.5, -0.5),
                   (-0.5, 0.5, -0.5), (-0.5, -0.5, -0.5),
                   (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5),
                   (0.5, -0.5, -0.5), (0.5, 0.5, -0.5),

                   (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5),
                   (-0.5, 0.5, 0.5), (-0.5, -0.5, 0.5),
                   (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5),
                   (0.5, -0.5, 0.5), (0.5, 0.5, 0.5),

                   (0.5, 0.5, -0.5), (0.5, 0.5, 0.5),
                   (-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5),
                   (-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5),
                   (0.5, -0.5, -0.5), (0.5, -0.5, 0.5))

cube_scale = 5.0

shape_Line_Cross = ((0, 0, 0), (0, 0, 1),
                    (0, 0, 0), (0, 0, -1),
                    (0, 0, 0), (0, 1, 0),
                    (0, 0, 0), (0, -1, 0),
                    (0, 0, 0), (1, 0, 0),
                    (0, 0, 0), (-1, 0, 0))


cube_coords = [(cube_scale * -.5, cube_scale * .5, cube_scale * -.5),
               (cube_scale * -.5, cube_scale * -.5, cube_scale * -.5),
               (cube_scale * .5, cube_scale * -.5, cube_scale * -.5),
               (cube_scale * .5, cube_scale * .5, cube_scale * -.5),
               (cube_scale * -.5, cube_scale * .5, cube_scale * .5),
               (cube_scale * -.5, cube_scale * -.5, cube_scale * .5),
               (cube_scale * .5, cube_scale * -.5, cube_scale * .5),
               (cube_scale * .5, cube_scale * .5, cube_scale * .5)]

shape_Tris_Cube = (cube_coords[0], cube_coords[1], cube_coords[3],
                   cube_coords[1], cube_coords[2], cube_coords[3],  # Face Bot

                   cube_coords[0], cube_coords[4], cube_coords[7],
                   cube_coords[7], cube_coords[3], cube_coords[0],  # Face Right

                   cube_coords[4], cube_coords[5], cube_coords[1],
                   cube_coords[4], cube_coords[1], cube_coords[0],  # Face Back

                   cube_coords[5], cube_coords[6], cube_coords[2],
                   cube_coords[5], cube_coords[2], cube_coords[1],  # Face Left

                   cube_coords[7], cube_coords[6], cube_coords[2],
                   cube_coords[7], cube_coords[2], cube_coords[3],  # Face Front

                   cube_coords[4], cube_coords[5], cube_coords[6],
                   cube_coords[4], cube_coords[6], cube_coords[7]  # Face Front
                   )


class GizmoShape_LineCube(bpy.types.Gizmo):
    bl_idname = "GizmoShape_Cube"

    __slots__ = ["shape"]

    def __init__(self):
        self.shape = None

    # noinspection PyUnusedLocal
    def draw(self, context):
        self.draw_custom_shape(self.shape)

    # noinspection PyUnusedLocal
    def draw_select(self, context, select_id):
        self.draw_custom_shape(self.shape, select_id=select_id)

    def setup(self):
        if not hasattr(self, "custom_shape"):
            self.shape = self.new_custom_shape('LINES', shape_Line_Cube)
            # self.shape = self.new_custom_shape('TRIS', shape_Tris_Cube)


class jointGizmos(bpy.types.GizmoGroup):
    bl_label = "Joint Gizmogroup"
    bl_idname = "OBJECT_GGT_gizmo_joints"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_options = {'3D', 'PERSISTENT'}

    __slots__ = ['giz1', 'giz2', 'giz3', 'giz4']

    def __init__(self):
        self.giz1 = None
        self.giz2 = None
        self.giz3 = None
        self.giz4 = None

    # noinspection PyUnusedLocal
    def setup(self, context):
        giz_type_1 = self.gizmos.new(GizmoShape_LineCube.bl_idname)
        giz_type_2 = self.gizmos.new(GizmoShape_LineCube.bl_idname)
        giz_type_3 = self.gizmos.new(GizmoShape_LineCube.bl_idname)
        giz_type_4 = self.gizmos.new(GizmoShape_LineCube.bl_idname)

        gizmo_matrix_basis = mathutils.Matrix.Translation((0, 0, 0)).normalized()
        gizmo_scale_basis = 1
        gizmo_color = 1.0, 0.5, 0
        gizmo_color_highlight = 1, 1, 1
        gizmo_line_width = 1.0

        giz_type_1.matrix_basis = gizmo_matrix_basis
        giz_type_2.matrix_basis = gizmo_matrix_basis
        giz_type_3.matrix_basis = gizmo_matrix_basis
        giz_type_4.matrix_basis = gizmo_matrix_basis

        giz_type_1.color = gizmo_color
        giz_type_2.color = gizmo_color
        giz_type_3.color = gizmo_color
        giz_type_4.color = gizmo_color

        giz_type_1.color_highlight = gizmo_color_highlight
        giz_type_2.color_highlight = gizmo_color_highlight
        giz_type_3.color_highlight = gizmo_color_highlight
        giz_type_4.color_highlight = gizmo_color_highlight

        giz_type_1.scale_basis = gizmo_scale_basis
        giz_type_2.scale_basis = gizmo_scale_basis
        giz_type_3.scale_basis = gizmo_scale_basis
        giz_type_4.scale_basis = gizmo_scale_basis

        giz_type_1.line_width = gizmo_line_width
        giz_type_2.line_width = gizmo_line_width
        giz_type_3.line_width = gizmo_line_width
        giz_type_4.line_width = gizmo_line_width

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
        self.giz1.hide = not context.scene.custom_props.gizmo_visibility

        # Joint 2
        vec = mathutils.Vector((context.scene.custom_props.talus_vec[0],
                                context.scene.custom_props.talus_vec[1],
                                context.scene.custom_props.talus_vec[2]))
        newMat[0][3] = vec[0]
        newMat[1][3] = vec[1]
        newMat[2][3] = vec[2]

        self.giz2.matrix_basis = newMat.normalized()
        self.giz2.hide = not context.scene.custom_props.gizmo_visibility

        # Joint 3
        vec = mathutils.Vector((context.scene.custom_props.antetarsus_vec[0],
                                context.scene.custom_props.antetarsus_vec[1],
                                context.scene.custom_props.antetarsus_vec[2]))
        newMat[0][3] = vec[0]
        newMat[1][3] = vec[1]
        newMat[2][3] = vec[2]

        self.giz3.matrix_basis = newMat.normalized()
        self.giz3.hide = not context.scene.custom_props.gizmo_visibility

        # Joint 4
        vec = mathutils.Vector((context.scene.custom_props.calcaneus_vec[0],
                                context.scene.custom_props.calcaneus_vec[1],
                                context.scene.custom_props.calcaneus_vec[2]))
        newMat[0][3] = vec[0]
        newMat[1][3] = vec[1]
        newMat[2][3] = vec[2]

        self.giz4.matrix_basis = newMat.normalized()
        self.giz4.hide = not context.scene.custom_props.gizmo_visibility


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
    AddAngleObject,
    SpawnAnchorPoints,
    SpawnBones,
    CenterSelected,
    GizmoShape_LineCube,
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
    bpy.types.VIEW3D_MT_mesh_add.append(add_object_button)
    bpy.types.Scene.custom_props = bpy.props.PointerProperty(type=CustomPropertyGroup)
    bpy.app.handlers.depsgraph_update_post.append(handlerFunc)


def unregister():
    from bpy.utils import unregister_class

    del bpy.types.Scene.custom_props

    for cls in classes:
        unregister_class(cls)

    bpy.types.VIEW3D_MT_mesh_add.remove(add_object_button)
    bpy.app.handlers.depsgraph_update_post.remove(handlerFunc)


if __name__ == '__main__':
    register()
