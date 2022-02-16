import bpy

custom_shape_verts = ((0.0, 0.0, 0.0), (4 * .5, 0.0, 4 * .5))


class ShapeWidget(bpy.types.Gizmo):
    bl_idname = "custom_shape"

    __slots__ = ["shape"]

    def draw(self, context):
        self.draw_custom_shape(self.shape)

    def draw_select(self, context, select_id):
        self.draw_custom_shape(self.shape, select_id=select_id)

    def setup(self):
        if not hasattr(self, "custom_shape"):
            self.shape = self.new_custom_shape('LINES', custom_shape_verts)


class MyLightWidgetGroup(bpy.types.GizmoGroup):
    bl_idname = "OBJECT_GGT_light_test"
    bl_label = "Test Light Widget"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_options = {'3D', 'PERSISTENT'}

    @classmethod
    def poll(cls, context):
        ob = context.object
        return ob and ob.name == "Cube"

    def setup(self, context):
        ob = context.object
        gz = self.gizmos.new(ShapeWidget.bl_idname)
        gz.matrix_basis = ob.matrix_world.normalized()

        gz.color = 1.0, 0.5, 0.0
        gz.alpha = 0.5

        gz.color_highlight = 1.0, 0.5, 1.0
        gz.alpha_highlight = 0.5

    def refresh(self, context):
        pass


bpy.utils.register_class(MyLightWidgetGroup)
bpy.utils.register_class(ShapeWidget)
