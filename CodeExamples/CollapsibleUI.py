import bpy


class HelloWorldPanel(bpy.types.Panel):
    """Creates a Panel in the Object properties window"""
    bl_label = "Hello World Panel"
    bl_idname = "OBJECT_PT_hello"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "object"

    def draw(self, context):
        layout = self.layout
        obj = context.object

        box = layout.box()

        row = box.row()
        row.prop(obj, "expanded",
            icon="TRIA_DOWN" if obj.expanded else "TRIA_RIGHT",
            icon_only=True, emboss=False
        )
        row.label(text="Active object is: " + obj.name)

        if obj.expanded:
            row = box.row()
            row.prop(obj, "name")

            row = box.row()
            row.label(text="Hello world!", icon='WORLD_DATA')

            row = box.row()
            row.operator("mesh.primitive_cube_add")


def register():
    bpy.utils.register_class(HelloWorldPanel)
    bpy.types.Object.expanded = bpy.props.BoolProperty(default=True)


def unregister():
    bpy.utils.unregister_class(HelloWorldPanel)
    del bpy.types.Object.expanded


if __name__ == "__main__":
    register()