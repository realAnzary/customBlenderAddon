# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

bl_info = {
	'name': 'Random Flow',
	'author': 'Ian Lloyd Dela Cruz',
	'version': (1, 6, 0),
	'blender': (3, 0, 0),
	'location': '3d View > Tool shelf',
	'description': 'Collection of random greebling functionalities',
	'warning': '',
	'wiki_url': '',
	'tracker_url': '',
	'category': 'Mesh'}

import bpy
import bgl
import blf
import gpu
import os
from gpu_extras.batch import batch_for_shader
import colorsys
import numpy as np
from random import random, randint, sample, uniform, choice, choices, seed, shuffle, triangular
from collections import Counter
import bmesh
import math
from math import *
import mathutils
from mathutils import *
from mathutils.geometry import intersect_line_plane
from mathutils.bvhtree import BVHTree
from itertools import chain
from bpy.props import *
from bpy_extras import view3d_utils
import rna_keymap_ui
from bpy.types import (
		AddonPreferences,
		PropertyGroup,
		Operator,
		Menu,
		Panel,
		)

def auto_smooth(obj, deg=radians(30), set=True):

	obj.data.use_auto_smooth = set
	obj.data.auto_smooth_angle = deg

	mesh = obj.data
	if mesh.is_editmode:
		bm = bmesh.from_edit_mesh(mesh)
	else:
		bm = bmesh.new()
		bm.from_mesh(mesh)

	for f in bm.faces:
		f.smooth = set

	if mesh.is_editmode:
		bmesh.update_edit_mesh(mesh)
	else:
		bm.to_mesh(mesh)
		mesh.update()

def get_evaluated_mesh(context, obj):

	dg = context.evaluated_depsgraph_get()
	obj_eval = obj.evaluated_get(dg)
	mesh_from_eval = obj_eval.to_mesh()

	return mesh_from_eval

def create_temp_obj(context, name):

	data = bpy.data.meshes.new(name)
	obj = bpy.data.objects.new(name, data)

	return obj

def clear_customdata(obj):

	mesh = obj.data

	if mesh.edges:
		bvl_wght = sum(e.bevel_weight for e in mesh.edges)/len(mesh.edges)
		if bvl_wght == 0: mesh.use_customdata_edge_bevel = False

def assign_vgroup(obj, bm, vlist, vgname, use_select=True):

	vg = obj.vertex_groups.get(vgname)
	if vg: obj.vertex_groups.remove(vg)

	vg = obj.vertex_groups.get(vgname) or obj.vertex_groups.new(name=vgname)
	idx = vg.index

	deform_layer = bm.verts.layers.deform.active or bm.verts.layers.deform.new()

	for v in vlist:
		if use_select:
			if v.select: v[deform_layer][idx] = 1.0
		else: v[deform_layer][idx] = 1.0

temp_mats_rflow = ["rflow_temp_mat1", "rflow_temp_mat2"]

def assign_temp_mats(mesh, flist):

	mats = mesh.materials
	for i, m in enumerate(temp_mats_rflow):
		mat = bpy.data.materials.get(m) or bpy.data.materials.new(name=m)
		if mat:
			mats.append(mat)
			if i > 0:
				mat.diffuse_color = (0.25,0.25,0.25,1)
				for f in flist:
					f.material_index = i

def assign_mat(self, source, target, mat_index):

	idx = mat_index
	mats = source.data.materials

	def append_mat(mat):

		if mat:
			if not mat.name in temp_mats_rflow \
				or bpy.context.scene.rflow_props.copy_temp_mats:
				target.data.materials.append(mat)

	if mats:
		if idx > -1:
			if idx <= (len(mats) - 1):
				append_mat(source.data.materials[idx])
			else:
				self.report({'WARNING'}, "Material not found.")
		else:
			append_mat(source.active_material)

def random_walk(
			bm,
			idx,
			size,
			snum,
			sampling='RECURSIVE',
			path='NONE',
			cut_threshold=radians(30),
			wrap_angle=True
			):

	split_edg = []
	cells = []
	counter = 0

	bm.faces.ensure_lookup_table()
	while idx:
		seed(snum + counter)
		x = choice(list(idx))
		idx.remove(x)

		f = bm.faces[x]
		face_cell = [f]
		edge_cell = list(f.edges)
		start_v = [v for v in f.verts]

		walk = 0

		def add_cells(f):

			idx.remove(f.index)
			face_cell.append(f)
			edge_cell.extend(f.edges)

		while walk < size:
			if sampling != 'RADIAL':
				seed(snum + counter)
				if path != 'NONE':
					link_edges = {e: e.calc_length() for e in f.edges}
					edge = sample(list(link_edges.keys()), len(link_edges.keys()))
					edge_length = list(link_edges.values())
				else:
					edge = sample(list(f.edges), len(f.edges))
					edge_length = [0]
				for e in edge:
					length = max(edge_length) if path == 'LONGEST' \
						else min(edge_length) if path == 'SHORTEST' else 0
					if e.calc_length() != length:
						f = next((i for i in e.link_faces if i.index in idx), None)
						if f:
							add_cells(f)
							walk += 1
							break
				else:
					if sampling == 'RECURSIVE':
						for e in edge_cell:
							walk += 1
							f = next((i for i in e.link_faces if i.index in idx), None)
							if f:
								add_cells(f)
								break
						else: break
					else:
						sample_edge = choice(list(edge_cell))
						f = next((i for i in sample_edge.link_faces if i.index in idx), None)
						if f:
							add_cells(f)
							walk += 1
						else:
							break
			else:
				new_v = []
				for v in start_v:
					for lf in v.link_faces:
						if lf.index in idx:
							add_cells(lf)
							new_v.extend(lf.verts)
							walk += 1

				if new_v:
					start_v = [v for v in new_v]
				else: break

		if wrap_angle:
			edges = set(edge_cell)
			faces = set(face_cell)

			for e in edges:
				check = all(i in faces for i in e.link_faces)
				if not check:
					edge_angle = e.calc_face_angle(None)
					if edge_angle and edge_angle >= cut_threshold:
						for lf in e.link_faces:
							if lf.index in idx \
								and not lf in faces: add_cells(lf)

		cells.append(face_cell)

		edge_cell = undupe(edge_cell)
		for e in edge_cell:
			check = all(i in face_cell for i in e.link_faces)
			if not check and \
				not e in split_edg: split_edg.append(e)

		counter += 1

	return split_edg, cells

def get_tris(bm, faces, perc, seedr):

	seed(seedr)
	flist = sample(list(faces), len(faces))
	tri_perc = int(len(flist) * (perc/100))

	return flist[:tri_perc]

def get_linked_faces(faces):

	listf = set(faces)
	linked_faces = set()

	if listf != None:
		while listf:
			traversal_stack = [listf.pop()]

			while len(traversal_stack) > 0:
				f_curr = traversal_stack.pop()
				linked_faces.add(f_curr)

				for e in f_curr.edges:
					if e.is_contiguous:
						for f_linked in e.link_faces:
							if f_linked not in linked_faces:
								traversal_stack.append(f_linked)
								if f_linked in listf: listf.remove(f_linked)

	return linked_faces

def get_islands(obj, bm, use_bm=True):

	if use_bm:
		paths={v.index:set() for v in bm.verts}
		bm.verts.ensure_lookup_table()
		for e in bm.edges:
			paths[e.verts[0].index].add(e.verts[1].index)
			paths[e.verts[1].index].add(e.verts[0].index)
	else:
		obj.update_from_editmode()
		mesh = obj.data

		paths={v.index:set() for v in mesh.vertices}
		for e in mesh.edges:
			paths[e.vertices[0]].add(e.vertices[1])
			paths[e.vertices[1]].add(e.vertices[0])

	lparts=[]

	while True:
		try:
			i = next(iter(paths.keys()))
		except StopIteration: break

		lpart={i}
		cur={i}

		while True:
			eligible = {sc for sc in cur if sc in paths}
			if not eligible:
				break
			cur = {ve for sc in eligible for ve in paths[sc]}
			lpart.update(cur)
			for key in eligible: paths.pop(key)

		lparts.append(lpart)

	return lparts

def clip_center(bm, obj, dist=0.001, axis=[True, True, True], range=2, smooth=0, smooth_axis=[True, True, True]):

	mirror = next((m for m in obj.modifiers if m.type == 'MIRROR'), None)
	if mirror:
		axis_dir = ["x","y","z"]
		for v in bm.verts:
			for i, n in enumerate(axis_dir):
				if mirror.use_axis[i] and axis[i]:
					if -dist <= v.co[i] <= dist:
						if smooth and smooth_axis[i]:
							if len(v.link_faces) > range:
								norms  = [f.normal for f in v.link_faces if not any(len(v.link_faces) < (range+1) for v in f.verts)]
								if norms:
									fn = sum(norms, Vector()) / len(norms)
									v.co -= smooth * fn
						setattr(v.co, n, 0)

def get_axis_faces(bm, obj, dist=1e-4):

	mirror = next((m for m in obj.modifiers if m.type == 'MIRROR'), None)
	faces = set()
	if mirror:
		for f in bm.faces:
			for i in range(3):
				if mirror.use_axis[i]:
					if -dist <= f.calc_center_median()[i] <= dist: faces.add(f)

	return list(faces)

def remove_axis_faces(bm, obj):

	axisf = get_axis_faces(bm, obj)
	bmesh.ops.delete(bm, geom=axisf, context='FACES')

def get_singles(verts):

	singles = []
	for v in verts:
		if len(v.link_edges) == 2:
			if v.is_boundary:
				direction = [(e.verts[1].co - e.verts[0].co) for e in v.link_edges]
				v1 = direction[0]
				v2 = direction[1]
				a1 = v1.angle(v2)
				if a1 > pi * 0.5:
					a1 = pi - a1
				if degrees(a1) == 0: singles.append(v)
			else:
				singles.append(v)

	return singles

def bisect_symmetry(bm, obj):

	mirror = next((m for m in obj.modifiers if m.type == 'MIRROR'), None)
	if mirror:
		mesh = obj.data
		vertices = np.empty((len(mesh.vertices), 3), 'f')
		mesh.vertices.foreach_get("co", np.reshape(vertices, len(mesh.vertices) * 3))
		origin = sum([Vector(co) for co in vertices], Vector()) / len(vertices)

		pivot = Vector()
		axes = [x for x in mirror.use_axis]
		axis = [Vector((1,0,0)), Vector((0,1,0)), Vector((0,0,1))]

		x_dir = axis[0] if origin.x > 0 else -axis[0]
		y_dir = axis[1] if origin.y > 0 else -axis[1]
		z_dir = axis[2] if origin.z > 0 else -axis[2]

		axis_dir = [x_dir if axes[0] else None, \
			y_dir if axes[1] else None, \
			z_dir if axes[2] else None]

		for n in axis_dir:
			if n:
				split = bm.verts[:] + bm.edges[:] + bm.faces[:]
				bmesh.ops.bisect_plane(
					bm,
					geom        = split,
					dist        = 0.0001,
					plane_co    = pivot,
					plane_no    = n,
					clear_inner = True,
					clear_outer = False
					)

def copy_modifiers(objs, mod_types=[]):

	sce = bpy.context.scene
	rf_props = sce.rflow_props

	orig_obj = objs[0]
	selected_objects = [o for o in objs if o != orig_obj]

	if rf_props.all_mods: mod_types.clear()

	def copy_mod_settings(obj, mSrc):

		mDst = obj.modifiers.get(mSrc.name, None) or \
			obj.modifiers.new(mSrc.name, mSrc.type)

		properties = [p.identifier for p in mSrc.bl_rna.properties
					  if not p.is_readonly]

		for prop in properties:
			setattr(mDst, prop, getattr(mSrc, prop))

	for obj in selected_objects:
		for mSrc in orig_obj.modifiers:
			if not mod_types:
				try:
					copy_mod_settings(obj, mSrc)
				except: pass
			else:
				if mSrc.type in mod_types:
					copy_mod_settings(obj, mSrc)

def remove_obj(obj):

	sce = bpy.context.scene

	in_master = True
	for c in bpy.data.collections:
		if obj.name in c.objects:
			c.objects.unlink(obj)
			in_master = False
			break

	if in_master:
		if obj.name in sce.collection.objects:
			sce.collection.objects.unlink(obj)

	bpy.data.objects.remove(obj)

def move_center_origin(origin, obj):

	pivot = obj.matrix_world.inverted() @ origin
	obj.data.transform(Matrix.Translation(-pivot))
	obj.matrix_world.translation = origin

def select_isolate(context, obj, obj_list):

	for o in obj_list:
		if o != obj:
			o.select_set(False)
		else:
			o.select_set(True)
			context.view_layer.objects.active = o

def copy_rotation(source_obj, obj):

	mat_source = source_obj.rotation_euler.to_matrix()
	mat_source.invert()
	mat_obj = obj.rotation_euler.to_matrix()

	if obj.type == 'MESH':
		mat = mat_source @ mat_obj
		for v in obj.data.vertices:
			vec = mat @ v.co
			v.co = vec

		obj.rotation_euler = source_obj.rotation_euler

def delta_increment(event, x, y, dim):

	v1 = abs(x - y)
	p =  dim * (0.001 * (0.1 if event.shift else 1.0))
	v2 = v1 * p

	return v2

def local_center(obj):

	mesh = obj.data
	center = sum([v.co for v in mesh.vertices], Vector()) / len(mesh.vertices)

	return center

def get_delta(context, event, obj, center):

	region = context.region
	pivot = Vector((region.width / 2, region.height / 2))

	if obj.data.polygons:
		pivot = view3d_utils.location_3d_to_region_2d(context.region, \
										   context.space_data.region_3d, \
										   center.xyz)

		if pivot is None: pivot = Vector((region.width / 2, region.height / 2))

	curr_mouse = Vector((event.mouse_region_x, event.mouse_region_y))
	prev_mouse = Vector((event.mouse_prev_x - context.region.x,
		event.mouse_prev_y - context.region.y))

	delta_x = (pivot - prev_mouse).length
	delta_y = (pivot - curr_mouse).length

	return delta_x, delta_y

def copy_loc_rot(obj, src_obj):

	loc = src_obj.matrix_world.translation
	pivot = obj.matrix_world.inverted() @ loc
	obj.data.transform(Matrix.Translation(pivot))

	rot_basis = src_obj.rotation_euler.to_matrix().to_4x4()
	rot_basis.invert()
	rot_obj = obj.rotation_euler.to_matrix().to_4x4()

	rot = (Matrix.Translation(loc) @
		rot_basis @
		rot_obj @
		Matrix.Translation(-loc))

	for v in obj.data.vertices:
		vec = rot.inverted() @ v.co
		v.co = vec

def v3d_to_v2d(context, points):

	points_2d = []
	x = type(None)

	for v in points:
		co = view3d_utils.location_3d_to_region_2d(context.region, \
										   context.space_data.region_3d, \
										   v)
		points_2d.append([co.x, co.y] if not isinstance(co, x) else x)

	if x in points_2d: points_2d.clear()

	return points_2d

def scene_ray_hit(context, co, ray_obj=None, scene_ray=False, hit_bounds=False):

	scene = context.scene
	region = context.region
	rv3d = context.region_data

	view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, co)
	ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, co)

	ray_target = ray_origin + view_vector

	def hit_objects_mat(objs):

		for o in objs:
			if o.type == 'MESH':
				yield (o, o.matrix_world.copy() if not o.data.is_editmode \
					else o.matrix_parent_inverse.copy())

	def scene_ray_cast(obj, matrix):

		matrix_inv = matrix.inverted()
		ray_origin_obj = matrix_inv @ ray_origin
		ray_target_obj = matrix_inv @ ray_target
		ray_direction_obj = ray_target_obj - ray_origin_obj

		if obj.data.is_editmode \
			or scene_ray:
			depsgraph = context.evaluated_depsgraph_get()
			hit, pos, normal, face_index, obj, matrix_world = scene.ray_cast(depsgraph, ray_origin_obj, ray_direction_obj)
		else:
			hit, pos, normal, face_index = obj.ray_cast(ray_origin_obj, ray_direction_obj)

		if hit:
			return pos, normal, face_index, obj
		else:
			return None, None, None, None

	best_length_squared = None
	best_hit = None
	best_normal = Vector()
	best_face_index = None
	best_obj = None

	display_types = ['TEXTURED', 'SOLID']
	if hit_bounds: display_types.extend(['WIRE', 'BOUNDS'])

	for obj, matrix in hit_objects_mat([ray_obj] if ray_obj \
		else context.visible_objects):
		if obj.type == 'MESH' \
			and obj.display_type in display_types:
			pos, normal, face_index, object = scene_ray_cast(obj, matrix)
			if pos is not None:
				hit_world = matrix @ pos
				length_squared = (hit_world - ray_origin).length_squared

				if not best_length_squared:
					best_length_squared = length_squared

				if length_squared <= best_length_squared:
					best_length_squared = length_squared
					best_hit = hit_world
					best_normal = normal
					best_face_index = face_index
					best_obj = object

	return best_hit, best_normal, best_face_index, best_obj

def duplicate_obj(name, copy_obj, get_eval=True, link=True):

	new_mesh = bpy.data.meshes.new(name)
	new_obj = bpy.data.objects.new(name, new_mesh)
	new_obj.data = get_evaluated_mesh(context, copy_obj).copy() \
		if (copy_obj.type != 'MESH' or get_eval) else copy_obj.data.copy()
	new_obj.scale = copy_obj.scale
	new_obj.rotation_euler = copy_obj.rotation_euler
	new_obj.location = copy_obj.location

	if link:
		bpy.context.scene.collection.objects.link(new_obj)
		new_obj.select_set(True)

	return new_obj

def draw_helper_text(self, context, text):

	if self.mouse_co:
		mx = self.mouse_co[0]
		my = self.mouse_co[1]

		font_id = 1
		blf.size(font_id, int(round(15 * context.preferences.view.ui_scale, 0)), 60)

		color1 = (1.0,1.0,1.0,0.8)
		color2 = (1.0,1.0,1.0,0.6)
		color3 = (0.262, 0.754, 1.0, 1.0)

		xheight = 2.1
		rect_offset = 15
		row_offset = 0

		if isinstance(text, list):
			height, row1_length, row2_length, comma = get_text_dimensions(context, text, xheight, font_id)

			txt_width = row1_length + row2_length + comma + rect_offset
			txt_height = height

			row_offset = row1_length + rect_offset
		else:
			txt_width = blf.dimensions(font_id, text)[0]
			txt_height = (blf.dimensions(font_id, "gM")[1] * xheight)

		xloc = mx + 30
		yloc = (my - 15) - txt_height
		offx = 15
		offy = 10

		rect_vertices = [(xloc - offx, yloc - offy), (xloc + txt_width + offx, yloc - offy), \
						 (xloc + txt_width + offx, yloc + txt_height + offy), (xloc - offx, yloc + txt_height + offy)]

		draw_shader((0.0,0.0,0.0,0.1), 'TRI_FAN', rect_vertices, 1)
		draw_string(color1, color2, color3, xloc, yloc, text, xheight, row_offset, font_id)

def get_text_dimensions(context, text, xheight, font_id):

	row1_length = max(list(blf.dimensions(font_id, row[0])[0] for row in text))
	row2_length = max(list(blf.dimensions(font_id, row[1])[0] for row in text))
	comma = blf.dimensions(font_id, "_:_")[0]

	line_height = (blf.dimensions(font_id, "gM")[1] * xheight)

	list_height = 0
	for row in text:
		list_height += line_height

	return list_height, row1_length, row2_length, comma

def draw_string(color1, color2, color3, left, bottom, text, xheight, row_offset, font_id):

	blf.enable(font_id,blf.SHADOW)
	blf.shadow(font_id, 0, 0.0, 0.0, 0.0, 1.0)
	blf.shadow_offset(font_id, 1, -1)
	line_height = (blf.dimensions(font_id, "gM")[1] * xheight)
	y_offset = 5

	if isinstance(text, list):
		for string in reversed(text):
			if sum(len(i) for i in string) > 0:
				heading = False
				colrkey = False

				if len(string[1]) == 0: heading = True

				if string[1].find("&") != -1:
					keys_string = string[1].split("&")
					colrkey = True

				blf.position(font_id, (left), (bottom + y_offset), 0)
				blf.color(font_id, *color1)

				if heading:
					blf.draw(font_id, string[0].upper())
				else:
					blf.draw(font_id, string[0].title())

				if not heading:
					colsep = " : "
					blf.position(font_id, (left + row_offset), (bottom + y_offset), 0)
					blf.color(font_id, *color2)
					blf.draw(font_id, colsep)

					coldim = blf.dimensions(font_id, colsep)[0]
					blf.position(font_id, (left + row_offset + coldim), (bottom + y_offset), 0)
					if colrkey:
						blf.color(font_id, *color2)
						blf.draw(font_id, keys_string[0].upper())
						blf.color(font_id, *color3)
						valdim = blf.dimensions(font_id, keys_string[0].title())[0]
						blf.position(font_id, (left + row_offset + valdim + coldim), (bottom + y_offset), 0)
						blf.draw(font_id, keys_string[1].title())
					else:
						blf.draw(font_id, string[1].title())

				y_offset += line_height
	else:
		blf.position(font_id, left, (bottom + y_offset), 0)
		blf.color(font_id, *color1)
		blf.draw(font_id, text)
		y_offset += line_height

	blf.disable(font_id,blf.SHADOW)

def draw_symmetry_helpers(self, context):

	help_txt = [[" ".join("Auto Mirror"), ""]]

	if self.set_axis:

		def enumerate_axis(axes):

			axis_text = ""
			for i in "XYZ":
				if axes[str(i)]: axis_text += " " + i

			return axis_text

		axis_text = enumerate_axis(self.axes)

		help_txt.extend([
				["Axis mirror", (axis_text.lstrip() if axis_text else "None") + "& X/Y/Z"],\
				["Reset", "&R"],\
				["Cancel", "&ESC/RMB"],\
				["Confirm", "&Space/Enter"],\
				])
	else:
		help_txt.extend([
				["Pick Area", "&LMB"],\
				["Cancel", "&ESC/RMB"],\
				])

	draw_helper_text(self, context, help_txt)

def draw_callback_px_quick_symm(self, context):

	r, g, b, a = (1.0,1.0,1.0,1.0)

	draw_symmetry_helpers(self, context)

	def get_axis(obj, src_obj, axis='X'):

		mesh = obj.data
		bm = bmesh.new()

		pivot = src_obj.matrix_world.inverted() @ src_obj.matrix_world.translation
		v0 = bm.verts.new(pivot)

		v1 = []
		v2 = []

		for co in draw_axis(pivot):
			v = bm.verts.new(co)
			bm.edges.new((v0, v))
			v1.append(v)

		if axis == 'X':
			v2 = [v1[2], v1[3], v1[4] ,v1[5]]
		if axis == 'Y':
			v2 = [v1[0], v1[1], v1[4] ,v1[5]]
		if axis == 'Z':
			v2 = [v1[0], v1[1], v1[2] ,v1[3]]

		bmesh.ops.delete(bm, geom=v2, context='VERTS')

		bm.to_mesh(mesh)
		bm.free()

		copy_loc_rot(obj, src_obj)

	if self.set_axis:
		hit = v3d_to_v2d(context, [self.mirror_axis])
		marker = draw_marker((hit[0][0], hit[0][1]))
		draw_shader((0, 0, 0, 0.8), 'LINES', marker, size=1)

		vertices, indices, loop_tris = get_draw_data(self, context, self.center_axis)
		draw_shader((1.0, 1.0, 1.0, 0.5), 'LINES', vertices, size=2, indices=indices)

		if self.axes['X']:
			get_axis(self.color_axis, self.mirror_obj, axis='X')
			vertices, indices, loop_tris = get_draw_data(self, context, self.color_axis)
			draw_shader((1.0, 0.0, 0.0, 1.0), 'LINES', vertices, size=2, indices=indices)

		if self.axes['Y']:
			get_axis(self.color_axis, self.mirror_obj, axis='Y')
			vertices, indices, loop_tris = get_draw_data(self, context, self.color_axis)
			draw_shader((0.0, 1.0, 0.0, 1.0), 'LINES', vertices, size=2, indices=indices)

		if self.axes['Z']:
			get_axis(self.color_axis, self.mirror_obj, axis='Z')
			vertices, indices, loop_tris = get_draw_data(self, context, self.color_axis)
			draw_shader((0.0, 0.0, 1.0, 1.0), 'LINES', vertices, size=2, indices=indices)

		try:
			symm_point = tuple(v3d_to_v2d(context, [self.mirror_axis])[0])
			draw_scale_strips(context, self.mirror_obj.matrix_world.translation, symm_point, alpha=0.8)
		except: pass

def draw_marker(origin):

	x, y = origin
	offset = 5
	points = [(x-offset, y+offset), (x, y), (x+offset, y+offset), \
			(x-offset, y-offset), (x, y), (x+offset, y-offset)]

	return points

def draw_axis(co):

	dim = bpy.context.active_object.dimensions.copy()
	size = min(0.5 * (max(dim) * 0.5), 1.0)

	x, y, z = co
	v0 = (x, y, z)
	x1 = (x + size, y, z)
	x2 = (x - size, y, z)
	y1 = (x, y + size, z)
	y2 = (x, y - size, z)
	z1 = (x, y, z + size)
	z2 = (x, y, z - size)

	return [x1, x2, y1, y2, z1 ,z2]

def draw_extract_helpers(self, context):

	scene = context.scene
	props = scene.rflow_props

	help_txt = [[" ".join("Extract Faces"), ""]]

	face_count = str(len(self.extract_faces))
	inset_val = str('% 0.2f' % self.inset_val).strip()
	influence_lvl = str('% 0.2f' % props.select_influence).strip()
	x_ray = "Off" if self.draw_solid else "On"

	if self.help_index == 1:
		help_txt.extend([
				["Select faces", face_count + "& LMB Click/Drag"],\
				["Inset faces", inset_val + "& Ctrl+Mousedrag /+Shift"],\
				["Select plus", "&Shift+LMB/RMB"],\
				["Select loop", "&Shift+Alt+LMB/RMB"],\
				["Select plus influence", influence_lvl + "& A-D /+Shift"],\
				["X-Ray", x_ray + "& X"],\
				["Toggle Help", "&H"],\
				])

	if self.help_index == 2:
		help_txt.extend([
				["Remove selection", "&RMB"],\
				["Undo selection", "&Z"],\
				["Reset Inset", "&T"],\
				["Reset selection", "&R"],\
				["Cancel", "&ESC"],\
				["Confirm", "&Space/Enter"],\
				["Toggle Help", "&H"],\
				])

	draw_helper_text(self, context, help_txt)

def draw_callback_px_draw_extract(self, context):

	draw_extract_helpers(self, context)

	if self.render_hit and self.view_render_hit:
		draw_shader((1.0,1.0,1.0,0.5), 'LINES', self.hit_verts, size=3, indices=self.hit_indices)

	try:
		if self.draw_strips:
			draw_scale_strips(context, local_center(self.extr_obj), self.mouse_co)
	except: pass

def draw_callback_px_draw_extract_shade(self, context):

	r, g, b = (1.0, 1.0, 1.0)

	vertices, indices, loop_tris = get_draw_data(self, context, self.extr_obj, convert2d=False)
	draw_shader((r, g, b, 0.1), 'TRIS', vertices, size=1, indices=loop_tris, solid=self.draw_solid)
	draw_shader((r, g, b, 0.8), 'LINES', vertices, size=2, indices=indices, solid=self.draw_solid)

def draw_scale_strips(context, coord1, coord2, alpha=0.8):

	def intermediates(p1, p2, nb_points=8):

		x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
		y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

		return [[p1[0] + i * x_spacing, p1[1] +  i * y_spacing]
				for i in range(1, nb_points+1)]

	pivot = tuple(v3d_to_v2d(context, [coord1])[0])
	distance = (Vector(pivot) - Vector(coord2)).length / 5
	line = intermediates(pivot, coord2, nb_points = int(round(distance)))

	draw_shader((0, 0, 0, alpha), 'LINES', line, size=1)

def get_draw_data(self, context, obj, get_verts=True, get_indices=True, get_tris=True, mat_verts=False, mat_obj=None, convert2d=True):

	coord = []; indices = []; loop_tris = []

	mesh = obj.data
	if get_verts:
		vertices = np.empty((len(mesh.vertices), 3), 'f')
		mesh.vertices.foreach_get("co", np.reshape(vertices, len(mesh.vertices) * 3))

		if mat_verts:
			if not mat_obj: mat_obj = obj
			for i, loc in enumerate(vertices):
				vec = mat_obj.matrix_world @ Vector((loc))
				loc = vec
				vertices[i] = loc

		if convert2d:
			coord = v3d_to_v2d(context, vertices)
		else: coord = vertices

	if get_indices:
		indices = mesh.edge_keys[:]

	if get_tris:
		mesh.calc_loop_triangles()
		loop_tris = np.empty((len(mesh.loop_triangles), 3), 'i')
		mesh.loop_triangles.foreach_get("vertices", np.reshape(loop_tris, len(mesh.loop_triangles) * 3))

	return coord, indices, loop_tris

def draw_shader(color, type, coords, size=1, indices=None, solid=False):

	vertex_shader = '''
		uniform mat4 u_ViewProjectionMatrix;

		in vec3 position;
		in float arcLength;

		out float v_ArcLength;

		void main()
		{
			v_ArcLength = arcLength;
			gl_Position = u_ViewProjectionMatrix * vec4(position, 1.0f);
		}
	'''

	fragment_shader = '''
		uniform float u_Scale;

		in float v_ArcLength;

		void main()
		{
			if (step(sin(v_ArcLength * u_Scale), 0.001) == 1) discard;
			gl_FragColor = vec4(1.0);
		}
	'''

	bgl.glEnable(bgl.GL_BLEND)
	bgl.glEnable(bgl.GL_LINE_SMOOTH)
	if solid: bgl.glEnable(bgl.GL_DEPTH_TEST)

	if type =='POINTS':
		bgl.glPointSize(size)
	else:
		bgl.glLineWidth(size)

	if len(coords) > 0:
		if type != "LINE_STRIP":
			if len(coords[0]) > 2 \
				or solid:
				shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')
			else:
				shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')

			batch = batch_for_shader(shader, type, {"pos": coords}, indices=indices)

			shader.bind()
			shader.uniform_float("color", color)
			batch.draw(shader)
		else:
			arc_lengths = [0]
			for a, b in zip(coords[:-1], coords[1:]):
				arc_lengths.append(arc_lengths[-1] + (Vector(a) - Vector(b)).length)

			shader = gpu.types.GPUShader(vertex_shader, fragment_shader)
			batch = batch_for_shader(
				shader, 'LINE_STRIP',
				{"position": coords, "arcLength": arc_lengths},
			)

			rv3d = bpy.context.region_data
			matrix = rv3d.perspective_matrix @ Matrix()

			shader.bind()
			shader.uniform_float("u_ViewProjectionMatrix", matrix)
			shader.uniform_float("u_Scale", 10)
			batch.draw(shader)

	bgl.glLineWidth(1)
	bgl.glPointSize(1)

	bgl.glDisable(bgl.GL_LINE_SMOOTH)
	bgl.glDisable(bgl.GL_BLEND)
	if solid: bgl.glDisable(bgl.GL_DEPTH_TEST)

def undupe(item):
	return list(dict.fromkeys(item))

def init_props(self, event, ops):

	if ops == 'rloop':
		if event.ctrl:
			self.loop_subdv = (0,0,0,0,0,0); self.cuts_base = 0; self.cuts_smooth = 0.0; self.cut_smooth_falloff = 'SMOOTH'

	if ops == "rpanels":
		if event.ctrl:
			self.cuts_base = 0; self.cuts_smooth = 0.0; self.cut_smooth_falloff = 'SMOOTH'

	if ops == "raxis":
		if event.ctrl:
			self.cuts_base = 0; self.inner_cut = (2,2,2)

	if ops == 'rscatter':
		if event.ctrl: self.scatter_points = 10

	if ops == 'rtubes':
		if event.ctrl:
			self.cuts_base = 0; self.cuts_smooth = 0.0; self.cut_smooth_falloff = 'SMOOTH'

	if ops == 'rvcol':
		if event.ctrl:
			self.color = (1.0,1.0,1.0); self.color_min = (0.0,0.0,0.0)

class MESH_OT_r_loop_extrude(Operator):
	'''Randomly extrudes faces to create interesting shapes'''
	bl_idname = 'rand_loop_extr.rflow'
	bl_label = 'Random Loop Extrude'
	bl_options = {'REGISTER', 'UNDO', 'PRESET'}

	loop_objs : EnumProperty(
		name = "Loop Objects",
		description = "Loop objects 1-6",
		items = (
			('1', '1','Add loop object 1'),
			('2', '2','Add loop object 2'),
			('3', '3','Add loop object 3'),
			('4', '4','Add loop object 4'),
			('5', '5','Add loop object 5'),
			('6', '6','Add loop object 6')),
		options = {"ENUM_FLAG"})
	lratio : FloatVectorProperty(
		name        = "Loop Ratio",
		description = "Number of randomized faces per loop",
		default     = (50.0,40.0,30.0,20.0,10.0,10.0),
		size        = 6,
		min         = 0.0,
		max         = 100,
		precision	= 2,
		step        = 1
		)
	lratio_seed : IntVectorProperty(
		name        = "Seed per loop ratio",
		description = "Randomize loop ration seed",
		default     = (1,1,1,1,1,1),
		size        = 6,
		min         = 1,
		soft_max    = 10000,
		step        = 1
		)
	size_mode : EnumProperty(
		name = 'Size Mode',
		description = "Size mode to use for loop ratio",
		items = (
			('PERCENT', 'Percent',''),
			('NUMBER', 'Number','')),
		default = 'PERCENT'
		)
	solver : EnumProperty(
		name = 'Solver',
		description = "Determines the method of generating islands",
		items = (
			('SAMPLING', 'Sampling','Chance to expand island by sampling previous cells'),
			('RECURSIVE', 'Recursive','Chance to expand island by walking previous cells'),
			('RADIAL', 'Radial','Expand island by square area')),
			default = 'SAMPLING'
		)
	path : EnumProperty(
		name = 'Path',
		description = "Determines what edge length to favor when generating new island cells",
		items = (
			('NONE', 'None','Do not favor any edge length when generating islands'),
			('SHORTEST', 'Shortest','Favor shorter edges when generating islands'),
			('LONGEST', 'Longest','Favor longer edges when generating islands')),
		default = 'NONE'
		)
	lsolver_num : IntVectorProperty(
		name        = "Solver",
		description = "Solver per loop",
		default     = (1,1,1,1,1,1),
		size        = 6,
		min         = 1,
		max			= 3,
		step        = 1
		)
	lpath_num : IntVectorProperty(
		name        = "Path",
		description = "Path per loop",
		default     = (1,1,1,1,1,1),
		size        = 6,
		min         = 1,
		max			= 3,
		step        = 1
		)
	lsize01 : FloatVectorProperty(
		name        = "Loop Panel Size %",
		description = "Panel size per loop object",
		default     = (5.0,5.0,5.0,5.0,5.0,5.0),
		size        = 6,
		min         = 0.0,
		max         = 100,
		precision	= 2,
		step        = 1
		)
	lsize02 : IntVectorProperty(
		name        = "Loop Panel Size",
		description = "Panel size per loop object",
		default     = (5,5,5,5,5,5),
		size        = 6,
		min         = 0,
		soft_max    = 1000,
		step        = 1
		)
	loop_subdv : IntVectorProperty(
		name        = "Loop Subdivision",
		description = "Subdivision per loop object",
		default     = (0,0,0,0,0,0),
		size        = 6,
		min         = 0,
		soft_max    = 6,
		step        = 1
		)
	ldepth_max : FloatVectorProperty(
		name        = "Loop Depth",
		description = "Inset depth per loop",
		default     = (0.025,0.025,0.025,0.025,0.025,0.025),
		size        = 6,
		min         = 0.0,
		soft_max    = 10.0,
		step        = 0.1,
		precision   = 3
		)
	ldepth_min : FloatVectorProperty(
		name        = "Loop Depth",
		description = "Inset depth per loop",
		default     = (0.01,0.01,0.01,0.01,0.01,0.01),
		size        = 6,
		min         = 0.0,
		soft_max    = 10.0,
		step        = 0.1,
		precision   = 3
		)
	ldepth_seed : IntVectorProperty(
		name        = "Loop Depth Seed",
		description = "Inset seed per loop",
		default     = (1,1,1,1,1,1),
		size        = 6,
		min         = 1,
		soft_max    = 10000,
		step        = 1
		)
	rand_seed : IntProperty(
		name        = "Randomize",
		description = "Randomize result",
		default     = 1,
		min         = 1,
		soft_max    = 10000,
		step        = 1
		)
	cuts_base : IntProperty(
		name        = "Base Cuts",
		description = "Number of subdivision cuts for base object",
		default     = 0,
		min         = 0,
		soft_max    = 6,
		step        = 1
		)
	cuts_smooth : FloatProperty(
		name        = "Cuts Smooth",
		default     = 0.0,
		min         = 0.0,
		soft_max    = 1.0,
		step        = 0.1,
		precision   = 3
		)
	cut_smooth_falloff : EnumProperty(
		name = "Smooth Falloff",
		description = "Falloff profile of subdivision smoothing",
		items = (
			('SMOOTH', 'Smooth', '', 'SMOOTHCURVE', 0),
			('SPHERE', 'Sphere', '', 'SPHERECURVE', 1),
			('ROOT', 'Root', '', 'ROOTCURVE', 2),
			('SHARP', 'Sharp', '', 'SMOOTHCURVE', 3),
			('LINEAR', 'Linear', '', 'LINCURVE', 4),
			('INVERSE_SQUARE', 'Inverse_square', '', 'INVERSESQUARECURVE', 5)),
		default = 'SMOOTH')
	cut_method : EnumProperty(
		name = "Cut Method",
		description = "Determines how sharp edges will be cut",
		items = (
			('ANGLE', 'Angle',''),
			('WRAP', 'Wrap Around','')),
		default = 'ANGLE'
		)
	cut_threshold : FloatProperty(
		name        = "Cut Angle",
		description = "Maximum angle threshold for edges to be cut",
		default     = radians(30),
		min         = radians(1),
		max         = radians(180),
		step        = 10,
		precision   = 3,
		subtype     = "ANGLE"
		)
	mat_index : IntProperty(
		name        = "Material Index",
		description = "Material assigned to duplicates",
		default     = -1,
		min         = -1,
		max         = 32767,
		step        = 1
		)
	rand_inset : BoolProperty(
		name        = "Randomize Island Height",
		description = "Randomize face islands inset height",
		default     = True
		)
	only_quads : BoolProperty(
		name        = "Quads Only",
		description = "Randomize only quad faces",
		default     = False
		)
	subd_once : BoolProperty(
		name        = "Original Only",
		description = "Use subdivision on original object only",
		default     = False
		)
	skip_border : BoolProperty(
		name        = "Ignore Border Faces",
		description = "Ignore border faces when extruding",
		default     = True
		)
	indiv_sp : BoolProperty(
		name        = "Individual Solver/Path",
		description = "Give each loop its own unique solver and path",
		default     = False
		)
	inset_mats : BoolProperty(
		name        = "Inner/Outer Materials",
		description = "Assign materials to inner and outer inset faces",
		default     = False
		)
	tri_perc : FloatProperty(
		name        = "Triangulate",
		description = "Triangulate faces",
		min         = 0,
		max         = 100,
		precision   = 0,
		default     = 0,
		subtype     = "PERCENTAGE"
		)
	use_clip : BoolProperty(
		name        = "Clip Center",
		description = "Clip center verts when using mirror modifier",
		default     = False
		)
	clip_dist : FloatProperty(
		name        = "Clip Distance",
		description = "Distance within which center vertices are clipped",
		default     = 0.001,
		min         = 0.0,
		soft_max    = 1.0,
		step        = 0.1,
		precision   = 4
		)
	clip_axis : BoolVectorProperty(
		name        = "Clip Axis",
		description = "Clip axis toggles",
		default     = (True, True, True),
		size		= 3,
		subtype		= "XYZ"
		)
	offset_range : IntProperty(
		name        = "Offset Range",
		description = "Number of adjacent faces for verts to be offset",
		default     = 2,
		min         = 0,
		max         = 32767,
		step        = 1
		)
	center_offset : FloatProperty(
		name        = "Center Offset",
		description = "Offset of symmetry verts",
		default     = 0.0,
		soft_min    = -1,
		soft_max    = 1,
		step        = 0.01,
		precision   = 4
	)
	offset_axis : BoolVectorProperty(
		name        = "Smooth",
		description = "Clip smooth axis toggles",
		default     = (True, True, True),
		size		= 3,
		subtype		= "XYZ"
		)
	use_dissolve : BoolProperty(
		name        = "Limited Dissolve",
		description = "Use limited dissolve to remove subdivision from loop object (Slower)",
		default     = True
		)
	angle : FloatProperty(
		name        = "Max Angle",
		description = "Angle limit",
		default     = radians(5),
		min         = 0,
		max         = radians(180),
		step        = 10,
		precision   = 3,
		subtype     = "ANGLE"
		)

	@classmethod
	def poll(cls, context):
		return context.active_object is not None and context.active_object.mode == "OBJECT"

	def execute(self, context):
		obj = context.active_object
		orig_mesh = obj.data
		cont_mesh = obj.data

		loop_objs = set()
		loop_count = [int(i) for i in self.loop_objs] \
			if self.loop_objs else [0]

		ret_idx = set()
		for i in range(0, max(loop_count)):
			bm = bmesh.new()
			temp_mesh = bpy.data.meshes.new(".temp")
			bm.from_mesh(cont_mesh if not self.subd_once else orig_mesh)

			if i == 0 or \
				self.subd_once:
				bmesh.ops.delete(bm, geom=[f for f in bm.faces if not f.select], context='FACES')
				bmesh.ops.subdivide_edges(bm, edges=bm.edges, smooth=self.cuts_smooth, cuts=self.cuts_base, \
					smooth_falloff=self.cut_smooth_falloff, use_grid_fill=True, use_smooth_even=True)

			if i > 0: bpy.data.meshes.remove(cont_mesh)

			if ret_idx \
				and self.skip_border:
				border_f = list(filter(lambda f: f.index in ret_idx, bm.faces))
				bmesh.ops.delete(bm, geom=border_f, context='FACES')

			if self.only_quads:
				non_quads = list(filter(lambda f: len(f.verts) != 4, bm.faces))
				bmesh.ops.delete(bm, geom=non_quads, context='FACES')

			if self.tri_perc:
				tris = get_tris(bm, bm.faces, self.tri_perc, self.lratio_seed[i])
				bmesh.ops.triangulate(bm, faces=tris)

			bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=self.loop_subdv[i], use_grid_fill=True)

			idx = set((f.index for f in bm.faces))
			if self.size_mode == 'PERCENT':
				size = int(len(idx) * self.lsize01[i]/100)
			else:
				size = self.lsize02[i]

			if self.indiv_sp:
				solvers = ["SAMPLING", "RECURSIVE", "RADIAL"]
				loop_solver = solvers[self.lsolver_num[i]-1]
				paths = ["NONE", "SHORTEST", "LONGEST"]
				loop_path = paths[self.lpath_num[i]-1]
			else:
				loop_solver = self.solver
				loop_path = self.path

			split_edg, cells = random_walk(bm, idx, size, self.lratio_seed[i], sampling=loop_solver, \
				path=loop_path, cut_threshold=self.cut_threshold, wrap_angle=self.cut_method == 'WRAP')

			seed(self.lratio_seed[i])
			sample_cells = sample(cells, int(len(cells) * (1 - self.lratio[i]/100)))
			remf = list(chain.from_iterable(sample_cells))

			bmesh.ops.delete(bm, geom=remf, context='FACES')
			if self.cut_method == 'WRAP':
				bmesh.ops.split_edges(bm, edges=list(filter(lambda e: e in split_edg, bm.edges)))
			else:
				bmesh.ops.split_edges(bm, edges=list(filter(lambda e: e.is_boundary or e in split_edg \
					or (e.calc_face_angle(None)	and e.calc_face_angle(None) >= self.cut_threshold), bm.edges)))

			seed(self.ldepth_seed[i])
			ldepth_uniform = uniform(self.ldepth_min[i], self.ldepth_max[i])

			ret_inset = []
			for fc in cells:
				if not fc in sample_cells:
					if self.rand_inset:
						ret = bmesh.ops.inset_region(bm, faces=fc, use_boundary=True, use_even_offset=True, \
							depth=uniform(self.ldepth_min[i], self.ldepth_max[i]))['faces']
					else:
						ret = bmesh.ops.inset_region(bm, faces=fc, use_boundary=True, use_even_offset=True, \
							depth=ldepth_uniform)['faces']
					ret_inset.extend(ret)

			if self.inset_mats:
				assign_temp_mats(temp_mesh, ret_inset)

			if self.tri_perc:
				bmesh.ops.join_triangles(bm, faces=bm.faces, angle_face_threshold=radians(180), \
					angle_shape_threshold=radians(180))

			if self.use_clip:
				if i == 0:
					clip_center(bm, obj, self.clip_dist, self.clip_axis, self.offset_range, \
						self.center_offset, self.offset_axis)
				else: clip_center(bm, obj, self.clip_dist, self.clip_axis)

			remove_axis_faces(bm, obj)

			if ret_inset \
				and self.skip_border \
				and not self.subd_once: ret_idx = { f.index for f in ret_inset if f in bm.faces }

			bm.to_mesh(temp_mesh)
			bm.free()

			cont_mesh = temp_mesh.copy()
			cont_mesh.materials.clear()

			if str(i+1) in self.loop_objs \
				and temp_mesh.polygons:
				new_obj = bpy.data.objects.new(obj.name + "_RLExtr", temp_mesh)
				orig_loc, orig_rot, orig_scale = obj.matrix_world.decompose()
				new_obj.scale = orig_scale
				new_obj.rotation_euler = orig_rot.to_euler()
				new_obj.location = orig_loc
				new_obj.data.use_auto_smooth = obj.data.use_auto_smooth
				new_obj.data.auto_smooth_angle = obj.data.auto_smooth_angle

				if not self.inset_mats:
					assign_mat(self, obj, new_obj, self.mat_index)
				copy_modifiers([obj, new_obj], mod_types=['MIRROR'])

				context.scene.collection.objects.link(new_obj)
				loop_objs.add(new_obj)

		if self.use_dissolve:
			for o in loop_objs:
				mesh = o.data
				bm = bmesh.new()
				bm.from_mesh(mesh)

				if self.use_dissolve:
					bmesh.ops.dissolve_limit(bm, angle_limit=self.angle, \
						use_dissolve_boundaries=False, verts=bm.verts, edges=bm.edges, delimit={'NORMAL'})

				bm.to_mesh(mesh)
				bm.free()

		if not context.scene.rflow_props.select_active:
			if loop_objs:
				objs = [obj] + list(loop_objs)
				select_isolate(context, objs[-1], objs)

		return {'FINISHED'}

	def draw(self, context):
		layout = self.layout
		col = layout.column()
		col.separator(factor=0.1)
		col.label(text="Loops | Ratio | Seed" + (" " * 3) + "(shift+click to add multiple or remove)")
		col.row(align=True).prop(self, "loop_objs", expand=True)
		col.row(align=True).prop(self, "lratio", text="")
		col.row(align=True).prop(self, "lratio_seed", text="")
		col.label(text="Loops Inset Depth Max | Min | Seed")
		col.row(align=True).prop(self, "ldepth_max", text="")
		col.row(align=True).prop(self, "ldepth_min", text="")
		col.row(align=True).prop(self, "ldepth_seed", text="")
		col.label(text="Loops Subdivision")
		col.row(align=True).prop(self, "loop_subdv", text="")
		if self.size_mode == 'PERCENT':
			col.label(text="Loops Panel Size (Percent)")
			col.row(align=True).prop(self, "lsize01", text="")
		else:
			col.label(text="Loops Panel Size (Number)")
			col.row(align=True).prop(self, "lsize02", text="")
		if self.indiv_sp:
			col.label(text="Solver: 1. Sampling 2. Recursive 3. Radial")
			col.row(align=True).prop(self, "lsolver_num", text="")
			col.label(text="Path: 1. None 2. Shortest 3. Longest")
			col.row(align=True).prop(self, "lpath_num", text="")
			col.separator(factor=0.5)
		else:
			col.separator(factor=0.5)
			row = col.row().split(factor=0.27, align=True)
			row.label(text="Solver:")
			row.row(align=True).prop(self, "solver", expand=True)
			row = col.row().split(factor=0.27, align=True)
			row.label(text="Path:")
			row.row(align=True).prop(self, "path", expand=True)
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Size Mode:")
		row.row(align=True).prop(self, "size_mode", expand=True)
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Base Cuts:")
		row.row(align=True).prop(self, "cuts_base", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Cuts Smooth:")
		row.row(align=True).prop(self, "cuts_smooth", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Smooth Falloff:")
		row.row(align=True).prop(self, "cut_smooth_falloff", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Cut Method:")
		row.row(align=True).prop(self, "cut_method", expand=True)
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Cut Angle:")
		row.row(align=True).prop(self, "cut_threshold", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Triangulate:")
		row.row(align=True).prop(self, "tri_perc", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Material Index:")
		row.row(align=True).prop(self, "mat_index", text="")
		col.separator(factor=0.5)
		flow = col.column_flow(columns=2, align=True)
		flow.prop(self, "rand_inset")
		flow.prop(self, "only_quads")
		flow.prop(self, "subd_once")
		flow.prop(self, "skip_border")
		flow.prop(self, "indiv_sp")
		flow.prop(self, "inset_mats")
		col.prop(self, "use_clip")
		if self.use_clip:
			flow = col.column_flow(columns=2, align=True)
			flow.prop(self, "clip_dist")
			flow.row(align=True).prop(self, "clip_axis", text="", expand=True)
			flow = col.column_flow(columns=2, align=True)
			flow.prop(self, "center_offset")
			flow.row(align=True).prop(self, "offset_axis", text="", expand=True)
			col.prop(self, "offset_range")
		col.prop(self, "use_dissolve")
		if self.use_dissolve:
			col.prop(self, "angle")

	def invoke(self, context, event):
		obj = context.active_object
		self.lratio_seed = (1,1,1,1,1,1)
		self.ldepth_seed = (1,1,1,1,1,1)
		if self.mat_index > -1: self.mat_index = -1
		self.loop_objs = set()

		obj.update_from_editmode()
		has_face = [f for f in obj.data.polygons if f.select]

		if has_face:
			init_props(self, event, ops='rloop')
			prefs = context.preferences.addons[__name__].preferences
			if prefs.use_confirm:
				return context.window_manager.invoke_props_dialog(self)
			else:
				return context.window_manager.invoke_props_popup(self, event)
		else:
			self.report({'WARNING'}, "No faces selected.")
			return {"FINISHED"}

class MESH_OT_r_panels(Operator):
	'''Create paneling details'''
	bl_idname = 'rand_panels.rflow'
	bl_label = 'Random Panels'
	bl_options = {'REGISTER', 'UNDO', 'PRESET'}

	solver : EnumProperty(
		name = 'Solver',
		description = "Determines the method of generating islands",
		items = (
			('SAMPLING', 'Sampling','Chance to expand island by sampling previous cells'),
			('RECURSIVE', 'Recursive','Chance to expand island by walking previous cells'),
			('RADIAL', 'Radial','Expand island by square area')),
		default = 'SAMPLING'
		)
	path : EnumProperty(
		name = 'Path',
		description = "Determines what edge length to favor when generating new island cells",
		items = (
			('NONE', 'None','Do not favor any edge length when generating islands'),
			('SHORTEST', 'Shortest','Favor shorter edges when generating islands'),
			('LONGEST', 'Longest','Favor longer edges when generating islands')),
		default = 'NONE'
		)
	size_mode : EnumProperty(
		name = 'Size Mode',
		items = (
			('PERCENT', 'Percent',''),
			('NUMBER', 'Number','')),
		default = 'PERCENT'
		)
	panel_amount : FloatProperty(
		name        = "Panel Amount",
		description = "Total number of panel islands",
		min         = 0,
		max         = 100,
		precision   = 0,
		default     = 100,
		subtype     = "PERCENTAGE"
		)
	panel_size_percent : FloatProperty(
		name        = "Panel Size",
		description = "Randomized panel size",
		min         = 0,
		max         = 100,
		precision   = 0,
		default     = 5,
		subtype     = "PERCENTAGE"
		)
	panel_size_number : IntProperty(
		name        = "Panel Size",
		description = "Randomized panel size",
		default     = 5,
		min         = 0,
		soft_max    = 1000,
		step        = 1
		)
	cut_smooth_falloff : EnumProperty(
		name = "Smooth Falloff",
		description = "Falloff profile of subdivision smoothing",
		items = (
			('SMOOTH', 'Smooth', '', 'SMOOTHCURVE', 0),
			('SPHERE', 'Sphere', '', 'SPHERECURVE', 1),
			('ROOT', 'Root', '', 'ROOTCURVE', 2),
			('SHARP', 'Sharp', '', 'SMOOTHCURVE', 3),
			('LINEAR', 'Linear', '', 'LINCURVE', 4),
			('INVERSE_SQUARE', 'Inverse_square', '', 'INVERSESQUARECURVE', 5)),
		default = 'SMOOTH')
	cuts_smooth : FloatProperty(
		name        = "Smooth",
		default     = 0.0,
		min         = 0.0,
		soft_max    = 1.0,
		step        = 0.1,
		precision   = 3
		)
	cuts_base : IntProperty(
		name        = "Cuts",
		description = "Number of subdivision cuts for panel object",
		default     = 0,
		min         = 0,
		soft_max    = 12,
		step        = 1
		)
	edge_seed : IntProperty(
		name        = "Seed",
		description = "Randomize panel cuts",
		default     = 1,
		min         = 1,
		soft_max    = 10000,
		step        = 1
		)
	bvl_offset_min : FloatProperty(
		name        = "Min",
		description = "Minimum bevel offset/width",
		default     = 0.0,
		min         = 0.0,
		soft_max    = 100.0,
		step        = 0.1,
		precision   = 4
	)
	bvl_offset_max : FloatProperty(
		name        = "Max",
		description = "Minimum bevel offset/width",
		default     = 0.0,
		min         = 0.0,
		soft_max    = 100.0,
		step        = 0.1,
		precision   = 4
	)
	bvl_seg : IntProperty(
		name        = "Segments",
		description = "Bevel segments",
		default     = 1,
		min         = 1,
		soft_max    = 100,
		step        = 1
	)
	bvl_angle : FloatProperty(
		name        = "Angle Limit",
		description = "Maximum angle threshold for vertices to get beveled",
		default     = radians(30),
		min         = radians(1),
		max         = radians(180),
		step        = 10,
		precision   = 3,
		subtype     = "ANGLE"
		)
	clamp_bvl : BoolProperty(
		name        = "Clamp Overlap (Bevel)",
		description = "Clamp the width to avoid overlap",
		default     = True
		)
	bvl_seed : IntProperty(
		name        = "Bevel Seed",
		description = "Randomize bevel width",
		default     = 1,
		min         = 1,
		soft_max    = 10000,
		step        = 1
		)
	margin : FloatProperty(
		name        = "Margin",
		description = "Island margin",
		default     = 0.0,
		min         = 0.0,
		soft_max    = 10.0,
		step        = 0.1,
		precision   = 4
		)
	thickness : FloatProperty(
		name        = "Thick",
		description = "Inset thickness",
		default     = 0.01,
		min         = 0.0,
		soft_max    = 10.0,
		step        = 0.1,
		precision   = 4
		)
	depth : FloatProperty(
		name        = "Depth",
		description = "Inset depth",
		default     = 0.01,
		min         = 0.0,
		soft_max    = 10.0,
		step        = 0.1,
		precision   = 4
		)
	cut_method : EnumProperty(
		name = "Cut Method",
		description = "Determines how sharp edges will be cut",
		items = (
			('ANGLE', 'Angle',''),
			('WRAP', 'Wrap Around','')),
		default = 'ANGLE'
		)
	cut_threshold : FloatProperty(
		name        = "Cut Angle",
		description = "Maximum angle threshold for edges to be cut",
		default     = radians(30),
		min         = radians(1),
		max         = radians(180),
		step        = 10,
		precision   = 3,
		subtype     = "ANGLE"
		)
	clear_faces : EnumProperty(
		name = 'Clear Faces',
		items = (
			('NONE', 'None',''),
			('INNER', 'Inner',''),
			('OUTER', 'Outer','')),
		default = 'NONE'
		)
	cells_height_min : FloatProperty(
		name        = "Min",
		description = "Minimum randomized cell height",
		default     = 0.0,
		min         = 0.0,
		soft_max    = 10.0,
		step        = 0.1,
		precision   = 4
		)
	cells_height_max : FloatProperty(
		name        = "Max",
		description = "Maximum randomized cell height",
		default     = 0.0,
		min         = 0.0,
		soft_max    = 10.0,
		step        = 0.1,
		precision   = 4
		)
	height_seed : IntProperty(
		name        = "Height Seed",
		description = "Height randomize seed",
		default     = 1,
		min         = 1,
		soft_max    = 10000,
		step        = 1
		)
	mat_index : IntProperty(
		name        = "Material Index",
		description = "Material assigned to duplicates",
		default     = -1,
		min         = -1,
		max         = 32767,
		step        = 1
		)
	tri_perc : FloatProperty(
		name        = "Triangulate",
		description = "Triangulate faces",
		min         = 0,
		max         = 100,
		precision   = 0,
		default     = 0,
		subtype     = "PERCENTAGE"
		)
	invert_panel_amount : BoolProperty(
		name        = "Invert Panel Amount",
		description = "Invert panel amount influence",
		default     = False
		)
	inset_mats : BoolProperty(
		name        = "Inner/Outer Materials",
		description = "Assign materials to inner and outer inset faces",
		default     = False
		)
	use_clip : BoolProperty(
		name        = "Clip Center",
		description = "Clip center verts when using mirror modifier",
		default     = False
		)
	clip_dist : FloatProperty(
		name        = "Clip Distance",
		description = "Distance within which center vertices are clipped",
		default     = 0.001,
		min         = 0,
		soft_max    = 1.0,
		step        = 0.1,
		precision   = 4
		)
	clip_axis : BoolVectorProperty(
		name        = "Clip Axis",
		description = "Clip axis toggles",
		default     = (True, True, True),
		size		= 3,
		subtype		= "XYZ"
		)
	offset_range : IntProperty(
		name        = "Offset Range",
		description = "Number of adjacent faces for verts to be offset",
		default     = 2,
		min         = 0,
		max         = 32767,
		step        = 1
		)
	center_offset : FloatProperty(
		name        = "Center Offset",
		description = "Offset of symmetry verts",
		default     = 0.0,
		soft_min    = -1,
		soft_max    = 1,
		step        = 0.01,
		precision   = 4
	)
	offset_axis : BoolVectorProperty(
		name        = "Smooth",
		description = "Clip smooth axis toggles",
		default     = (True, True, True),
		size		= 3,
		subtype		= "XYZ"
		)
	use_dissolve : BoolProperty(
		name        = "Limited Dissolve",
		description = "Use limited dissolve to unify faces",
		default     = False
		)
	angle : FloatProperty(
		name        = "Max Angle",
		description = "Angle limit",
		default     = radians(5),
		min         = 0,
		max         = radians(180),
		step        = 10,
		precision   = 3,
		subtype     = "ANGLE"
		)

	@classmethod
	def poll(cls, context):
		return context.active_object is not None and context.active_object.mode == "OBJECT"

	def execute(self, context):
		obj = bpy.context.active_object
		orig_mesh = obj.data

		bm = bmesh.new()
		temp_mesh = bpy.data.meshes.new(".temp")
		bm.from_mesh(orig_mesh)

		bmesh.ops.delete(bm, geom=[f for f in bm.faces if not f.select], context='FACES')
		bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)

		bmesh.ops.subdivide_edges(bm, edges=bm.edges, smooth=self.cuts_smooth, cuts=self.cuts_base, \
			smooth_falloff=self.cut_smooth_falloff, use_grid_fill=True, use_smooth_even=True)

		if self.tri_perc:
			tris = get_tris(bm, bm.faces, self.tri_perc, self.edge_seed)
			bmesh.ops.triangulate(bm, faces=tris)

		idx = set([f.index for f in bm.faces])
		if self.size_mode == 'PERCENT':
			numf = len(idx)
			size = int(numf * (self.panel_size_percent/100))
		else: size = self.panel_size_number

		split_edg, cells = random_walk(bm, idx, size, self.edge_seed, sampling=self.solver, \
			path=self.path, cut_threshold=self.cut_threshold, wrap_angle=self.cut_method == 'WRAP')

		if self.panel_amount < 100:
			tot = len(cells)
			amt = int(tot * (self.panel_amount/100))
			cells = cells[:amt] if not self.invert_panel_amount else cells[(tot-amt):]
			cells_flat = list(chain.from_iterable(cells))
			split_edg = list(filter(lambda e: any(f in cells_flat for f in e.link_faces), split_edg))

		if self.cut_method == 'WRAP':
			bmesh.ops.split_edges(bm, edges=split_edg)
		else:
			bmesh.ops.split_edges(bm, edges=list(filter(lambda e: e in split_edg or (e.calc_face_angle(None) \
				and e.calc_face_angle(None) >= self.cut_threshold), bm.edges)))

		if self.margin > 0:
			margin = bmesh.ops.inset_region(bm, faces=bm.faces, use_boundary=True, use_even_offset=True, \
				thickness=self.margin)
			bmesh.ops.delete(bm, geom=margin['faces'], context='FACES')

		if sum([self.bvl_offset_min, self.bvl_offset_max]) > 0:
			for i, c in enumerate(cells):
				old_faces = []
				new_faces = []
				for x, f in enumerate(c):
					corner_verts = list(filter(lambda v: v.is_boundary and v.calc_edge_angle(None) \
						and v.calc_edge_angle(None) >= self.bvl_angle, f.verts))
					if corner_verts:
						old_faces.append(f)
						bvl_verts = []
						for y, v in enumerate(corner_verts):
							seed(self.bvl_seed + i + x + y)
							bvl_offset = uniform(self.bvl_offset_min, self.bvl_offset_max)
							bvl = bmesh.ops.bevel(
								bm,
								geom            = [v],
								offset          = bvl_offset,
								offset_type     = 'OFFSET',
								segments        = self.bvl_seg,
								profile         = 0.5,
								affect          = 'VERTICES',
								clamp_overlap	= self.clamp_bvl
								)
							if bvl['verts']:
								new_faces.append(bvl['verts'][0].link_faces[0])
								bvl_verts.extend(bvl['verts'])
						bmesh.ops.remove_doubles(bm, verts=bvl_verts, dist=1e-4)

				for f in old_faces:
					cells[i].remove(f)
				cells[i].extend([f for f in new_faces if f in bm.faces])
				cells[i] = undupe((cells[i]))

		ret = bmesh.ops.inset_region(bm, faces=bm.faces, use_boundary=True, use_even_offset=True, \
			thickness=self.thickness, depth=self.depth)['faces']

		if self.inset_mats:
			assign_temp_mats(temp_mesh, ret)

		if self.clear_faces != 'NONE':
			remf = list(set(bm.faces).difference(set(ret))) \
				if self.clear_faces == 'INNER' else ret
			bmesh.ops.delete(bm, geom=remf, context='FACES')

		if sum([self.cells_height_min, self.cells_height_max]) > 0 \
			and self.clear_faces != 'INNER':
			for i, c in enumerate(cells):
				seed(self.height_seed + i)
				up = uniform(self.cells_height_min, self.cells_height_max)
				fv = undupe(sum((list(f.verts) for f in c), []))
				for v in fv:
					normals = [f.normal for f in v.link_faces if f in c]
					n = sum(normals, Vector()) / len(normals)
					v.co += up * n

		if self.tri_perc:
			bmesh.ops.join_triangles(bm, faces=bm.faces, angle_face_threshold=radians(180), \
				angle_shape_threshold=radians(180))

		if self.use_dissolve:
			bmesh.ops.dissolve_limit(bm, angle_limit=self.angle, \
				use_dissolve_boundaries=False, verts=bm.verts, edges=bm.edges, delimit={'NORMAL'})

		if self.use_clip:
			clip_center(bm, obj, self.clip_dist, self.clip_axis, self.offset_range, \
				self.center_offset, self.offset_axis)
			remove_axis_faces(bm, obj)

		bm.to_mesh(temp_mesh)
		bm.free()

		new_obj = bpy.data.objects.new(obj.name + "_RPanel", temp_mesh)
		orig_loc, orig_rot, orig_scale = obj.matrix_world.decompose()
		new_obj.scale = orig_scale
		new_obj.rotation_euler = orig_rot.to_euler()
		new_obj.location = orig_loc
		new_obj.data.use_auto_smooth = obj.data.use_auto_smooth
		new_obj.data.auto_smooth_angle = obj.data.auto_smooth_angle

		if not self.inset_mats:
			assign_mat(self, obj, new_obj, self.mat_index)
		copy_modifiers([obj, new_obj], mod_types=['MIRROR'])

		context.scene.collection.objects.link(new_obj)

		if not context.scene.rflow_props.select_active:
			objs = [obj, new_obj]
			select_isolate(context, new_obj, objs)

		return {"FINISHED"}

	def draw(self, context):
		layout = self.layout
		col = layout.column()
		col.separator(factor=0.1)
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Solver:")
		row.row(align=True).prop(self, "solver", expand=True)
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Path:")
		row.row(align=True).prop(self, "path", expand=True)
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Size Mode:")
		row.row(align=True).prop(self, "size_mode", expand=True)
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Panel Amount:")
		split = row.split(factor=0.9, align=True)
		split.row(align=True).prop(self, "panel_amount", text="")
		split.prop(self, "invert_panel_amount", text="", icon="ARROW_LEFTRIGHT")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Panel Size:")
		if self.size_mode == 'PERCENT':
			row.row(align=True).prop(self, "panel_size_percent", text="")
		else:
			row.row(align=True).prop(self, "panel_size_number", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Panel Seed:")
		row.row(align=True).prop(self, "edge_seed", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Subdivision:")
		split = row.split(factor=0.5, align=True)
		split.row(align=True).prop(self, "cuts_base")
		split.row(align=True).prop(self, "cuts_smooth")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Smooth Falloff:")
		row.row(align=True).prop(self, "cut_smooth_falloff", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Margin:")
		row.row(align=True).prop(self, "margin", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Inset:")
		split = row.split(factor=0.5, align=True)
		split.row(align=True).prop(self, "thickness")
		split.row(align=True).prop(self, "depth")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Cut Method:")
		row.row(align=True).prop(self, "cut_method", expand=True)
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Cut Angle:")
		row.row(align=True).prop(self, "cut_threshold", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Bevel Offset:")
		split = row.split(factor=0.5, align=True)
		split.row(align=True).prop(self, "bvl_offset_min")
		split.row(align=True).prop(self, "bvl_offset_max")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Bvl Seg/Angle:")
		split = row.split(factor=0.5, align=True)
		split.row(align=True).prop(self, "bvl_seg")
		split.row(align=True).prop(self, "bvl_angle")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Bevel Seed:")
		row.row(align=True).prop(self, "bvl_seed", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Clear Faces:")
		row.row(align=True).prop(self, "clear_faces", expand=True)
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Height:")
		split = row.split(factor=0.5, align=True)
		split.row(align=True).prop(self, "cells_height_min")
		split.row(align=True).prop(self, "cells_height_max")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Height Seed:")
		row.row(align=True).prop(self, "height_seed", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Triangulate:")
		row.row(align=True).prop(self, "tri_perc", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Material Index:")
		row.row(align=True).prop(self, "mat_index", text="")
		col.separator(factor=0.5)
		flow = col.column_flow(columns=2, align=True)
		flow.prop(self, "clamp_bvl")
		flow.prop(self, "inset_mats")
		col.prop(self, "use_clip")
		if self.use_clip:
			flow = col.column_flow(columns=2, align=True)
			flow.prop(self, "clip_dist")
			flow.row(align=True).prop(self, "clip_axis", text="", expand=True)
			flow = col.column_flow(columns=2, align=True)
			flow.prop(self, "center_offset")
			flow.row(align=True).prop(self, "offset_axis", text="", expand=True)
			col.prop(self, "offset_range")
		col.prop(self, "use_dissolve")
		if self.use_dissolve:
			col.prop(self, "angle")

	def invoke(self, context, event):
		obj = context.active_object
		self.panel_amount = 100
		self.edge_seed = 1
		self.bvl_seed = 1
		self.height_seed = 1
		if self.mat_index > -1: self.mat_index = -1

		obj.update_from_editmode()
		has_face = [f for f in obj.data.polygons if f.select]

		if has_face:
			init_props(self, event, ops='rpanels')
			prefs = context.preferences.addons[__name__].preferences
			if prefs.use_confirm:
				return context.window_manager.invoke_props_dialog(self)
			else:
				return context.window_manager.invoke_props_popup(self, event)
		else:
			self.report({'WARNING'}, "No faces selected.")
			return {"FINISHED"}

class MESH_OT_r_axis_extrude(Operator):
	'''Extrude randomly in the xyz axis'''
	bl_idname = 'rand_axis_extr.rflow'
	bl_label = 'Random Axis Extrude'
	bl_options = {'REGISTER', 'UNDO', 'PRESET'}

	cuts_base : IntProperty(
		name        = "Cuts",
		description = "Number of subdivision cuts for selected faces",
		default     = 0,
		min         = 0,
		soft_max    = 12,
		step        = 1
		)
	extr_offset : IntVectorProperty(
		name        = "Source Extrude",
		description = "Number of loops to extrude from original selection",
		default     = (1,1,1),
		size        = 3,
		min         = 0,
		soft_max    = 100,
		step        = 1,
		subtype		= "XYZ"
		)
	loop_offset : FloatVectorProperty(
		name        = "Loop Offset",
		description = "Minimum threshold to push extrusion for the next loop",
		default     = (0.5,0.5,0.5),
		size        = 3,
		min         = 0.0,
		max    		= 1.0,
		step        = 0.1,
		precision   = 3,
		subtype		= "XYZ"
		)
	axis_order : EnumProperty(
		name = 'Axis Order',
		description = "Order on which axis to loop first",
		items = (
			('HIGH', 'Highest','Highest axis loop first'),
			('LOW', 'Lowest','Lowest axis loop first')),
		default = 'HIGH'
		)
	axis_loop : IntVectorProperty(
		name        = "Axis Loop",
		description = "Axis loop extrusion count",
		default     = (0,0,0),
		size        = 3,
		min         = 0,
		soft_max    = 100,
		step        = 1,
		subtype		= "XYZ"
		)
	loop_seed : IntVectorProperty(
		name        = "Loop Seed",
		description = "Axis loop seed",
		default     = (1,1,1),
		size        = 3,
		min         = 1,
		soft_max    = 10000,
		step        = 1
		)
	depth_max : FloatVectorProperty(
		name        = "Depth Max",
		description = "Maximum extrusion depth",
		default     = (0.5,0.5,0.5),
		size        = 3,
		min         = 0.0,
		soft_max    = 10.0,
		step        = 0.5,
		precision   = 3,
		subtype		= "XYZ"
		)
	depth_min : FloatVectorProperty(
		name        = "Depth Min",
		description = "Minimum extrusion depth",
		default     = (0.1,0.1,0.1),
		size        = 3,
		min         = 0.0,
		soft_max    = 10.0,
		step        = 0.5,
		precision   = 3,
		subtype		= "XYZ"
		)
	depth_seed : IntVectorProperty(
		name        = "Depth Seed",
		description = "Extrusion depth seed",
		default     = (1,1,1),
		size        = 3,
		min         = 1,
		soft_max    = 10000,
		step        = 1
		)
	inner_cut : IntVectorProperty(
		name        = "Inner Cut",
		description = "Number subdivisions on extrusion edges",
		default     = (2,2,2),
		size        = 3,
		min         = 0,
		soft_max    = 12,
		step        = 1,
		subtype		= "XYZ"
		)
	influence : IntVectorProperty(
		name        = "Influence",
		description = "Number of linked faces for face verts to have in order to extrude",
		default     = (4,4,4),
		size        = 3,
		min         = 1,
		soft_max    = 10,
		step        = 1,
		subtype		= "XYZ"
		)
	scale_max : FloatVectorProperty(
		name        = "Scale Max",
		description = "Minimum extrusion scale",
		default     = (1.0,1.0,1.0),
		size        = 3,
		min			= 0.0,
		soft_min    = 0.01,
		soft_max    = 10.0,
		step        = 0.1,
		precision   = 3,
		subtype		= "XYZ"
		)
	scale_min : FloatVectorProperty(
		name        = "Scale Min",
		description = "Minimum extrusion scale",
		default     = (1.0,1.0,1.0),
		size        = 3,
		min			= 0.0,
		soft_min    = 0.01,
		soft_max    = 10.0,
		step        = 0.1,
		precision   = 3,
		subtype		= "XYZ"
		)
	scale_seed : IntVectorProperty(
		name        = "Scale Seed",
		description = "Extrusion scale seed",
		default     = (1,1,1),
		size        = 3,
		min         = 1,
		soft_max    = 10000,
		step        = 1
		)
	threshold : FloatVectorProperty(
		name        = "Min",
		description = "Normal limit on faces in order to be extruded",
		default     = (0.5,0.5,0.5),
		size        = 3,
		min         = 0.0,
		soft_min    = 0.0001,
		soft_max    = 1.0,
		step        = 0.5,
		precision   = 4,
		subtype		= "XYZ"
		)
	scale_clipping : FloatProperty(
		name        = "Scale Clipping",
		description = "Clipping distance when scaling faces in the symmetry lines",
		default     = 0.0001,
		min			= 0.0,
		soft_min    = 0.0001,
		soft_max    = 1.0,
		step        = 0.01,
		precision   = 4
		)
	mat_index : IntProperty(
		name        = "Material Index",
		description = "Material assigned to duplicates",
		default     = -1,
		min         = -1,
		max         = 32767,
		step        = 1
		)
	single_obj : BoolProperty(
		name        = "Single Object",
		description = "Make the extrusions part of the source mesh",
		default     = False
		)
	hit_self : BoolProperty(
		name        = "Hit Self",
		description = "Allows extrusion in faces that points to source mesh",
		default     = False
		)
	cut_symm : BoolProperty(
		name        = "Bisect Symmetry",
		description = "Bisect symmetry line if has mirror modifier",
		default     = True
		)

	@classmethod
	def poll(cls, context):
		return context.active_object is not None and context.active_object.mode == "OBJECT"

	def hit_source(self, context, obj, loc, norm):

		hit = False
		if not self.hit_self:
			depsgraph = context.evaluated_depsgraph_get()
			_, _, _, _, hit_obj, _ = context.scene.ray_cast(depsgraph, loc + norm, norm)
			if hit_obj == obj: hit = True

		return hit

	def execute(self, context):
		obj = context.active_object
		mesh = obj.data

		split = not self.single_obj
		mirror = self.mirror

		bm = bmesh.new()
		if split:
			temp_mesh = bpy.data.meshes.new(".temp")
		bm.from_mesh(mesh)

		axis_vector = [Vector((1,0,0)),Vector((0,1,0)),Vector((0,0,1))]
		axis_invert = True if self.axis_order == 'HIGH' else False

		loops = self.axis_loop
		loop_order = {(self.extr_offset[i], self.loop_offset[i], tuple(axis_vector[i]), self.threshold[i], \
			self.loop_seed[i], self.depth_seed[i], self.depth_min[i], self.depth_max[i], self.inner_cut[i], \
			self.influence[i], self.scale_max[i], self.scale_min[i], self.scale_seed[i]):x for i, x in enumerate(loops)}
		loop_order = sorted(loop_order, key=loop_order.get, reverse=axis_invert)

		loops = sorted(loops, reverse=axis_invert)
		offset1 = [x[0] for x in loop_order]
		offset2 = [x[1] for x in loop_order]
		axis = [x[2] for x in loop_order]
		thresh = [x[3] for x in loop_order]
		loop_seed = [x[4] for x in loop_order]
		depth_seed = [x[5] for x in loop_order]
		depth_min = [x[6] for x in loop_order]
		depth_max = [x[7] for x in loop_order]
		inner_cut = [x[8] for x in loop_order]
		influence = [x[9] for x in loop_order]
		scale_max = [x[10] for x in loop_order]
		scale_min = [x[11] for x in loop_order]
		scale_seed = [x[12] for x in loop_order]

		orig_faces = []
		init_faces = [f for f in bm.faces if f.select]

		if self.cuts_base > 0:
			f_edges = sum([list(f.edges) for f in init_faces], [])
			ret_subd1 = bmesh.ops.subdivide_edges(bm, edges=undupe(f_edges), cuts=self.cuts_base, use_grid_fill=True)
			init_faces = [f for f in ret_subd1['geom_inner'] if isinstance(f, bmesh.types.BMFace)]

		if split: orig_faces = bm.faces[:]

		loop_hi = max(loops)
		loop_lo = min(loops) or 1

		newf = []
		for i in range(loop_hi):
			for n, vec in enumerate(axis):
				vec = Vector(vec)
				if i < loops[n]:
					if i >= offset1[n]:
						extf = undupe(newf)
					else: extf = undupe(init_faces + newf)

					faces = [f for f in extf if \
						(vec - f.normal).length < thresh[n] \
						or (vec + f.normal).length < thresh[n]]

					if faces:
						seed(loop_seed[n] + i)
						face = choice(faces)

						loc = face.calc_center_median()
						norm = face.normal
						hit = self.hit_source(context, obj, loc, norm)

						push = False
						loop_range = loop_hi if axis_invert else loop_lo
						if loops[n] < loop_range:
							push = max((offset2[n] - ((1 / loop_range) * (i + 1))), 0) >= random()

						if not hit \
							and not push \
							and not any(v for v in face.verts if len(v.link_faces) > influence[n]):
							if i < offset1[n]:
								if split and face in orig_faces: orig_faces.remove(face)

							seed(depth_seed[n] + i)
							ret_inset = bmesh.ops.inset_region(bm, faces=[face], use_boundary=True, use_even_offset=True, \
							depth=uniform(depth_min[n], depth_max[n]))['faces']

							center = face.calc_center_median()
							if mirror:
								dist = self.scale_clipping
								midp = set()
								for v in face.verts:
									near_zeros = 0
									for x in range(3):
										if mirror.use_axis[x]:
											if -dist <= v.co[x] <= dist:
												midp.add(tuple(v.co))
												near_zeros += 1
									if near_zeros > 1:
										center = v.co
										midp.clear()
										break

								if midp: center = sum((Vector(co) for co in midp), Vector()) / len(midp)

							seed(scale_seed[n] + i)
							sca = uniform(scale_min[n], scale_max[n])
							bmesh.ops.scale(bm, vec=Vector((sca,sca,sca)), \
								space=Matrix.Translation(center).inverted(), verts=face.verts)

							if inner_cut[n] > 0:
								inner_edges = set()
								for f in ret_inset:
									for e in f.edges:
										if all(f in ret_inset for f in e.link_faces): inner_edges.add(e)

								inner_faces = []
								ret_subd2 = bmesh.ops.subdivide_edges(bm, edges=list(inner_edges), cuts=inner_cut[n])
								for e in ret_subd2['geom_inner']:
									for f in e.link_faces: inner_faces.append(f)

								newf.extend(inner_faces + [face])
							else: newf.extend(ret_inset + [face])
						else: loops[n] += 1

		if split:
			bmesh.ops.delete(bm, geom=orig_faces, context='FACES')
			mesh = temp_mesh

		if self.cut_symm:
			bisect_symmetry(bm, obj)

		remove_axis_faces(bm, obj)

		bm.to_mesh(mesh)
		bm.free()

		if split and \
			temp_mesh.polygons:
			new_obj = bpy.data.objects.new(obj.name + "_RAExtr", temp_mesh)
			orig_loc, orig_rot, orig_scale = obj.matrix_world.decompose()
			new_obj.scale = orig_scale
			new_obj.rotation_euler = orig_rot.to_euler()
			new_obj.location = orig_loc
			new_obj.data.use_auto_smooth = obj.data.use_auto_smooth
			new_obj.data.auto_smooth_angle = obj.data.auto_smooth_angle

			assign_mat(self, obj, new_obj, self.mat_index)
			copy_modifiers([obj, new_obj], mod_types=['MIRROR'])

			context.scene.collection.objects.link(new_obj)

			if not context.scene.rflow_props.select_active:
				objs = [obj, new_obj]
				select_isolate(context, new_obj, objs)

		return {"FINISHED"}

	def draw(self, context):
		layout = self.layout
		col = layout.column()
		col.separator(factor=0.1)
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Base Cut:")
		row.row(align=True).prop(self, "cuts_base", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Source Extrude:")
		row.row(align=True).prop(self, "extr_offset", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Loop Offset:")
		row.row(align=True).prop(self, "loop_offset", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Axis Order:")
		row.row(align=True).prop(self, "axis_order", expand=True)
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Axis Loop:")
		row.row(align=True).prop(self, "axis_loop", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Loop Seed:")
		row.row(align=True).prop(self, "loop_seed", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Depth Max:")
		row.row(align=True).prop(self, "depth_max", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Depth Min:")
		row.row(align=True).prop(self, "depth_min", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Depth Seed:")
		row.row(align=True).prop(self, "depth_seed", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Threshold:")
		row.row(align=True).prop(self, "threshold", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Inner Cut:")
		row.row(align=True).prop(self, "inner_cut", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Influence:")
		row.row(align=True).prop(self, "influence", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Scale Max:")
		row.row(align=True).prop(self, "scale_max", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Scale Min:")
		row.row(align=True).prop(self, "scale_min", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Scale Seed:")
		row.row(align=True).prop(self, "scale_seed", text="")
		if self.mirror:
			row = col.row().split(factor=0.27, align=True)
			row.label(text="Scale Clipping:")
			row.row(align=True).prop(self, "scale_clipping", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Material Index:")
		row.row(align=True).prop(self, "mat_index", text="")
		col.separator(factor=0.5)
		col.prop(self, "hit_self")
		col.prop(self, "cut_symm")
		col.prop(self, "single_obj")

	def invoke(self, context, event):
		obj = context.active_object
		self.loop_seed = (1,1,1)
		self.depth_seed = (1,1,1)
		self.mirror = next((m for m in obj.modifiers if m.type == 'MIRROR'), None)

		obj.update_from_editmode()
		has_face = [f for f in obj.data.polygons if f.select]

		if has_face:
			init_props(self, event, ops='raxis')
			prefs = context.preferences.addons[__name__].preferences
			if prefs.use_confirm:
				return context.window_manager.invoke_props_dialog(self)
			else:
				return context.window_manager.invoke_props_popup(self, event)
		else:
			self.report({'WARNING'}, "No faces selected.")
			return {"FINISHED"}

class MESH_OT_r_scatter(Operator):
	'''Create scatter details'''
	bl_idname = 'rand_scatter.rflow'
	bl_label = 'Random Scatter'
	bl_options = {'REGISTER', 'UNDO', 'PRESET'}

	scatter_type : EnumProperty(
		name = 'Type',
		description = "Scatter type",
		items = (
			('CUBE', 'Cube',''),
			('MESH', 'Mesh',''),
			('COLLECTION', 'Collection','')),
		default = 'CUBE'
		)
	list : StringProperty(
		name        = "Mesh",
		description = "Mesh object for scatter"
		)
	list_col : StringProperty(
		name        = "Collections",
		description = "Collection objects for scatter"
		)
	meshes : CollectionProperty(type=PropertyGroup)
	collections : CollectionProperty(type=PropertyGroup)
	coll_seed : IntProperty(
		name        = "Object Seed",
		description = "Randomize seed for collection objects",
		default     = 1,
		min         = 1,
		soft_max    = 10000,
		step        = 1
		)
	scatter_seed : IntProperty(
		name        = "Scatter Seed",
		description = "Randomize scatter points",
		default     = 1,
		min         = 1,
		soft_max    = 10000,
		step        = 1
		)
	scatter_points : IntProperty(
		name        = "Scatter Points",
		description = "Number of scatter points",
		default     = 10,
		min         = 1,
		soft_max    = 1000,
		step        = 1
		)
	size_seed : IntProperty(
		name        = "Size Seed",
		description = "Randomize size of scatter object",
		default     = 1,
		min         = 1,
		soft_max    = 10000,
		step        = 1
		)
	size_min : FloatProperty(
		name        = "Min",
		description = "Minimum scatter size",
		default     = 0.1,
		min         = 0.0,
		soft_max    = 10.0,
		step        = 0.1,
		precision   = 2
		)
	size_max : FloatProperty(
		name        = "Max",
		description = "Maximum scatter size",
		default     = 1.0,
		min         = 0.0,
		soft_max    = 10.0,
		step        = 0.1,
		precision   = 2
		)
	scale_seed : IntVectorProperty(
		name        = "Scatter Seed",
		description = "Scatter object scaling seed",
		default     = (1,1,1),
		size        = 3,
		min         = 1,
		soft_max	= 10000,
		step        = 1
		)
	scale : FloatVectorProperty(
		name        = "Scale",
		default     = (1.0,1.0,1.0),
		size        = 3,
		soft_min    = 0.01,
		soft_max    = 10.0,
		step        = 1.0,
		description = "Randomized faces scale"
		)
	rot_axis : FloatVectorProperty(
		name        = "Rotation",
		description = "Rotate axis",
		default     = (0,0,0),
		size        = 3,
		min         = radians(-360),
		max         = radians(360),
		step        = 10,
		precision   = 3,
		subtype     = "EULER"
		)
	rot_seed : IntVectorProperty(
		name        = "Rotation Seed",
		description = "Scatter object rotation seed",
		default     = (1,1,1),
		size        = 3,
		min         = 1,
		soft_max	= 10000,
		step        = 1
		)
	explode_min : FloatProperty(
		name        = "Min",
		description = "Minimum explode offset",
		default     = 0.0,
		min         = 0.0,
		soft_max    = 10.0,
		step        = 0.1,
		precision   = 3
		)
	explode_max : FloatProperty(
		name        = "Max",
		description = "Maximum explode offset",
		default     = 0.0,
		min         = 0.0,
		soft_max    = 10.0,
		step        = 0.1,
		precision   = 3
		)
	explode_seed : IntProperty(
		name        = "Explode Seed",
		description = "Randomize explode offset of scatter object",
		default     = 1,
		min         = 1,
		soft_max    = 10000,
		step        = 1
		)
	cluster : FloatProperty(
		name        = "Cluster",
		description = "Cluster offset relative to tri center",
		default     = 0.0,
		min         = 0.0,
		soft_max    = 1.0,
		step        = 0.1,
		precision   = 4
		)
	margin : FloatProperty(
		name        = "Margin",
		description = "Margin from boundary edges",
		default     = 0.0,
		soft_min    = -1.0,
		soft_max    = 1.0,
		step        = 0.1,
		precision   = 4
		)
	mat_index : IntProperty(
		name        = "Material Index",
		description = "Material assigned to duplicates",
		default     = -1,
		min         = -1,
		max         = 32767,
		step        = 1
		)
	get_material : BoolProperty(
		name        = "Inherit Material",
		description = "Get materials from source mesh",
		default     = True
		)
	single_scatter : BoolProperty(
		name        = "Single Object",
		description = "Generate scatter objects as single mesh",
		default     = True
		)

	@classmethod
	def poll(cls, context):
		return context.active_object is not None and context.active_object.mode == "OBJECT"

	def point_on_triangle(self, face):
		'''https://blender.stackexchange.com/a/221597'''

		a, b, c = map(lambda v: v.co, face.verts)
		a2b = b - a
		a2c = c - a
		height = triangular(low=0.0, high=1.0, mode=0.0)

		return a + a2c*height + a2b*(1-height) * random(), face.normal, face.calc_center_median()

	def add_scatter(self, obj, mesh_data, matlist):

		context = bpy.context

		new_obj = bpy.data.objects.new(obj.name + "_RScatter", mesh_data)
		orig_loc, orig_rot, orig_scale = obj.matrix_world.decompose()
		new_obj.scale = orig_scale
		new_obj.rotation_euler = orig_rot.to_euler()
		new_obj.location = orig_loc
		new_obj.data.use_auto_smooth = obj.data.use_auto_smooth
		new_obj.data.auto_smooth_angle = obj.data.auto_smooth_angle

		if self.get_material:
			assign_mat(self, obj, new_obj, self.mat_index)
		else:
			for m in matlist:
				new_obj.data.materials.append(m)

		copy_modifiers([obj, new_obj], mod_types=['MIRROR'])
		context.scene.collection.objects.link(new_obj)

		if not context.scene.rflow_props.select_active \
			and self.single_scatter:
			objs = [obj, new_obj]
			select_isolate(context, new_obj, objs)

	def execute(self, context):
		obj = bpy.context.active_object
		orig_mesh = obj.data

		bm = bmesh.new()
		temp_mesh = bpy.data.meshes.new(".temp")
		bm.from_mesh(orig_mesh)
		bm.to_mesh(temp_mesh)

		bmesh.ops.delete(bm, geom=[f for f in bm.faces if not f.select], context='FACES')

		if bool(self.margin):
			bounds_margin = bmesh.ops.inset_region(bm, faces=bm.faces, use_boundary=True, use_even_offset=True, \
				thickness=self.margin, depth=0.0)['faces']
			bmesh.ops.delete(bm, geom=bounds_margin, context='FACES')

		triangles = bmesh.ops.triangulate(bm, faces=bm.faces)['faces']
		surfaces = map(lambda t: t.calc_area(), triangles)
		seed(self.scatter_seed)
		listp = choices(population=triangles, weights=surfaces, k=self.scatter_points)
		points = map(self.point_on_triangle, listp)

		def get_rot(track, obj):

			quat = normal.to_track_quat(track, 'Y')
			mat = obj.matrix_world @ quat.to_matrix().to_4x4()
			rot = mat.to_3x3().normalized()

			return rot

		def list_mat(obj, matlist):

			for m in obj.data.materials:
				if m not in matlist: matlist.append(m)

		cont = True
		scatter_obj = None
		scatter_type = self.scatter_type
		matlist = []
		single_obj = self.single_scatter

		for i, p in enumerate(list(points)):
			bm_scatter = bmesh.new()
			scatter_data = bpy.data.meshes.new(".temp_scatter")

			loc = p[0]
			normal = p[1]
			center = p[2]

			if scatter_type == 'CUBE':
				seed(self.size_seed + i)
				scatter_verts = bmesh.ops.create_cube(bm_scatter, size=uniform(self.size_min, self.size_max))['verts']
				rot = get_rot('-Z', obj)
			else:
				if scatter_type == 'MESH':
					scatter_obj = bpy.data.objects.get(self.list)
				elif scatter_type == 'COLLECTION':
					collection = bpy.data.collections.get(self.list_col)
					if collection:
						mesh_objs = [o for o in bpy.data.collections.get(self.list_col).all_objects \
							if o.type == 'MESH' and o != obj]
						if mesh_objs:
							seed(self.coll_seed + i)
							coll_obj = choice(mesh_objs)
							scatter_obj = bpy.data.objects.get(coll_obj.name)

				if scatter_obj:
					bm_scatter.from_mesh(scatter_obj.data)
					scatter_verts = bm_scatter.verts
					rot = get_rot('Z', scatter_obj)
					if not self.get_material: list_mat(scatter_obj, matlist)
				else: cont = False

			if cont:
				loc += ((center-loc) * self.cluster)
				if sum([self.explode_min, self.explode_max]) > 0:
					seed(self.explode_seed + i)
					loc += normal * uniform(self.explode_min, self.explode_max)

				bmesh.ops.translate(
					bm_scatter,
					verts   = scatter_verts,
					vec     = loc
					)

				if scatter_type != 'CUBE':
					seed(self.size_seed + i)
					sz = uniform(self.size_min, self.size_max)
					bmesh.ops.scale(
						bm_scatter,
						vec     = Vector((sz, sz, sz)),
						space   = Matrix.Translation(loc).inverted(),
						verts   = scatter_verts
						)

				def sca_seed(x, y, z):

					scale = [x, y, z]
					for n, v in enumerate(scale):
						seed(self.scale_seed[n] + i)
						scale[n] = uniform(self.size_min, v)
						seed(0)

					return scale

				x, y, z = self.scale
				scale = sca_seed(x, y, z)
				bmesh.ops.scale(
					bm_scatter,
					vec     = Vector(scale),
					space   = Matrix.Translation(loc).inverted(),
					verts   = scatter_verts
					)

				def rot_seed(x, y, z):

					axis = [x, y, z]
					for n, v in enumerate(axis):
						if self.rot_seed[n] > 1:
							seed(self.rot_seed[n] + i)
							axis[n] = uniform(-v, v)
						else: axis[n] = v
						seed(0)

					return Euler(Vector(axis))

				x, y, z = self.rot_axis
				rot_axis = rot_seed(x, y, z)
				_, orig_rot, _ = obj.matrix_world.decompose()
				bmesh.ops.rotate(
					bm_scatter,
					verts   = scatter_verts,
					cent    = loc,
					matrix  = orig_rot.to_matrix().inverted() @ rot @ rot_axis.to_matrix()
					)

			bisect_symmetry(bm_scatter, obj)

			bm_scatter.to_mesh(scatter_data)
			bm_scatter.free()

			if not single_obj:
				if scatter_data.polygons:
					self.add_scatter(obj, scatter_data, matlist)
			else:
				bm.from_mesh(scatter_data)
				bpy.data.meshes.remove(scatter_data)

		if single_obj: bmesh.ops.delete(bm, geom=triangles, context='FACES')

		bm.to_mesh(temp_mesh)
		bm.free()

		if not single_obj:
			bpy.data.meshes.remove(temp_mesh)
		else: self.add_scatter(obj, temp_mesh, matlist)

		return {"FINISHED"}

	def draw(self, context):
		layout = self.layout
		col = layout.column()
		col.separator(factor=0.1)
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Type:")
		row.row(align=True).prop(self, "scatter_type", expand=True)
		if self.scatter_type == 'MESH':
			row = col.row().split(factor=0.27, align=True)
			row.label(text="Mesh:")
			row.prop_search(
				self,
				"list",
				self,
				"meshes",
				text="",
				icon = "MESH_DATA"
				)
		if self.scatter_type == 'COLLECTION':
			row = col.row().split(factor=0.27, align=True)
			row.label(text="Collection:")
			row.prop_search(
				self,
				"list_col",
				self,
				"collections",
				text="",
				icon = "OUTLINER_COLLECTION"
				)
			row = col.row().split(factor=0.27, align=True)
			row.label(text="Object Seed:")
			row.row(align=True).prop(self, "coll_seed", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Points:")
		row.row(align=True).prop(self, "scatter_points", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Point Seed:")
		row.row(align=True).prop(self, "scatter_seed", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Scatter Size:")
		split = row.split(factor=0.5, align=True)
		split.row(align=True).prop(self, "size_min")
		split.row(align=True).prop(self, "size_max")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Size Seed:")
		row.row(align=True).prop(self, "size_seed", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Scatter Scale:")
		row.row(align=True).prop(self, "scale", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Scale Seed:")
		row.row(align=True).prop(self, "scale_seed", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Rotation:")
		row.row(align=True).prop(self, "rot_axis", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Rotation Seed:")
		row.row(align=True).prop(self, "rot_seed", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Explode:")
		split = row.split(factor=0.5, align=True)
		split.row(align=True).prop(self, "explode_min")
		split.row(align=True).prop(self, "explode_max")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Explode Seed:")
		row.row(align=True).prop(self, "explode_seed", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Cluster:")
		row.row(align=True).prop(self, "cluster", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Margin:")
		row.row(align=True).prop(self, "margin", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Material Index:")
		row.row(align=True).prop(self, "mat_index", text="")
		col.separator(factor=0.5)
		col.prop(self, "get_material")
		col.prop(self, "single_scatter")

	def invoke(self, context, event):
		sce = context.scene
		obj = context.active_object
		self.coll_seed = 1
		self.scatter_seed = 1
		self.size_seed = 1
		self.scale_seed = (1,1,1)
		self.rot_seed = (1,1,1)
		self.mat_index = -1

		self.list = ""
		self.meshes.clear()
		self.list_col = ""
		self.collections.clear()

		obj.update_from_editmode()
		has_face = [f for f in obj.data.polygons if f.select]

		if has_face:
			init_props(self, event, ops='rscatter')
			for o in sce.objects:
				if o.type == 'MESH' and \
					o != obj:
					newListItem = self.meshes.add()
					newListItem.name = o.name

			for c in bpy.data.collections:
				newListItem = self.collections.add()
				newListItem.name = c.name

			prefs = context.preferences.addons[__name__].preferences
			if prefs.use_confirm:
				return context.window_manager.invoke_props_dialog(self)
			else:
				return context.window_manager.invoke_props_popup(self, event)
		else:
			self.report({'WARNING'}, "No faces selected.")
			return {"FINISHED"}

class MESH_OT_r_tubes(Operator):
	'''Create random tubes'''
	bl_idname = 'rand_tubes.rflow'
	bl_label = 'Random Tubes'
	bl_options = {'REGISTER', 'UNDO', 'PRESET'}

	path : EnumProperty(
		name = 'Path',
		items = (
			('NONE', 'None',''),
			('SHORTEST', 'Shortest',''),
			('LONGEST', 'Longest','')),
		default = 'NONE'
		)
	panel_num : IntProperty(
		name        = "Number",
		description = "Panel amount",
		default     = 5,
		min         = 1,
		soft_max    = 100,
		step        = 1
		)
	edg_length : IntProperty(
		name        = "Length",
		description = "Randomized Edge Length",
		default     = 5,
		min         = 1,
		soft_max    = 100,
		step        = 1
		)
	edg_seed : IntProperty(
		name        = "Seed",
		description = "Random edge walk seed",
		default     = 1,
		min         = 1,
		soft_max    = 10000,
		step        = 1
		)
	edg_offset_min : FloatProperty(
		name        = "Min",
		description = "Minimum edge offset",
		default     = 0.0,
		min         = 0.0,
		soft_max    = 10.0,
		step        = 0.1,
		precision   = 3
		)
	edg_offset_max : FloatProperty(
		name        = "Max",
		description = "Maximum edge offset",
		default     = 0.1,
		min         = 0.0,
		soft_max    = 10.0,
		step        = 0.1,
		precision   = 3
		)
	offset_seed : IntProperty(
		name        = "Offset Seed",
		description = "Random offset seed",
		default     = 1,
		min         = 1,
		soft_max    = 10000,
		step        = 1
		)
	margin : FloatProperty(
		name        = "Margin",
		description = "Margin from boundary edges",
		default     = 0.0,
		min         = 0.0,
		soft_max    = 1.0,
		step        = 0.1,
		precision   = 4
		)
	width : FloatProperty(
		name        = "Depth",
		description = "Depth of curve object",
		default     = 0.05,
		min         = 0,
		soft_max    = 100.0,
		step        = 0.1,
		precision   = 4
	)
	resnum : IntProperty(
		name        = "Resolution",
		description = "Bevel resolution of curve object",
		default     = 6,
		min         = 1,
		soft_max    = 100,
		step        = 1
	)
	bvl_offset_min : FloatProperty(
		name        = "Min",
		description = "Minimum bevel offset/width",
		default     = 0.0,
		min         = 0.0,
		soft_max    = 100.0,
		step        = 0.1,
		precision   = 4
	)
	bvl_offset_max : FloatProperty(
		name        = "Max",
		description = "Maximum bevel offset/width",
		default     = 0.0,
		min         = 0.0,
		soft_max    = 100.0,
		step        = 0.1,
		precision   = 4
	)
	bvl_seg : IntProperty(
		name        = "Segments",
		description = "Bevel segments",
		default     = 2,
		min         = 1,
		soft_max    = 100,
		step        = 1
	)
	bvl_angle : FloatProperty(
		name        = "Angle Limit",
		description = "Maximum angle threshold for curve points to get beveled",
		default     = radians(30),
		min         = radians(1),
		max         = radians(180),
		step        = 10,
		precision   = 3,
		subtype     = "ANGLE"
		)
	bvl_seed : IntProperty(
		name        = "Bevel Seed",
		description = "Randomize bevel offset",
		default     = 1,
		min         = 1,
		soft_max    = 10000,
		step        = 1
		)
	cuts_base : IntProperty(
		name        = "Cuts",
		description = "Number of subdivision cuts for panel object",
		default     = 0,
		min         = 0,
		soft_max    = 12,
		step        = 1
		)
	cuts_smooth : FloatProperty(
		name        = "Smooth",
		default     = 0.0,
		min         = 0.0,
		soft_max    = 1.0,
		step        = 0.1,
		precision   = 3
		)
	cut_smooth_falloff : EnumProperty(
		name = "Smooth Falloff",
		description = "Falloff profile of subdivision smoothing",
		items = (
			('SMOOTH', 'Smooth', '', 'SMOOTHCURVE', 0),
			('SPHERE', 'Sphere', '', 'SPHERECURVE', 1),
			('ROOT', 'Root', '', 'ROOTCURVE', 2),
			('SHARP', 'Sharp', '', 'SMOOTHCURVE', 3),
			('LINEAR', 'Linear', '', 'LINCURVE', 4),
			('INVERSE_SQUARE', 'Inverse_square', '', 'INVERSESQUARECURVE', 5)),
		default = 'SMOOTH')
	tri_perc : FloatProperty(
		name        = "Triangulate",
		description = "Triangulate faces",
		min         = 0,
		max         = 100,
		precision   = 0,
		default     = 0,
		subtype     = "PERCENTAGE"
		)
	mat_index : IntProperty(
		name        = "Material Index",
		description = "Material assigned to duplicates",
		default     = -1,
		min         = -1,
		max         = 32767,
		step        = 1
		)
	clamp_bvl : BoolProperty(
		name        = "Clamp Overlap (Bevel)",
		description = "Clamp the width to avoid overlap",
		default     = True
		)
	limit_body : BoolProperty(
		name        = "Limit Body",
		description = "Prevent start/end points originating from other tube bodies",
		default     = False
		)
	smooth_shade : BoolProperty(
		name        = "Shade Smooth",
		description = "Smooth shade curve object",
		default     = True
		)

	@classmethod
	def poll(cls, context):
		return context.active_object is not None and context.active_object.mode == "OBJECT"

	def curve_convert(self, obj, width, resnum, smooth=True):

		bpy.ops.object.convert(target='CURVE')

		obj.data.fill_mode = 'FULL'
		obj.data.bevel_depth = width
		obj.data.bevel_resolution = resnum

		for spline in obj.data.splines:
			spline.use_smooth = smooth

	def execute(self, context):
		obj = bpy.context.active_object
		orig_mesh = obj.data

		bm = bmesh.new()
		temp_mesh = bpy.data.meshes.new(".temp")
		bm.from_mesh(orig_mesh)

		face_sel = [f for f in bm.faces if not f.select]
		bmesh.ops.delete(bm, geom=face_sel, context='FACES')

		bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)

		if self.margin > 0:
			margin = bmesh.ops.inset_region(bm, faces=bm.faces, use_boundary=True, use_even_offset=True, \
				thickness=self.margin, depth=0.0)['faces']
			bmesh.ops.delete(bm, geom=margin, context='FACES')

		bm.to_mesh(temp_mesh)
		bm.free()

		bm = bmesh.new()
		bm.from_mesh(temp_mesh)

		bmesh.ops.subdivide_edges(bm, edges=bm.edges, smooth=self.cuts_smooth, cuts=self.cuts_base, \
			smooth_falloff=self.cut_smooth_falloff, use_grid_fill=True, use_smooth_even=True)

		if self.tri_perc:
			tris = get_tris(bm, bm.faces, self.tri_perc, self.edg_seed)
			bmesh.ops.triangulate(bm, faces=tris)

		oldv = list(bm.verts)
		idx = set([e.index for e in bm.edges])
		cells = []

		pnum = 0
		while idx and pnum < self.panel_num:
			seed(self.edg_seed)
			x = choice(list(idx))
			idx.remove(x)

			bm.edges.ensure_lookup_table()
			edg = bm.edges[x]
			cell = [x]
			walk = 0

			cell_e = set()
			last_v = None

			def add_body_limit(edge, vert):

				otv = edge.other_vert(vert)
				for link_e in otv.link_edges:
					if not link_e in cell_e: cell_e.add(link_e)

			while walk < (self.edg_length-1):
				last_e = edg; curr_v = edg.verts[0]
				if walk == 0:
					edg_verts = sample(list(last_e.verts), len(last_e.verts))
					curr_v = choice(edg_verts)
				else:
					curr_v = last_e.other_vert(last_v)
				add_body_limit(last_e, curr_v)
				link_edges = { e.index: e.calc_length() for e in curr_v.link_edges }
				if len(set(list(link_edges.keys())).intersection(set(cell))) < 2:
					for curr_e in curr_v.link_edges:
						edge_length = list(link_edges.values())
						length_solver = min(edge_length) if self.path == 'LONGEST' \
							else max(edge_length) if self.path == 'SHORTEST' else 0.0
						if curr_e.index in idx and \
							curr_e.calc_length() != length_solver:
							if not curr_e in cell_e:
								index = curr_e.index
								edg = bm.edges[index]
								idx.remove(index)
								cell.append(index)
								last_v = curr_v
								walk += 1
								break

				if last_e.index == cell[-1]:
					break

			if cell:
				for i in cell:
					x = False
					if not self.limit_body:
						if i == cell[0] or i == cell[-1]: x = True
					else: x = True

					if x:
						set_v = set()
						for v in bm.edges[i].verts:
							if not v in set_v:
								for e in v.link_edges:
									if e.index in idx: idx.remove(e.index)
								set_v.add(v)

				cells.append(cell)
				pnum += 1

		for i, edges in enumerate(cells):
			edges = [e for e in bm.edges if e.index in edges]
			ret = bmesh.ops.duplicate(bm, geom=edges)['vert_map']
			newv = [v for v in ret if not v in oldv]
			ends = list(filter(lambda v: len(v.link_edges) < 2, newv))
			bmesh.ops.extrude_vert_indiv(bm, verts=ends)
			for v1 in newv:
				source_v = list(filter(lambda v: v.co == v1.co, oldv))
				if source_v:
					v0 = source_v[0]
					seed(self.offset_seed + i)
					v1.co += (v0.normal * (v0.calc_shell_factor() \
						* uniform(self.edg_offset_min, self.edg_offset_max)))
					seed(0)

		bmesh.ops.delete(bm, geom=oldv, context='VERTS')

		if sum([self.bvl_offset_min, self.bvl_offset_max]) > 0:
			counter = 0
			cur_v = bm.verts
			for v in cur_v:
				angle = v.calc_edge_angle(None)
				if angle and \
					angle >= self.bvl_angle:
					seed(self.bvl_seed + counter)
					bvl_offset = uniform(self.bvl_offset_min, self.bvl_offset_max)
					ret = bmesh.ops.bevel(
						bm,
						geom            = [v],
						offset          = bvl_offset,
						offset_type     = 'OFFSET',
						segments        = self.bvl_seg,
						profile         = 0.5,
						affect          = 'VERTICES',
						clamp_overlap	= self.clamp_bvl
						)
					counter += 1
					bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)

		bm.to_mesh(temp_mesh)
		bm.free()

		if temp_mesh.vertices:
			new_obj = bpy.data.objects.new(obj.name + "_RPipes", temp_mesh)
			orig_loc, orig_rot, orig_scale = obj.matrix_world.decompose()
			new_obj.scale = orig_scale
			new_obj.rotation_euler = orig_rot.to_euler()
			new_obj.location = orig_loc
			new_obj.data.use_auto_smooth = obj.data.use_auto_smooth
			new_obj.data.auto_smooth_angle = obj.data.auto_smooth_angle

			context.scene.collection.objects.link(new_obj)

			new_obj.select_set(True)
			select_isolate(context, new_obj, [obj, new_obj])
			self.curve_convert(new_obj, self.width, self.resnum, self.smooth_shade)
			copy_modifiers([obj, new_obj], mod_types=['MIRROR'])
			assign_mat(self, obj, new_obj, self.mat_index)

			if context.scene.rflow_props.select_active:
				objs = [obj, new_obj]
				select_isolate(context, obj, objs)
		else:
			bpy.data.meshes.remove(temp_mesh)

		return {"FINISHED"}

	def draw(self, context):
		layout = self.layout
		col = layout.column()
		col.separator(factor=0.1)
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Path:")
		row.row(align=True).prop(self, "path", expand = True)
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Amount:")
		row.row(align=True).prop(self, "panel_num", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Length:")
		row.row(align=True).prop(self, "edg_length", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Seed:")
		row.row(align=True).prop(self, "edg_seed", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Offset:")
		split = row.split(factor=0.5, align=True)
		split.row(align=True).prop(self, "edg_offset_min")
		split.row(align=True).prop(self, "edg_offset_max")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Offset Seed:")
		row.row(align=True).prop(self, "offset_seed", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Margin:")
		row.row(align=True).prop(self, "margin", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Curve:")
		split = row.split(factor=0.5, align=True)
		split.row(align=True).prop(self, "width")
		split.row(align=True).prop(self, "resnum")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Bvl Offset:")
		split = row.split(factor=0.5, align=True)
		split.row(align=True).prop(self, "bvl_offset_min")
		split.row(align=True).prop(self, "bvl_offset_max")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Bvl Seg/Angle:")
		split = row.split(factor=0.5, align=True)
		split.row(align=True).prop(self, "bvl_seg")
		split.row(align=True).prop(self, "bvl_angle")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Bevel Seed:")
		row.row(align=True).prop(self, "bvl_seed", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Subdivision:")
		split = row.split(factor=0.5, align=True)
		split.row(align=True).prop(self, "cuts_base")
		split.row(align=True).prop(self, "cuts_smooth")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Smooth Falloff:")
		row.row(align=True).prop(self, "cut_smooth_falloff", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Triangulate:")
		row.row(align=True).prop(self, "tri_perc", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Material Index:")
		row.row(align=True).prop(self, "mat_index", text="")
		col.separator(factor=0.5)
		col.prop(self, "clamp_bvl")
		col.prop(self, "limit_body")
		col.prop(self, "smooth_shade")

	def invoke(self, context, event):
		obj = context.active_object
		self.edg_seed = 1
		self.offset_seed = 1
		self.mat_index = -1

		obj.update_from_editmode()
		has_face = [f for f in obj.data.polygons if f.select]

		if has_face:
			init_props(self, event, ops='rtubes')
			prefs = context.preferences.addons[__name__].preferences
			if prefs.use_confirm:
				return context.window_manager.invoke_props_dialog(self)
			else:
				return context.window_manager.invoke_props_popup(self, event)
		else:
			self.report({'WARNING'}, "No faces selected.")
			return {"FINISHED"}

class MESH_OT_r_cables(Operator):
	'''Create scatter details'''
	bl_idname = 'rand_cables.rflow'
	bl_label = 'Random Cables'
	bl_options = {'REGISTER', 'UNDO', 'PRESET'}

	points_seed : IntProperty(
		name        = "Seed",
		description = "Randomize cable points",
		default     = 1,
		min         = 1,
		soft_max    = 10000,
		step        = 1
		)
	cable_num : IntProperty(
		name        = "Number of Cables",
		description = "Number of cable to randomize",
		default     = 10,
		min         = 1,
		soft_max    = 100,
		step        = 1
		)
	steps : IntProperty(
		name		= "Steps",
		description	= "Resolution of the curve",
		default		= 16,
		min			= 2,
		max			= 1024,
		)
	slack_seed : IntProperty(
		name        = "Seed",
		description = "Randomize cable slack",
		default     = 1,
		min         = 1,
		soft_max    = 10000,
		step        = 1
		)
	slack_min : FloatProperty(
		name		= "Min Slack",
		description	= "Minimum cable slack",
		precision	= 4,
		default		= 0.1,
		min			= 0.0001,
		max			= 100.0
		)
	slack_max : FloatProperty(
		name		= "Max Slack",
		description	= "Maximum cable slack",
		precision	= 4,
		default		= 2.0,
		min			= 0.0001,
		max			= 100.0
		)
	spline_type : EnumProperty(
		name		= "Spline Type",
		description = "Spline type",
		items		= [('POLY', "Poly", "Poly spline"),
					('BEZIER', "Bezier", "Bezier spline")],
		default		= 'BEZIER',
		)
	res_u : IntProperty(
		name		= "Resolution U",
		description = "Curve resolution u",
		default		= 8,
		min			= 0,
		max			= 64
		)
	bvl_depth : FloatProperty(
		name		= "Radius",
		description	= "Bevel depth",
		default		= 0.01,
		min			= 0.0,
		precision	= 3
		)
	bvl_res : IntProperty(
		name		= "Resolution",
		description	= "Bevel resolution",
		default		= 0,
		min			= 0,
		max			= 32
		)
	extrude : FloatProperty(
		name		= "Extrude",
		description	= "Extrude amount",
		default		= 0.0,
		min			= 0.0,
		precision	= 3
		)
	twist_mode : EnumProperty(
		name		= "Twisting",
		description	= "Twist method, type of tilt calculation",
		items		= [('Z_UP', "Z-Up", 'Z Up'),
					('MINIMUM', "Minimum", "Minimum"),
					('TANGENT', "Tangent", "Tangent")],
		default		= 'MINIMUM',
		)
	twist_smooth : FloatProperty(
		name		= "Smooth",
		description	= "Twist smoothing amount for tangents",
		default		= 0.0,
		min			= 0.0,
		precision	= 3
		)
	tilt : FloatProperty(
		name		= "Tilt",
		description	= "Spline handle tilt",
		default		= 0.0,
		precision	= 3
		)
	r_radius : FloatProperty(
		name		= "Radius",
		description	= "Randomise radius of spline controlpoints",
		default		= 0.0,
		min			= 0.0,
		precision	= 3
		)
	radius_seed : IntProperty(
		name        = "Radius Seed",
		description = "Randomize cable radius",
		default     = 1,
		min         = 1,
		soft_max    = 10000,
		step        = 1
		)
	offset : FloatProperty(
		name        = "Offset",
		description = "Offset source mesh by this amount to produce overlap",
		default     = 0.0,
		soft_min    = -1.0,
		soft_max    = 1.0,
		step        = 0.01,
		precision   = 4
		)
	margin : FloatProperty(
		name        = "Margin",
		description = "Margin from boundary edges",
		default     = 0.0,
		soft_min    = -1.0,
		soft_max    = 1.0,
		step        = 0.1,
		precision   = 4
		)
	mat_index : IntProperty(
		name        = "Material Index",
		description = "Material assigned to duplicates",
		default     = -1,
		min         = -1,
		max         = 32767,
		step        = 1
		)
	island_limit : BoolProperty(
		name        = "Limit Origin To Islands",
		description = "Make sure start and end points are different per face island selected",
		default     = False
		)
	join_curves : BoolProperty(
		name        = "Join Curves",
		description = "Join curves into one object",
		default     = True
		)

	@classmethod
	def poll(cls, context):
		return context.active_object is not None and context.active_object.mode == "OBJECT"

	def point_on_triangle(self, face):
		'''https://blender.stackexchange.com/a/221597'''

		a, b, c = map(lambda v: v.co, face.verts)
		a2b = b - a
		a2c = c - a
		height = triangular(low=0.0, high=1.0, mode=0.0)

		return a + a2c*height + a2b*(1-height) * random(), face

	def catenary_curve(
				self,
				start=[-2, 0, 2],
				end=[2, 0, 2],
				steps=24,
				a=2.0
				):

		points = []
		lx = end[0] - start[0]
		ly = end[1] - start[1]
		lr = sqrt(pow(lx, 2) + pow(ly, 2))
		lv = lr / 2 - (end[2] - start[2]) * a / lr
		zv = start[2] - pow(lv, 2) / (2 * a)
		slx = lx / steps
		sly = ly / steps
		slr = lr / steps
		i = 0

		while i <= steps:
			x = start[0] + i * slx
			y = start[1] + i * sly
			z = zv + pow((i * slr) - lv, 2) / (2 * a)
			points.append([x, y, z])
			i += 1

		return points

	def add_curve_object(
				self,
				verts,
				matrix,
				spline_name="Spline",
				spline_type='BEZIER',
				resolution_u=12,
				bevel=0.0,
				bevel_resolution=0,
				extrude=0.0,
				spline_radius=0.0,
				twist_mode='MINIMUM',
				twist_smooth=0.0,
				tilt=0.0
				):

		cur_data = bpy.data.curves.new(spline_name, 'CURVE')
		cur_data.dimensions = '3D'
		spline = cur_data.splines.new(spline_type)
		curve = bpy.data.objects.new(spline_name, cur_data)
		spline.radius_interpolation = 'BSPLINE'
		spline.tilt_interpolation = 'BSPLINE'

		if spline_type == 'BEZIER':
			spline.bezier_points.add(int(len(verts) - 1))
			for i in range(len(verts)):
				spline.bezier_points[i].co = verts[i]
				spline.bezier_points[i].handle_right_type = 'AUTO'
				spline.bezier_points[i].handle_left_type = 'AUTO'
				spline.bezier_points[i].radius += spline_radius * random()
				spline.bezier_points[i].tilt = radians(tilt)
		else:
			spline.points.add(int(len(verts) - 1))
			for i in range(len(verts)):
				spline.points[i].co = verts[i][0], verts[i][1], verts[i][2], 1

		bpy.context.scene.collection.objects.link(curve)
		curve.data.resolution_u = resolution_u
		curve.data.fill_mode = 'FULL'
		curve.data.bevel_depth = bevel
		curve.data.bevel_resolution = bevel_resolution
		curve.data.extrude = extrude
		curve.data.twist_mode = twist_mode
		curve.data.twist_smooth = twist_smooth
		curve.matrix_world = matrix

		return curve

	def execute(self, context):
		obj = bpy.context.active_object
		sel_objs = context.selected_objects
		limit = self.island_limit

		bm = bmesh.new()
		temp_mesh = bpy.data.meshes.new(".temp")

		for o in sel_objs:
			clone = o.data.copy()
			pivot = o.matrix_world.inverted() @ obj.matrix_world.translation
			clone.transform(Matrix.Translation(-pivot))
			bm.from_mesh(clone)
			bmesh.ops.delete(bm, geom=[f for f in bm.faces if not f.select], context='FACES')
			bpy.data.meshes.remove(clone)

		if not bm.faces:
			self.report({'WARNING'}, "No faces selected.")
			bm.free()
			bpy.data.meshes.remove(temp_mesh)
			return {"FINISHED"}

		if bool(self.offset):
			for v in bm.verts:
				v.co -= v.normal * (v.calc_shell_factor() * self.offset)

		if bool(self.margin):
			margin = bmesh.ops.inset_region(bm, faces=bm.faces, use_boundary=True, use_even_offset=True, \
				thickness=self.margin, depth=0.0)['faces']
			bmesh.ops.delete(bm, geom=margin, context='FACES')

		triangles = bmesh.ops.triangulate(bm, faces=bm.faces)['faces']
		surfaces = map(lambda t: t.calc_area(), triangles)
		seed(self.points_seed)
		listp = choices(population=triangles, weights=surfaces, k=self.cable_num*4)
		points = list(map(self.point_on_triangle, listp))

		if limit:
			islands = get_islands(obj, bm, use_bm=True)

		amount = self.cable_num
		pairs = []

		def island_index(l1, l2):

			idx = 0
			for i, p in enumerate(l2):
				if all(x in p for x in l1): idx = i

			return idx


		while amount > 0:
			start = choice(points)
			points.remove(start)
			end = choice(points)
			points.remove(end)

			if limit:
				vl1 = island_index([v.index for v in start[1].verts], islands)
				vl2 = island_index([v.index for v in end[1].verts], islands)

			if start[0] != end[0]:
				if limit:
					if vl1 != vl2:
						pairs.append([start[0], end[0]])
				else: pairs.append([start[0], end[0]])

			amount -= 1
			if not points: break

		bm.free()
		bpy.data.meshes.remove(temp_mesh)

		new_curves = []
		for i, p in enumerate(pairs):
			try:
				steps = self.steps
				seed(self.slack_seed + i)
				slack = uniform(self.slack_min, self.slack_max)
				points1 = self.catenary_curve(
						obj.matrix_world @ p[0],
						obj.matrix_world @ p[1],
						steps,
						slack
						)
				seed(self.radius_seed)
				curve = self.add_curve_object(
						points1,
						Matrix(),
						'RCables',
						self.spline_type,
						self.res_u,
						self.bvl_depth,
						self.bvl_res,
						self.extrude,
						self.r_radius,
						self.twist_mode,
						self.twist_smooth,
						self.tilt
						)
				new_curves.append(curve)
				copy_modifiers([obj, curve], mod_types=['MIRROR'])
			except:
				pass

		if new_curves:
			bpy.ops.object.select_all(action='DESELECT')
			for o in new_curves:
				o.select_set(True)
				if o == new_curves[0]:
					context.view_layer.objects.active = o
					copy_modifiers([obj, o], mod_types=['MIRROR'])
					assign_mat(self, obj, o, self.mat_index)

			if self.join_curves: bpy.ops.object.join()

		if not context.scene.rflow_props.select_active == False:
			bpy.ops.object.select_all(action='DESELECT')
			for o in sel_objs:
				o.select_set(True)
				if o == sel_objs[-1]:
					context.view_layer.objects.active = o

		return {"FINISHED"}

	def draw(self, context):
		layout = self.layout
		col = layout.column()
		col.separator(factor=0.1)
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Spline Type:")
		row.row(align=True).prop(self, "spline_type", expand = True)
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Cable Amount:")
		row.row(align=True).prop(self, "cable_num", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Origin Seed:")
		row.row(align=True).prop(self, "points_seed", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Slack:")
		split = row.split(factor=0.5, align=True)
		split.row(align=True).prop(self, "slack_min")
		split.row(align=True).prop(self, "slack_max")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Slack Seed:")
		row.row(align=True).prop(self, "slack_seed", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Bevel:")
		split = row.split(factor=0.5, align=True)
		split.row(align=True).prop(self, "bvl_depth")
		split.row(align=True).prop(self, "bvl_res")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Resolution U:")
		row.row(align=True).prop(self, "res_u", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Steps:")
		row.row(align=True).prop(self, "steps", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Extrude:")
		row.row(align=True).prop(self, "extrude", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Twist Mode:")
		row.row(align=True).prop(self, "twist_mode", expand = True)
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Twist Smooth:")
		row.row(align=True).prop(self, "twist_smooth", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Tilt:")
		row.row(align=True).prop(self, "tilt", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Radius:")
		row.row(align=True).prop(self, "r_radius", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Radius Seed:")
		row.row(align=True).prop(self, "radius_seed", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Offset:")
		row.row(align=True).prop(self, "offset", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Margin:")
		row.row(align=True).prop(self, "margin", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Material Index:")
		row.row(align=True).prop(self, "mat_index", text="")
		col.separator(factor=0.5)
		col.prop(self, "island_limit")
		col.prop(self, "join_curves")

	def invoke(self, context, event):
		self.points_seed = 1
		self.slack_seed = 1

		prefs = context.preferences.addons[__name__].preferences
		if prefs.use_confirm:
			return context.window_manager.invoke_props_dialog(self)
		else:
			return context.window_manager.invoke_props_popup(self, event)

class MESH_OT_r_vertex_color(Operator):
	'''Randomize vertex color fill for selected objects'''
	bl_idname = 'rand_vcol.rflow'
	bl_label = 'Random VColor'
	bl_options = {'REGISTER', 'UNDO', 'PRESET'}

	vgname : StringProperty(
		name        = "Vertex Color",
		description = "Name of vertex color",
		default		= "Vertex Color"
		)
	vclist : StringProperty(
		name        = "Vertex Color List",
		description = "Vertex color group list from all selected objects"
		)
	vcolors : CollectionProperty(type=PropertyGroup)
	color_max : FloatVectorProperty(
		name        = "Color Max",
		description = "Vertex color maximum rgb values",
		subtype     = 'COLOR_GAMMA',
		default     = (1.0,1.0,1.0),
		size        = 3,
		min         = 0.0,
		max         = 1.0
		)
	color_min : FloatVectorProperty(
		name        = "Color Min",
		description = "Vertex color minimum rgb values",
		subtype     = 'COLOR_GAMMA',
		default     = (0.0,0.0,0.0),
		size        = 3,
		min         = 0.0,
		max         = 1.0
		)
	limit : EnumProperty(
		name = 'Limit',
		items = (
			('OBJECT', 'Object',''),
			('ISLAND', 'Island',''),
			('SELECT', 'Selected','')),
		default = 'OBJECT'
		)
	col_seed : IntProperty(
		name        = "H",
		description = "Color rgb randomize seed",
		default     = 1,
		min         = 1,
		soft_max    = 10000,
		step        = 1
		)
	sat_min : FloatProperty(
		name        = "Min",
		description = "Minimum random value for saturation factor of hsv color",
		default     = 0,
		min         = 0,
		max         = 1.0,
		step        = 0.1,
		precision   = 3
	)
	sat_max : FloatProperty(
		name        = "Max",
		description = "Maximum random value for saturation factor of hsv color",
		default     = 1.0,
		min         = 0,
		max         = 1.0,
		step        = 0.1,
		precision   = 3
	)
	sat_seed : IntProperty(
		name        = "S",
		description = "Color saturation randomize seed",
		default     = 1,
		min         = 1,
		soft_max    = 10000,
		step        = 1
		)
	val_min : FloatProperty(
		name        = "Min",
		description = "Minimum random value for value factor of hsv color",
		default     = 0,
		min         = 0,
		max         = 1.0,
		step        = 0.1,
		precision   = 3
	)
	val_max : FloatProperty(
		name        = "Max",
		description = "Maximum random value for value factor of hsv color",
		default     = 1.0,
		min         = 0,
		max         = 1.0,
		step        = 0.1,
		precision   = 3
	)
	val_seed : IntProperty(
		name        = "V",
		description = "Color value randomize seed",
		default     = 1,
		min         = 1,
		soft_max    = 10000,
		step        = 1
		)
	offset : IntProperty(
		name        = "Island Offset",
		description = "Number offset before changing island color",
		default     = 1,
		min         = 1,
		soft_max    = 100,
		step        = 1
		)
	obj_offset : IntProperty(
		name        = "Object Offset",
		description = "Number offset before changing object color",
		default     = 1,
		min         = 1,
		soft_max    = 100,
		step        = 1
		)
	single_color : BoolProperty(
		name        = "Single Color",
		description = "Assign single color for faces selected per object",
		default     = False
		)
	use_hg : EnumProperty(
		name = "Hard Gradient",
		description = "Use hard gradient for HSV",
		items = (
			('COL', 'Hue','Use hard gradient for color or hue'),
			('SAT', 'Saturation','Use hard gradient for saturation'),
			('VAL', 'Value','Use hard gradient for value')),
		options = {"ENUM_FLAG"})
	col_hg_stops : IntProperty(
		name        = "H",
		description = "Number of stops for hard gradient hue set at even spacing",
		default     = 2,
		min         = 2,
		soft_max    = 100,
		step        = 1
	)
	sat_hg_stops : IntProperty(
		name        = "S",
		description = "Number of stops for hard gradient saturation set at even spacing",
		default     = 2,
		min         = 2,
		soft_max    = 100,
		step        = 1
	)
	val_hg_stops : IntProperty(
		name        = "V",
		description = "Number of stops for hard gradient value set at even spacing",
		default     = 2,
		min         = 2,
		soft_max    = 100,
		step        = 1
	)

	@classmethod
	def poll(cls, context):
		return context.active_object is not None and context.active_object.mode == "OBJECT"

	def get_random_color(self, var):

		col_hgs = self.col_hg_stops - 1
		sat_hgs = self.sat_hg_stops - 1
		val_hgs = self.val_hg_stops - 1

		def make_stops(minv, maxv, nstops):

			d = abs(maxv - minv)
			stop_value = round(d/nstops, 3)
			val = min(minv, maxv) + (stop_value * randint(0, nstops))

			return val

		seed(var + self.col_seed)
		if not "COL" in self.use_hg:
			r, g, b = [uniform(self.color_min[i], n) for i, n in enumerate(self.color_max)]
		else:
			r, g, b = [make_stops(self.color_min[i], n, col_hgs) for i, n in enumerate(self.color_max)]
		h, s, v = colorsys.rgb_to_hsv(r, g, b)

		seed(var + self.sat_seed)
		if not "SAT" in self.use_hg:
			s = uniform(self.sat_min, self.sat_max)
		else:
			s = make_stops(self.sat_min, self.sat_max, sat_hgs)

		seed(var + self.val_seed)
		if not "VAL" in self.use_hg:
			v = uniform(self.val_min, self.val_max)
		else:
			v = make_stops(self.val_min, self.val_max, val_hgs)

		return colorsys.hsv_to_rgb(h, s, v) + (1.0,)

	def execute(self, context):
		act_obj = context.active_object
		if act_obj: act_obj.select_set(True)

		objs = context.selected_objects

		def loops_color_layer(face, color):

			for loop in face.loops:
				loop[color_layer] = color

		x = 0
		obj_offset = 0
		for o in objs:
			if o.type == 'MESH':
				mesh = o.data

				if self.vclist:
					layer = self.vclist
				else:
					layer = self.vgname if len(self.vgname) else "Vertex Color"
				color_layer = mesh.vertex_colors.get(layer) \
					or mesh.vertex_colors.new(name=layer)

				mesh.vertex_colors.active = mesh.vertex_colors[layer]

				bm = bmesh.new()
				bm.from_mesh(mesh)

				color_layer = bm.loops.layers.color.get(layer) \
					or bm.loops.layers.color.new(layer)

				if self.limit == 'OBJECT':
					vertex_color = self.get_random_color(x)
					for f in bm.faces:
						loops_color_layer(f, vertex_color)
				elif self.limit == 'ISLAND':
					listf = list(bm.faces)
					if listf != None:
						i = x
						offset = 0
						islands = get_islands(o, bm, use_bm=True)

						for lp in islands:
							vertex_color = self.get_random_color(i)
							faces = [bm.verts[idx].link_faces for idx in lp]
							faces = undupe(list(chain.from_iterable(faces)))
							loops = [f.loops for f in faces]
							loops = undupe(list(chain.from_iterable(loops)))
							for l in loops:
								l[color_layer] = vertex_color

							offset += 1
							if offset == self.offset:
								i += 1
								offset = 0
				else:
					listf = [f for f in bm.faces if f.select]

					if self.single_color:
						vertex_color = self.get_random_color(x)
						for f in listf:
							loops_color_layer(f, vertex_color)
					else:
						linked_faces = set()

						if listf != None:
							i = x
							offset = 0
							while listf:
								traversal_stack = [listf.pop()]

								vertex_color = self.get_random_color(i)
								loops_color_layer(traversal_stack[0], vertex_color)

								while len(traversal_stack) > 0:
									f_curr = traversal_stack.pop()
									linked_faces.add(f_curr)

									for e in f_curr.edges:
										if e.is_contiguous and e.select:
											for f_linked in e.link_faces:
												if f_linked not in linked_faces and f_linked.select:
													traversal_stack.append(f_linked)
													loops_color_layer(f_linked, vertex_color)
													if f_linked in listf: listf.remove(f_linked)

								offset += 1
								if offset == self.offset:
									i += 1
									offset = 0

				bm.to_mesh(mesh)
				bm.free()
				mesh.update()

			obj_offset += 1
			if obj_offset == self.obj_offset:
				x += 1
				obj_offset = 0

		return {"FINISHED"}

	def draw(self, context):
		layout = self.layout
		col = layout.column()
		col.separator()
		col.template_color_picker(self, "color_max", value_slider=False)
		col.separator()
		row = col.row().split(factor=0.27, align=True)
		row.label(text="VG Name:")
		row.row(align=True).prop(self, "vgname", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="")
		row.prop_search(
			self,
			"vclist",
			self,
			"vcolors",
			text="",
			icon = "GROUP_VCOL"
			)
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Limit:")
		row.row(align=True).prop(self, "limit", expand=True)
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Hue Max:")
		row.row(align=True).prop(self, "color_max", text="", expand=True)
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Hue Min:")
		row.row(align=True).prop(self, "color_min", text="", expand=True)
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Saturation:")
		split = row.split(factor=0.5, align=True)
		split.row(align=True).prop(self, "sat_min")
		split.row(align=True).prop(self, "sat_max")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Value:")
		split = row.split(factor=0.5, align=True)
		split.row(align=True).prop(self, "val_min")
		split.row(align=True).prop(self, "val_max")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Random Seeds:")
		split = row.split(factor=0.33, align=True)
		split.row(align=True).prop(self, "col_seed")
		split.row(align=True).prop(self, "sat_seed")
		split.row(align=True).prop(self, "val_seed")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Hard Gradient:")
		row.row(align=True).prop(self, "use_hg", expand=True)
		if self.use_hg != set():
			row = col.row().split(factor=0.27, align=True)
			row.label(text="HG Stops:")
			split = row.split(factor=0.33, align=True)
			split.row(align=True).prop(self, "col_hg_stops")
			split.row(align=True).prop(self, "sat_hg_stops")
			split.row(align=True).prop(self, "val_hg_stops")
		if self.limit != 'OBJECT':
			row = col.row().split(factor=0.27, align=True)
			row.label(text="Island Offset:")
			row.row(align=True).prop(self, "offset", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Object Offset:")
		row.row(align=True).prop(self, "obj_offset", text="")
		if self.limit == 'SELECT':
			col.separator(factor=0.5)
			col.prop(self, "single_color")

	def invoke(self, context, event):
		self.color_seed = 1
		self.sat_seed = 1
		self.val_seed = 1
		self.vclist = ""

		init_props(self, event, ops='rvcol')
		context.space_data.shading.color_type = 'VERTEX'

		for o in context.selected_objects:
			if o.type == 'MESH':
				for vc in o.data.vertex_colors:
					if vc.name not in self.vcolors:
						newListItem = self.vcolors.add()
						newListItem.name = vc.name

		prefs = context.preferences.addons[__name__].preferences
		if prefs.use_confirm:
			return context.window_manager.invoke_props_dialog(self)
		else:
			return context.window_manager.invoke_props_popup(self, event)

class MESH_OT_make_flanges(Operator):
	'''Generate flanges and couplings for curve objects'''
	bl_idname = 'make_flanges.rflow'
	bl_label = 'Flanges/Couplings'
	bl_options = {'REGISTER', 'UNDO', 'PRESET'}

	mesh_type : EnumProperty(
		name = 'Type',
		description = "Type",
		items = (
			('CYLINDER', 'Cylinder',''),
			('CUSTOM', 'Custom',''),
			('COLLECTION', 'Collection','')),
		default = 'CYLINDER'
		)
	list : StringProperty(
		name        = "Import Mesh",
		description = "Use selected for cap design"
		)
	meshes : CollectionProperty(type=PropertyGroup)
	list_col : StringProperty(
		name        = "Collections",
		description = "Collection objects for scatter"
		)
	collections : CollectionProperty(type=PropertyGroup)
	coll_seed : IntProperty(
		name        = "Object Seed",
		description = "Randomize seed for collection objects",
		default     = 1,
		min         = 1,
		soft_max    = 10000,
		step        = 1
		)
	amount : IntProperty(
		name        = "Amount",
		description = "Number of couplings to generate",
		default     = 1,
		min         = 1,
		soft_max    = 100,
		step        = 1
		)
	step : IntProperty(
		name        = "Step",
		description = "Interval spacing between flanges/couplings",
		default     = 2,
		min         = 1,
		soft_max    = 100,
		step        = 1
		)
	cap_offset : FloatProperty(
		name        = "Caps",
		description = "Offset generated cap flanges",
		default     = 0,
		soft_min	= -100.0,
		soft_max	= 100.0,
		step        = 0.01,
		precision   = 3
		)
	bod_offset : FloatProperty(
		name        = "Body",
		description = "Offset generated body couplings",
		default     = 0,
		soft_min	= -100.0,
		soft_max	= 100.0,
		step        = 0.01,
		precision   = 3
		)
	radius : FloatProperty(
		name        = "Radius",
		description = "Radius of flanges/couplings",
		default     = 0.05,
		min			= 0,
		step        = 0.01,
		precision   = 3
		)
	rot_z : FloatProperty(
		name        = "Rotation",
		description = "Rotation of flanges/couplings",
		default     = 0,
		min         = radians(-360),
		max         = radians(360),
		step        = 10,
		precision   = 3,
		subtype     = "ANGLE"
		)
	rot_seed : IntProperty(
		name        = "Rotation Seed",
		description = "Randomize seed for rotation of flanges/couplings",
		default     = 1,
		min         = 1,
		soft_max    = 10000,
		step        = 1
		)
	depth : FloatProperty(
		name        = "Depth",
		description = "Depth of flanges/couplings",
		default     = 0.005,
		min			= 0,
		step        = 0.01,
		precision   = 3
		)
	segment : IntProperty(
		name        = "Segment",
		description = "Total segment of flanges/couplings",
		default     = 6,
		min         = 3,
		max         = 500
		)
	mat_index : IntProperty(
		name        = "Material Index",
		description = "Material assigned to duplicates",
		default     = -1,
		min         = -1,
		max         = 32767,
		step        = 1
		)
	limit : EnumProperty(
		name = 'Limit',
		description = "Limit to caps, body or none",
		items = (
			('NONE', 'None',''),
			('CAPS', 'Caps',''),
			('BODY', 'Body','')),
		default = 'NONE'
		)
	even_count : BoolProperty(
		name        = "Even Count",
		description = "Even count of flanges/couplings along tube body",
		default     = True
		)

	@classmethod
	def poll(cls, context):
		return context.active_object is not None and context.active_object.mode == "OBJECT" \
			and context.active_object.type == "CURVE"

	def make_fittings(self, curve_objs):

		bm = bmesh.new()
		temp_mesh = bpy.data.meshes.new(".temp")

		def add_fittings(curve, bm, pos, tangent, idx):

			context = bpy.context
			type = self.mesh_type

			def orient_to_curve(bm_cont, vlist):

				if type in ['CUSTOM', 'COLLECTION']:
					bmesh.ops.scale(
						bm_temp,
						vec     = Vector(tuple([self.radius] * 3)),
						space   = Matrix(),
						verts   = vlist
						)

				z = self.rot_z
				if self.rot_seed > 1:
					seed(self.rot_seed + idx)
					z = uniform(360-z, z)

				rot_axis = [0 , 0, z]

				bmesh.ops.rotate(
					bm_cont,
					verts   = vlist,
					cent    = Vector(),
					matrix  = Euler(Vector(rot_axis)).to_matrix()
					)

				bmesh.ops.translate(
						bm_cont,
						verts   = vlist,
						vec     = pos
						)

				quat = tangent.to_track_quat('-Z', 'Y')
				mat = curve.matrix_world @ quat.to_matrix().to_4x4()
				rot = mat.to_3x3().normalized()

				_, orig_rot, _ = curve.matrix_world.decompose()
				bmesh.ops.rotate(
						bm_cont,
						verts   = vlist,
						cent    = pos,
						matrix  = orig_rot.to_matrix().inverted() @ rot
						)

			if type == 'CYLINDER':
				listv = bmesh.ops.create_cone(
					bm,
					cap_ends    = True,
					segments    = self.segment,
					radius1     = self.radius,
					radius2     = self.radius,
					depth       = self.depth
					)['verts']

				orient_to_curve(bm, listv)

			if type == 'CUSTOM':
				if self.list:
					file_name = self.list
					custom_data = bpy.data.meshes.get(file_name + "_rflow_mesh", None)
					if not custom_data:
						if file_name.find(".stl") == -1: file_name += ".stl"
						filepath = os.path.join(os.path.dirname(
							os.path.abspath(__file__)), "./flanges/" + file_name)
						bpy.ops.import_mesh.stl(filepath=filepath, global_scale=1.0)

						import_obj = context.selected_objects[0]
						custom_data = import_obj.data
						custom_data.name += "_rflow_mesh"

						curve.select_set(True)
						context.view_layer.objects.active = curve

						bpy.data.objects.remove(import_obj)

					bm_temp = bmesh.new()
					temp_mesh1 = bpy.data.meshes.new(".temp1")
					bm_temp.from_mesh(custom_data)

					orient_to_curve(bm_temp, bm_temp.verts[:])

					bm_temp.to_mesh(temp_mesh1)
					bm_temp.free()

					bm.from_mesh(temp_mesh1)
					bpy.data.meshes.remove(temp_mesh1)

			if type == 'COLLECTION':
				objs = bpy.data.collections.get(self.list_col)
				if objs:
					mesh_objs = [o for o in bpy.data.collections.get(self.list_col).all_objects \
						if o.type == 'MESH']
					if mesh_objs:
						seed(self.coll_seed + idx)
						rand_obj = choice(mesh_objs)
						coll_obj = bpy.data.objects.get(rand_obj.name)

						bm_temp = bmesh.new()
						temp_mesh2 = bpy.data.meshes.new(".temp2")
						bm_temp.from_mesh(coll_obj.data)

						orient_to_curve(bm_temp, bm_temp.verts[:])

						bm_temp.to_mesh(temp_mesh2)
						bm_temp.free()

						bm.from_mesh(temp_mesh2)
						bpy.data.meshes.remove(temp_mesh2)

		for o in curve_objs:
			curve = o
			for spline in curve.data.splines:
				save_type = spline.type
				spline.type = 'BEZIER'

				bez_points = spline.bezier_points
				bez_len = len(bez_points)

				if bez_len >= 2:
					if not self.limit in ['BODY']:
						cap_points = [[bez_points[0], bez_points[1]], \
								[bez_points[-1], bez_points[-2]]]
						for n, i in enumerate(cap_points):
							p1 = i[0].co; p2 = i[1].co
							tan = p1 - p2
							tan.normalize()
							p3 = p1 - (tan * self.cap_offset)
							add_fittings(curve, bm, p3, tan, n)

					if not self.limit in ['CAPS']:
						points_on_curve = []; total_length = []
						i_range = range(1, bez_len, 1)
						for i in i_range:
							curr_point = bez_points[i-1]
							next_point = bez_points[i]

							delta = (curr_point.co - next_point.co).length
							count = int(max(1, delta/0.01)) + 1

							calc_points = geometry.interpolate_bezier(
								curr_point.co,
								curr_point.handle_right,
								next_point.handle_left,
								next_point.co,
								count)

							if i != bez_len - 1:
								calc_points.pop()

							points_on_curve += calc_points
							total_length.append(delta)

						spline.type = save_type

						total_points = len(points_on_curve)
						p_range = range(1, total_points, 1)

						if not self.even_count:
							tl = (sum(total_length)/self.step) * self.amount
							fnum = int(ceil(total_points/tl))
						else:
							fnum = int(ceil(total_points/(self.amount + 1)))

						for i in p_range:
							x = i + int(ceil(total_points*self.bod_offset))
							if x % fnum == 0:
								p1 = points_on_curve[i-1]
								p2 = points_on_curve[i]
								tan = p1 - p2
								tan.normalize()
								add_fittings(curve, bm, p1, tan, x)

		bm.to_mesh(temp_mesh)
		bm.free()

		obj = curve_objs[0]
		new_obj = bpy.data.objects.new(obj.name + "_Flanges", temp_mesh)
		orig_loc, orig_rot, orig_scale = obj.matrix_world.decompose()
		new_obj.scale = orig_scale
		new_obj.rotation_euler = orig_rot.to_euler()
		new_obj.location = orig_loc
		new_obj.data.use_auto_smooth = True

		bpy.context.scene.collection.objects.link(new_obj)

		copy_modifiers([obj, new_obj], mod_types=['MIRROR'])
		assign_mat(self, obj, new_obj, self.mat_index)

	def execute(self, context):
		act_obj = context.active_object
		if act_obj: act_obj.select_set(True)

		curve_objs = []
		for o in context.selected_objects:
			if o.type == 'CURVE':
				curve_objs.append(o)

		self.make_fittings(curve_objs)

		return {"FINISHED"}

	def draw(self, context):
		layout = self.layout
		col = layout.column()
		col.separator(factor=0.1)
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Type:")
		row.row(align=True).prop(self, "mesh_type", expand=True)
		if self.mesh_type == 'CUSTOM':
			row = col.row().split(factor=0.27, align=True)
			row.label(text="Import Mesh:")
			row.prop_search(
				self,
				"list",
				self,
				"meshes",
				text="",
				icon = "MESH_DATA"
				)
		if self.mesh_type == 'COLLECTION':
			row = col.row().split(factor=0.27, align=True)
			row.label(text="Collection:")
			row.prop_search(
				self,
				"list_col",
				self,
				"collections",
				text="",
				icon = "OUTLINER_COLLECTION"
				)
			row = col.row().split(factor=0.27, align=True)
			row.label(text="Object Seed:")
			row.row(align=True).prop(self, "coll_seed", text="")
		if self.even_count:
			row = col.row().split(factor=0.27, align=True)
			row.label(text="Amount:")
			row.row(align=True).prop(self, "amount", text="")
		else:
			row = col.row().split(factor=0.27, align=True)
			row.label(text="Quantity:")
			split = row.split(factor=0.5, align=True)
			split.row(align=True).prop(self, "amount")
			split.row(align=True).prop(self, "step")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Offset:")
		split = row.split(factor=0.5, align=True)
		split.row(align=True).prop(self, "cap_offset")
		split.row(align=True).prop(self, "bod_offset")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Radius:")
		row.row(align=True).prop(self, "radius", text="")
		if self.mesh_type == 'CYLINDER':
			row = col.row().split(factor=0.27, align=True)
			row.label(text="Depth:")
			row.row(align=True).prop(self, "depth", text="")
			row = col.row().split(factor=0.27, align=True)
			row.label(text="Segments:")
			row.row(align=True).prop(self, "segment", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Rotation:")
		row.row(align=True).prop(self, "rot_z", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Rotation Seed:")
		row.row(align=True).prop(self, "rot_seed", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Limit:")
		row.row(align=True).prop(self, "limit", expand=True)
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Material Index:")
		row.row(align=True).prop(self, "mat_index", text="")
		col.separator(factor=0.5)
		col.prop(self, "even_count")

	def invoke(self, context, event):
		self.list = ""
		self.meshes.clear()
		self.list_col = ""
		self.collections.clear()

		path =  os.path.join(os.path.dirname(
			os.path.abspath(__file__)), "./flanges/")
		files = os.listdir(path)

		if files:
			for f in files:
				newListItem = self.meshes.add()
				file_name = f.split(".")[0]
				newListItem.name = file_name

		for c in bpy.data.collections:
			newListItem = self.collections.add()
			newListItem.name = c.name

		prefs = context.preferences.addons[__name__].preferences
		if prefs.use_confirm:
			return context.window_manager.invoke_props_dialog(self)
		else:
			return context.window_manager.invoke_props_popup(self, event)

class MESH_OT_panel_screws(Operator):
	'''Generate panel screws on selected island faces'''
	bl_idname = 'panel_screws.rflow'
	bl_label = 'Panel Screws'
	bl_options = {'REGISTER', 'UNDO', 'PRESET'}

	mesh_type : EnumProperty(
		name = 'Type',
		description = "Type",
		items = (
			('CYLINDER', 'Cylinder',''),
			('CUSTOM', 'Custom',''),
			('COLLECTION', 'Collection','')),
		default = 'CYLINDER'
		)
	list : StringProperty(
		name        = "Import Mesh",
		description = "Use selected for cap design"
		)
	meshes : CollectionProperty(type=PropertyGroup)
	list_col : StringProperty(
		name        = "Collections",
		description = "Collection objects for scatter"
		)
	collections : CollectionProperty(type=PropertyGroup)
	coll_seed : IntProperty(
		name        = "Object Seed",
		description = "Randomize seed for collection objects",
		default     = 1,
		min         = 1,
		soft_max    = 10000,
		step        = 1
		)
	margin : FloatProperty(
		name        = "Margin",
		description = "Margin of screws from island boundary",
		default     = 0.01,
		step        = 0.1,
		precision   = 4
		)
	offset : FloatProperty(
		name        = "Offset",
		description = "Offset generated screws",
		default     = 0,
		step        = 0.1,
		precision   = 4
		)
	spacing : FloatProperty(
		name        = "Spacing",
		description = "Edge spacing to determine the number of screws to make",
		default     = 0.05,
		min			= 0,
		soft_min    = 0.01,
		soft_max    = 1.0,
		step        = 0.01,
		precision   = 3
		)
	step : IntProperty(
		name        = "Step",
		description = "Spacing interval between generated screws",
		default     = 5,
		min         = 1,
		soft_max    = 100,
		step        = 1
		)
	threshold : FloatProperty(
		name        = "Threshold",
		description = "Maximum vert angle threshold for screws to appear",
		default     = radians(30),
		min         = radians(1),
		max         = radians(180),
		step        = 10,
		precision   = 3,
		subtype     = "ANGLE"
		)
	radius : FloatProperty(
		name        = "Radius",
		description = "Radius of screws",
		default     = 0.025,
		min			= 0,
		step        = 0.01,
		precision   = 3
		)
	rot_z : FloatProperty(
		name        = "Rotation",
		description = "Rotation of flanges/couplings",
		default     = 0,
		min         = radians(-360),
		max         = radians(360),
		step        = 10,
		precision   = 3,
		subtype     = "ANGLE"
		)
	rot_seed : IntProperty(
		name        = "Rotation Seed",
		description = "Randomize seed for rotation of flanges/couplings",
		default     = 1,
		min         = 1,
		soft_max    = 10000,
		step        = 1
		)
	depth : FloatProperty(
		name        = "Depth",
		description = "Depth of flanges/couplings",
		default     = 0.025,
		min			= 0,
		step        = 0.01,
		precision   = 3
		)
	segment : IntProperty(
		name        = "Segments",
		description = "Total segment of flanges/couplings",
		default     = 6,
		min         = 3,
		max         = 500
		)
	bvl_offset : FloatProperty(
		name        = "Offset",
		description = "Bevel offset/width",
		default     = 0.0,
		min         = 0.0,
		soft_max    = 100.0,
		step        = 0.01,
		precision   = 4
	)
	bvl_seg : IntProperty(
		name        = "Segments",
		description = "Bevel segments",
		default     = 1,
		min         = 1,
		soft_max    = 100,
		step        = 1
	)
	birth_perc : FloatProperty(
		name        = "Birth*",
		description = "Percentage to determine if screw appears at this point",
		min         = 0,
		max         = 100,
		precision   = 0,
		default     = 100,
		subtype     = "PERCENTAGE"
		)
	birth_seed : IntProperty(
		name        = "Birth Seed",
		description = "Randomize seed for birth of screws",
		default     = 1,
		min         = 1,
		soft_max    = 10000,
		step        = 1
		)
	mat_index : IntProperty(
		name        = "Material Index",
		description = "Material assigned to duplicates",
		default     = -1,
		min         = -1,
		max         = 32767,
		step        = 1
		)
	select_bound : BoolProperty(
		name        = "Selection Boundary",
		description = "Generate screws at selection boundary else at face boundaries",
		default     = True
		)
	use_angle : BoolProperty(
		name        = "Use Angle Threshold",
		description = "Use angle threshold for boundary vertices for screws to appear",
		default     = True
		)
	tagged_only : BoolProperty(
		name        = "Tagged Verts Only",
		description = "Generate screws from tagged verts only",
		default     = False
		)
	use_dissolve : BoolProperty(
		name        = "Limited Dissolve",
		description = "Use limited dissolve to unify faces",
		default     = False
		)
	angle : FloatProperty(
		name        = "Max Angle",
		description = "Angle limit",
		default     = radians(5),
		min         = 0,
		max         = radians(180),
		step        = 10,
		precision   = 3,
		subtype     = "ANGLE"
		)

	@classmethod
	def poll(cls, context):
		return context.active_object is not None and context.active_object.mode == "OBJECT"

	def make_screws(self, obj):

		orig_mesh = obj.data

		bm = bmesh.new()
		temp_mesh = bpy.data.meshes.new(".temp")
		bm.from_mesh(orig_mesh)

		def add_screw(obj, bm, pos, normal, idx):

			context = bpy.context

			bm_temp = bmesh.new()
			temp_mesh = bpy.data.meshes.new(".temp")

			type = self.mesh_type

			if type == 'CYLINDER':
				listv = bmesh.ops.create_cone(
					bm_temp,
					cap_ends    = True,
					segments    = self.segment,
					radius1     = self.radius,
					radius2     = self.radius,
					depth       = self.depth
					)['verts']

				M = obj.matrix_world
				faces = sorted(bm_temp.faces, key=lambda f: (M @ f.calc_center_median())[2])

				if faces:
					basis = (M @ faces[-1].calc_center_median())[2]
					top_faces = [f for f in faces if abs((M @ f.calc_center_median())[2] - basis) < 0.0001]

					bmesh.ops.bevel(
						bm_temp,
						geom            = [e for e in top_faces[0].edges],
						offset          = self.bvl_offset,
						offset_type     = 'OFFSET',
						segments        = self.bvl_seg,
						profile         = 0.5,
						affect			= 'EDGES',
						clamp_overlap	= True
						)

			if type == 'CUSTOM':
				if self.list:
					file_name = self.list
					custom_data = bpy.data.meshes.get(file_name + "_rflow_mesh", None)
					if not custom_data:
						if file_name.find(".stl") == -1: file_name += ".stl"
						filepath = os.path.join(os.path.dirname(
							os.path.abspath(__file__)), "./screws/" + file_name)
						bpy.ops.import_mesh.stl(filepath=filepath, global_scale=1.0)

						import_obj = context.selected_objects[0]
						custom_data = import_obj.data
						custom_data.name += "_rflow_mesh"

						obj.select_set(True)
						context.view_layer.objects.active = obj

						bpy.data.objects.remove(import_obj)

					if custom_data:
						bm_temp.from_mesh(custom_data)

			if type == 'COLLECTION':
				collection = self.list_col
				objs = bpy.data.collections.get(collection)
				if objs:
					mesh_objs = [o for o in bpy.data.collections.get(collection).all_objects \
						if o.type == 'MESH']
					if mesh_objs:
						seed(self.coll_seed + idx)
						rand_obj = choice(mesh_objs)
						coll_obj = bpy.data.objects.get(rand_obj.name)

						if coll_obj:
							bm_temp.from_mesh(coll_obj.data)

			if type in ['CUSTOM', 'COLLECTION']:
				bmesh.ops.scale(
					bm_temp,
					vec     = Vector(tuple([self.radius] * 3)),
					space   = Matrix(),
					verts   = bm_temp.verts
					)

			z = self.rot_z
			if self.rot_seed > 1:
				seed(self.rot_seed + idx)
				z = uniform(360-z, z)

			rot_axis = [0 , 0, z]

			bmesh.ops.rotate(
				bm_temp,
				verts   = bm_temp.verts,
				cent    = Vector(),
				matrix  = Euler(Vector(rot_axis)).to_matrix()
				)

			bmesh.ops.translate(
					bm_temp,
					verts   = bm_temp.verts,
					vec     = pos
					)

			quat = normal.to_track_quat('Z', 'Y')
			mat = obj.matrix_world @ quat.to_matrix().to_4x4()
			rot = mat.to_3x3().normalized()

			_, orig_rot, _ = obj.matrix_world.decompose()
			bmesh.ops.rotate(
					bm_temp,
					verts   = bm_temp.verts,
					cent    = pos,
					matrix  = orig_rot.to_matrix().inverted() @ rot
					)

			bm_temp.to_mesh(temp_mesh)
			bm_temp.free()

			bm.from_mesh(temp_mesh)
			bpy.data.meshes.remove(temp_mesh)

		face_sel = [f for f in bm.faces if not f.select]
		bmesh.ops.delete(bm, geom=face_sel, context='FACES')

		if self.use_dissolve:
			bmesh.ops.dissolve_limit(bm, angle_limit=self.angle, \
				use_dissolve_boundaries=False, verts=bm.verts, edges=bm.edges, delimit={'NORMAL'})

		if self.select_bound:
			margin = bmesh.ops.inset_region(bm, faces=bm.faces, use_boundary=True, use_even_offset=True, \
				thickness=self.margin, depth=self.offset)['faces']
		else:
			margin = bmesh.ops.inset_individual(bm, faces=bm.faces, use_even_offset=True, \
				thickness=self.margin, depth=self.offset)['faces']

		bmesh.ops.delete(bm, geom=margin, context='FACES')

		vgroup = "tagged_verts_rflow"
		vg = obj.vertex_groups.get(vgroup)
		if vg:
			idx = vg.index
			deform_layer = bm.verts.layers.deform.active or bm.verts.layers.deform.new()

		if not self.use_angle \
			and bool(self.spacing):
			for edge in bm.edges:
				tagged = len([v for v in edge.verts if vg.index in v[deform_layer]]) == len(edge.verts) \
					if self.tagged_only else True
				if edge.is_boundary and tagged:
					length_e = edge.calc_length()
					segments = length_e / (self.step * self.spacing)
					cut_x_times = int(floor(segments - (segments / 2 )))
					bmesh.ops.subdivide_edges(bm, edges=[edge], cuts=cut_x_times)

		origin_faces = bm.faces[:]
		origin_verts = bm.verts[:]

		if self.tagged_only and vg:
			origin_verts = [v for v in bm.verts if idx in v[deform_layer]]

		for i, v in enumerate(origin_verts):
			if v.is_boundary:
				proc = False
				if self.use_angle:
					angle = v.calc_edge_angle(None)
					if angle and angle >= self.threshold: proc = True
				else: proc = True

				seed(self.birth_seed + i)
				if random() > self.birth_perc/100: proc = False

				if proc:
					add_screw(obj, bm, v.co, v.normal, i)

		bmesh.ops.delete(bm, geom=origin_faces, context='FACES')

		bm.to_mesh(temp_mesh)
		bm.free()

		new_obj = bpy.data.objects.new(obj.name + "_Screws", temp_mesh)
		orig_loc, orig_rot, orig_scale = obj.matrix_world.decompose()
		new_obj.scale = orig_scale
		new_obj.rotation_euler = orig_rot.to_euler()
		new_obj.location = orig_loc
		new_obj.data.use_auto_smooth = True

		bpy.context.scene.collection.objects.link(new_obj)

		copy_modifiers([obj, new_obj], mod_types=['MIRROR'])
		assign_mat(self, obj, new_obj, self.mat_index)


	def execute(self, context):

		self.make_screws(context.active_object)

		return {"FINISHED"}

	def draw(self, context):
		layout = self.layout
		col = layout.column()
		col.separator(factor=0.1)
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Type:")
		row.row(align=True).prop(self, "mesh_type", expand=True)
		if self.mesh_type == 'CUSTOM':
			row = col.row().split(factor=0.27, align=True)
			row.label(text="Import Mesh:")
			row.prop_search(
				self,
				"list",
				self,
				"meshes",
				text="",
				icon = "MESH_DATA"
				)
		if self.mesh_type == 'COLLECTION':
			row = col.row().split(factor=0.27, align=True)
			row.label(text="Collection:")
			row.prop_search(
				self,
				"list_col",
				self,
				"collections",
				text="",
				icon = "OUTLINER_COLLECTION"
				)
			row = col.row().split(factor=0.27, align=True)
			row.label(text="Object Seed:")
			row.row(align=True).prop(self, "coll_seed", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Position:")
		split = row.split(factor=0.5, align=True)
		split.row(align=True).prop(self, "margin")
		split.row(align=True).prop(self, "offset")
		if not self.use_angle:
			row = col.row().split(factor=0.27, align=True)
			row.label(text="Quantity:")
			split = row.split(factor=0.5, align=True)
			split.row(align=True).prop(self, "spacing")
			split.row(align=True).prop(self, "step")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Radius:")
		row.row(align=True).prop(self, "radius", text="")
		if self.mesh_type == 'CYLINDER':
			row = col.row().split(factor=0.27, align=True)
			row.label(text="Cylinder Prop:")
			split = row.split(factor=0.5, align=True)
			split.row(align=True).prop(self, "depth")
			split.row(align=True).prop(self, "segment")
			row = col.row().split(factor=0.27, align=True)
			row.label(text="Bevel:")
			split = row.split(factor=0.5, align=True)
			split.row(align=True).prop(self, "bvl_offset")
			split.row(align=True).prop(self, "bvl_seg")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Rotation:")
		row.row(align=True).prop(self, "rot_z", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Rotation Seed:")
		row.row(align=True).prop(self, "rot_seed", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Birth:")
		row.row(align=True).prop(self, "birth_perc", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Birth Seed:")
		row.row(align=True).prop(self, "birth_seed", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Material Index:")
		row.row(align=True).prop(self, "mat_index", text="")
		col.separator(factor=0.5)
		flow = col.column_flow(columns=2, align=True)
		flow.prop(self, "use_angle")
		flow.prop(self, "select_bound")
		if self.use_angle:
			col.prop(self, "threshold")
		col.prop(self, "tagged_only")
		col.prop(self, "use_dissolve")
		if self.use_dissolve:
			col.prop(self, "angle")

	def invoke(self, context, event):
		obj = context.active_object
		self.list = ""
		self.meshes.clear()
		self.list_col = ""
		self.collections.clear()
		self.select_bound = True

		obj.update_from_editmode()
		has_face = [f for f in obj.data.polygons if f.select]

		if has_face:
			path =  os.path.join(os.path.dirname(
				os.path.abspath(__file__)), "./screws/")
			files = os.listdir(path)

			if files:
				for f in files:
					newListItem = self.meshes.add()
					file_name = f.split(".")[0]
					newListItem.name = file_name

			for c in bpy.data.collections:
				newListItem = self.collections.add()
				newListItem.name = c.name

			prefs = context.preferences.addons[__name__].preferences
			if prefs.use_confirm:
				return context.window_manager.invoke_props_dialog(self)
			else:
				return context.window_manager.invoke_props_popup(self, event)
		else:
			self.report({'WARNING'}, "No faces selected.")
			return {"FINISHED"}


class MESH_OT_tag_verts(Operator):
	'''Assign vertices to vertex group for use in Quad Slice'''
	bl_idname = 'tag_verts.rflow'
	bl_label = 'Tag Verts'
	bl_options = {'REGISTER', 'UNDO'}

	@classmethod
	def poll(cls, context):
		return context.active_object is not None and context.active_object.mode == "EDIT"

	def execute(self, context):
		obj = context.active_object
		vgname = "tagged_verts_rflow"

		mesh = obj.data
		bm =  bmesh.from_edit_mesh(mesh)

		assign_vgroup(obj, bm, bm.verts, vgname)

		bmesh.update_edit_mesh(mesh)

		return {"FINISHED"}

class MESH_OT_quad_slice(Operator):
	'''Draw lines from vertices or edges using view or tangent as direction'''
	bl_idname = 'quad_slice.rflow'
	bl_label = 'Quad Slice'
	bl_options = {'REGISTER', 'UNDO'}

	direction : EnumProperty(
		name = "Direction",
		items = (
			('TANGENT', 'Tangent','Use face tangents from selected as direction'),
			('VIEW', 'View','Use view angle as direction')),
		default = 'TANGENT')
	tangent_idx : IntProperty(
		name        = "Tangent",
		description = "Tangent index",
		default     = 1,
		min         = 1,
		max         = 10000,
		step        = 1
		)
	origin : EnumProperty(
		name = "Origin",
		items = (
			('VERT', 'Verts','Source cut lines from selected verts'),
			('EDGE', 'Edges','Source cut lines from selected edges'),
			('TAGGED', 'Tagged','Source cut lines from tagged verts')),
		default = 'VERT')
	use_geo_v : EnumProperty(
		name = "Geometry",
		items = (
			('SHARED', 'Shared Face','Limit cut to shared face'),
			('ALL', 'All Faces','Cut all faces'),
			('LINKED', 'Linked Faces','Cut linked faces')),
		default = 'SHARED')
	face_idx : IntProperty(
		name        = "Face",
		description = "Shared face to cut",
		default     = 1,
		min         = 1,
		max         = 10000,
		step        = 1
		)
	use_geo_f : EnumProperty(
		name = "Geometry",
		items = (
			('SELECT', 'Selected','Limit cut to selected faces'),
			('ALL', 'All Faces','Cut all faces'),
			('LINKED', 'Linked Faces','Cut linked faces')),
		default = 'SELECT')
	limit : EnumProperty(
		name = "Limit",
		items = (
			('NONE', 'None','Limit cut to none'),
			('LINE1', 'X','Limit to cut direction X'),
			('LINE2', 'Y','Limit to cut direction Y')),
		default = 'NONE')
	bisect_dist : FloatProperty(
		name        = "Distance",
		description = "Minimum distance when testing if a vert is exactly on the plane",
		default     = 0.0001,
		min			= 0.0,
		soft_min    = 0.0001,
		soft_max    = 1.0,
		step        = 0.01,
		precision   = 4
		)
	slide_factor : FloatProperty(
		name        = "Factor",
		description = "Split location on selected edge",
		default     = 0.5,
		min         = 0.0,
		max         = 1.0,
		step        = 0.1,
		precision   = 4
	)
	cut_rot : FloatProperty(
		name        = "Rotation",
		description = "Rotate to X axis",
		default     = 0,
		min         = radians(-360),
		max         = radians(360),
		step        = 10,
		precision   = 3,
		subtype     = "ANGLE"
		)
	rem_doubles : BoolProperty(
		name        = "Remove Doubles",
		description = "Remove overlapping verts",
		default     = False
		)
	doubles_dist : FloatProperty(
		name        = "Merge Distance",
		description = "Maximum distance between elements to merge",
		default     = 0.0001,
		min         = 0.0,
		soft_max    = 10.0,
		step        = 0.1,
		precision   = 4
		)
	remove_singles : BoolProperty(
		name        = "Remove 2EV's",
		description = "Remove verts with only two connecting edges",
		default     = False
		)

	@classmethod
	def poll(cls, context):
		return context.active_object is not None and context.active_object.mode == "EDIT"

	def execute(self, context):
		obj = context.active_object

		rv3d = context.region_data
		vrot = rv3d.view_rotation

		mesh = obj.data
		bm = bmesh.from_edit_mesh(mesh)

		orig_geo = []
		split_points = []
		origin = []

		if self.origin == 'EDGE':
			edg = [e for e in bm.edges if e.select]
			ret = bmesh.ops.bisect_edges(bm, edges=edg, cuts=1, edge_percents={e:self.slide_factor for e in edg})
			split_points = [v for v in ret['geom_split'] if isinstance(v, bmesh.types.BMVert)]

		vf = { v: v.link_faces for v in bm.verts if v.select }
		if vf.keys():
			cut_points = list(vf.keys())
			list_faces = undupe(list(chain.from_iterable(vf.values())))
			if self.has_face:
				list_faces = [f for f in list_faces if f.select]

			if self.use_geo_v == 'LINKED' \
				or self.use_geo_f == 'LINKED':
				if cut_points:
					island = get_linked_faces(list_faces)
					if island:
						fv = sum([f.verts[:] for f in island], [])
						fe = sum([f.edges[:] for f in island], [])
						orig_geo = fv + fe + list(island) + split_points
			elif self.use_geo_v == 'ALL' \
				or self.use_geo_f == 'ALL':
				orig_geo = bm.verts[:] + bm.edges[:] + bm.faces[:] + split_points
			else:
				if self.has_face:
					fv = sum([f.verts[:] for f in list_faces], [])
					fe = sum([f.edges[:] for f in list_faces], [])
					orig_geo = fv + fe + list_faces + split_points
				else:
					idxf = [[f.index for f in v.link_faces] for v in cut_points]
					shared_f = list(set(idxf[0]).intersection(*idxf))
					if shared_f:
						bm.faces.ensure_lookup_table()
						n = len(shared_f)
						face = bm.faces[shared_f[(n + (self.face_idx - 1)) % n]]
						orig_geo = face.verts[:] + face.edges[:] + [face] + split_points

			if self.origin == 'TAGGED':
				vgroup = "tagged_verts_rflow"
				vg = obj.vertex_groups.get(vgroup)
				if vg:
					idx = vg.index
					deform_layer = bm.verts.layers.deform.active or bm.verts.layers.deform.new()
					origin = [v for v in bm.verts if idx in v[deform_layer]]
				else:
					self.report({'WARNING'}, "Vertex group not found!")
			else:
				origin = cut_points if self.origin == 'VERT' else split_points

			if origin:
				xt = None; yt = None
				if self.direction == 'TANGENT':
					if list_faces:
						tangents = [[tuple(f.calc_tangent_edge()), tuple(f.normal)] for f in list_faces]

						n = len(tangents)
						t = tangents[(n + (self.tangent_idx - 1)) % n]
						tangent = Vector(t[0])
						normal = Vector(t[1])
						xt = tangent
						yt = xt.cross(normal).normalized()

				x = vrot @ Vector((0,-1,0))
				y = vrot @ Vector((-1,0,0))

				if xt and yt:
					x = xt; y = yt

				cutdir = [x, y]
				cutdir = cutdir[:-1] if self.limit == 'LINE1' else cutdir[-1:] \
					if self.limit == 'LINE2' else cutdir

				new_geo = []
				P = x.cross(y).normalized()
				M = Matrix.Rotation(self.cut_rot, 3, P)
				for v in origin:
					co = v.co
					for t in cutdir:
						t = t @ M.inverted()
						geo = undupe(orig_geo + new_geo)
						ret = bmesh.ops.bisect_plane(bm, geom=geo, plane_co=co, plane_no=t, dist=self.bisect_dist)

						new_geo.extend(ret['geom'] + ret['geom_cut'])
						new_geo = undupe(new_geo)

				bmesh.ops.dissolve_degenerate(bm, dist=1e-4, edges=bm.edges[:])

				if self.rem_doubles \
					or self.remove_singles:
					vcount = len(bm.verts)
					if self.rem_doubles:
						new_v = [v for v in new_geo if isinstance(v, bmesh.types.BMVert) and v in bm.verts]
						bmesh.ops.remove_doubles(bm, verts=new_v, dist=self.doubles_dist)

					if self.remove_singles:
						singles = get_singles(bm.verts)
						bmesh.ops.dissolve_verts(bm, verts=singles)

					new_vcount = vcount - len(bm.verts)
					self.report({'INFO'}, "Removed " + str(new_vcount) + " vertices.")

		bmesh.update_edit_mesh(mesh)
		bpy.ops.mesh.select_all(action='DESELECT')

		return {'FINISHED'}

	def draw(self, context):
		layout = self.layout
		col = layout.column()
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Direction:")
		row.row(align=True).prop(self, "direction", expand=True)
		if self.direction == 'TANGENT':
			row = col.row().split(factor=0.27, align=True)
			row.label(text="Tangent:")
			row.row(align=True).prop(self, "tangent_idx", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Origin:")
		row.row(align=True).prop(self, "origin", expand=True)
		if self.has_face:
			row = col.row().split(factor=0.27, align=True)
			row.label(text="Geometry:")
			row.row(align=True).prop(self, "use_geo_f", expand=True)
		else:
			row = col.row().split(factor=0.27, align=True)
			row.label(text="Geometry:")
			row.row(align=True).prop(self, "use_geo_v", expand=True)
			if self.use_geo_v == 'SHARED':
				row = col.row().split(factor=0.27, align=True)
				row.label(text="Face:")
				row.row(align=True).prop(self, "face_idx", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Limit:")
		row.row(align=True).prop(self, "limit", expand=True)
		if self.origin == 'EDGE':
			row = col.row().split(factor=0.27, align=True)
			row.label(text="Factor:")
			row.row(align=True).prop(self, "slide_factor", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Rotation:")
		row.row(align=True).prop(self, "cut_rot", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Distance:")
		row.row(align=True).prop(self, "bisect_dist", text="")
		col.separator(factor=0.5)
		col.prop(self, "rem_doubles")
		if self.rem_doubles:
			col.prop(self, "doubles_dist")
		col.prop(self, "remove_singles")

	def invoke(self, context, event):
		self.face_idx = 1
		self.tangent_idx = 1
		self.limit = 'NONE'
		self.cut_rot = 0
		self.slide_factor = 0.5

		obj = context.active_object
		obj.update_from_editmode()
		self.has_face = [f for f in obj.data.polygons if f.select]

		return context.window_manager.invoke_props_popup(self, event)

class MESH_OT_grid_project(Operator):
	'''Project grid cuts on selected faces'''
	bl_idname = 'grid_project.rflow'
	bl_label = 'Grid Project'
	bl_options = {'REGISTER', 'UNDO'}

	grid_center : EnumProperty(
		name = "Center",
		items = (
			('AVERAGE', 'Average','Use average location of selected faces as grid center'),
			('PERFACE', 'Per Face','Use individual face location as grid center')),
		default = 'AVERAGE')
	direction : EnumProperty(
		name = "Direction",
		items = (
			('TANGENT', 'Tangent','Align direction to longest edge'),
			('EDGE', 'Edge','Align direction to specific edge'),
			('VIEW', 'View','Align direction to view')),
		default = 'TANGENT')
	tangent_src : EnumProperty(
		name = "Source",
		items = (
			('INDIV', 'Individual','Get tangent from individual faces'),
			('SELECT', 'Select','Get tangent from specific face')),
		default = 'INDIV')
	tangent_idx : IntProperty(
		name        = "Tangent",
		description = "Tangent index",
		default     = 1,
		min         = 1,
		max         = 10000,
		step        = 1
		)
	cut_geo : EnumProperty(
		name = "Geometry",
		items = (
			('SELECT', 'Selected','Limit cut to selected faces'),
			('ALL', 'All Faces','Cut all faces'),
			('LINKED', 'Linked Faces','Cut linked faces')),
		default = 'SELECT')
	cuts_amt : EnumProperty(
		name = "Cuts Amount",
		items = (
			('SAME', 'Uniform','Use same cut number for all axes'),
			('INDIV', 'Individual','Use individual cut number for x and y axis')),
		default = 'SAME')
	cuts : IntProperty(
		name        = "X Cuts",
		description = "Number of cuts for all axes",
		default     = 10,
		min         = 0,
		soft_max    = 20,
		step        = 1
	)
	cut_xy : IntVectorProperty(
		name        = "Cuts",
		description = "Number of cuts for x and y axis",
		default     = (10,10),
		size        = 2,
		min         = 0,
		soft_max	= 20,
		step        = 1,
		subtype		= "XYZ"
		)
	offset_xy : FloatVectorProperty(
		name        = "Offset",
		description = "Offset x and y axis cuts",
		default     = (0.0, 0.0),
		size        = 2,
		soft_min    = -10.0,
		soft_max    = 10.0,
		step        = 0.01,
		precision   = 4,
		subtype		= "XYZ"
		)
	size : FloatProperty(
		name        = "Size",
		description = "Grid size",
		default     = 1,
		min         = 0.0001,
		max         = 100,
		step        = 0.1,
		precision   = 4
	)
	rot : FloatProperty(
		name        = "Rotation",
		description = "Rotate to X axis",
		default     = 0,
		min         = radians(-360),
		max         = radians(360),
		step        = 10,
		precision   = 3,
		subtype     = "ANGLE"
		)
	rem_doubles : BoolProperty(
		name        = "Remove Doubles",
		description = "Remove overlapping verts",
		default     = False
		)
	doubles_dist : FloatProperty(
		name        = "Merge Distance",
		description = "Maximum distance between elements to merge",
		default     = 0.0001,
		min         = 0.0,
		soft_max    = 10.0,
		step        = 0.1,
		precision   = 4
		)
	tri_ngons : BoolProperty(
		name        = "Triangulate Ngons",
		description = "Triangulate ngons in cut faces",
		default     = False
		)
	remove_singles : BoolProperty(
		name        = "Remove 2EV's",
		description = "Remove verts with only two connecting edges",
		default     = False
		)

	def get_edge_index(self, context):

		items = []
		elist = self.face_edge_index.split()
		for i, n in enumerate(elist):
			items.append((n, str(i), ""))

		return tuple(items)

	idx : EnumProperty(
		name = "Face Edges",
		items = get_edge_index,
		default = 0,
		)
	face_edge_index : StringProperty()

	@classmethod
	def poll(cls, context):
		return context.active_object is not None and context.active_object.mode == 'EDIT'

	def cut_grid(self, bm, orig_geo, center, area, normal, tangent):

		t1 = tangent
		t2 = normal.cross(t1)
		vec = t1 - t2 * 0.00001

		P = normal
		M = Matrix.Rotation(self.rot, 3, P)
		tangent = vec @ M.inverted()

		new_geo = []

		x = self.cuts if self.cuts_amt == 'SAME' else self.cut_xy[0]
		xdir = tangent
		center += self.offset_xy[0] * xdir
		if sum(xdir) != 0:
			for i in range(-x+1, x, 1):
				geo = undupe(orig_geo + new_geo)
				vec = ((i * (sqrt(area)/x)) * xdir) * self.size
				ret = bmesh.ops.bisect_plane(bm, geom=geo, plane_co=center+vec, plane_no=xdir)

				new_geo.extend(ret['geom'] + ret['geom_cut'])
				new_geo = undupe(new_geo)

		y = self.cuts if self.cuts_amt == 'SAME' else self.cut_xy[1]
		ydir = tangent.cross(normal).normalized()
		center += self.offset_xy[1] * ydir
		if sum(ydir) != 0:
			for i in range(-y+1, y, 1):
				geo = undupe(orig_geo + new_geo)
				vec = ((i * (sqrt(area)/y)) * ydir) * self.size
				ret = bmesh.ops.bisect_plane(bm, geom=geo, plane_co=center+vec, plane_no=ydir)

				new_geo.extend(ret['geom'] + ret['geom_cut'])
				new_geo = undupe(new_geo)

		return new_geo

	def execute(self, context):
		obj = context.active_object

		rv3d = context.region_data
		vrot = rv3d.view_rotation

		if self.has_face:
			mesh = obj.data
			bm = bmesh.from_edit_mesh(mesh)

			flist = [f for f in bm.faces if f.select]

			def get_tangent(bm):

				bm.edges.ensure_lookup_table()
				edge = bm.edges[int(self.idx)]
				t = edge.link_loops[0].calc_tangent()

				return t

			orig_geo = []

			if self.tangent_src == 'SELECT' or \
				self.grid_center == 'AVERAGE':
				listdir = [[f.calc_tangent_edge(), f.normal] for f in flist]
				n = len(listdir)
				fdir = listdir[(n + (self.tangent_idx - 1)) % n]
				ft = fdir[0]
				fn = fdir[1]

			if self.grid_center == 'PERFACE':
				for f in flist:
					center = f.calc_center_bounds()
					area = f.calc_area()
					normal = f.normal if self.tangent_src == 'INDIV' else fn
					tsource = f.calc_tangent_edge() if self.tangent_src == 'INDIV' else ft
					tangent = tsource if self.direction == 'TANGENT' else get_tangent(bm) \
						if self.direction == 'EDGE' else vrot @ Vector((0,-1,0))

					orig_geo = f.verts[:] + f.edges[:] + [f]
					new_geo = self.cut_grid(bm, orig_geo, center, area, normal, tangent)
			else:
				center = sum([f.calc_center_bounds() for f in flist], Vector()) / len(flist)
				area = sum(f.calc_area() for f in flist)
				normal = fn
				tangent = ft if self.direction == 'TANGENT' else get_tangent(bm) \
					if self.direction == 'EDGE' else vrot @ Vector((0,-1,0))

				if self.cut_geo == 'SELECT':
					fv = sum([f.verts[:] for f in flist], [])
					fe = sum([f.edges[:] for f in flist], [])
					orig_geo = fv + fe + flist
				elif self.cut_geo == 'ALL':
					fv = sum([f.verts[:] for f in bm.faces], [])
					fe = sum([f.edges[:] for f in bm.faces], [])
					orig_geo = fv + fe + bm.faces[:]
				else:
					island = get_linked_faces(flist)
					if island:
						fv = sum([f.verts[:] for f in island], [])
						fe = sum([f.edges[:] for f in island], [])
						orig_geo = fv + fe + list(island)

				new_geo = self.cut_grid(bm, orig_geo, center, area, normal, tangent)

			bmesh.ops.dissolve_degenerate(bm, dist=1e-4, edges=bm.edges[:])

			if self.tri_ngons:
				newf = [f for f in new_geo if isinstance(f, bmesh.types.BMFace) and f in bm.faces]
				ret = bmesh.ops.triangulate(bm, faces=newf, quad_method='BEAUTY', ngon_method='BEAUTY')
				bmesh.ops.join_triangles(bm, faces=ret['faces'],angle_face_threshold=180, angle_shape_threshold=180)

			if self.rem_doubles \
				or self.remove_singles:
				vcount = len(bm.verts)
				if self.rem_doubles:
					new_v = [v for v in new_geo if isinstance(v, bmesh.types.BMVert) and v in bm.verts]
					bmesh.ops.remove_doubles(bm, verts=new_v, dist=self.doubles_dist)
				if self.remove_singles:
					singles = get_singles(bm.verts)
					bmesh.ops.dissolve_verts(bm, verts=singles)

				new_vcount = vcount - len(bm.verts)
				self.report({'INFO'}, "Removed " + str(new_vcount) + " vertices.")

			bmesh.update_edit_mesh(mesh)
			bpy.ops.mesh.select_all(action='DESELECT')

		return {'FINISHED'}

	def draw(self, context):
		layout = self.layout
		col = layout.column()
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Center:")
		row.row(align=True).prop(self, "grid_center", expand=True)
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Direction:")
		row.row(align=True).prop(self, "direction", expand=True)
		if self.direction == 'TANGENT' \
			and self.grid_center == 'PERFACE':
			row = col.row().split(factor=0.27, align=True)
			row.label(text="Tangent Basis:")
			row.row(align=True).prop(self, "tangent_src", expand=True)
			if self.tangent_src == 'SELECT':
				row = col.row().split(factor=0.27, align=True)
				row.label(text="Tangent:")
				row.row(align=True).prop(self, "tangent_idx", text="")
		if self.direction == 'TANGENT' \
			and self.grid_center == 'AVERAGE':
			row = col.row().split(factor=0.27, align=True)
			row.label(text="Tangent:")
			row.row(align=True).prop(self, "tangent_idx", text="")
		if self.direction == 'EDGE':
			row = col.row().split(factor=0.27, align=True)
			row.label(text="Edge:")
			row.row(align=True).prop(self, "idx", text="")
		if self.grid_center == 'AVERAGE':
			row = col.row().split(factor=0.27, align=True)
			row.label(text="Geometry:")
			row.row(align=True).prop(self, "cut_geo", expand=True)
		if self.cuts_amt == 'SAME':
			row = col.row().split(factor=0.27, align=True)
			row.label(text="Cuts:")
			row.row(align=True).prop(self, "cuts", text="")
		else:
			row = col.row().split(factor=0.27, align=True)
			row.label(text="Cuts:")
			row.row(align=True).prop(self, "cut_xy", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Cuts Amount:")
		row.row(align=True).prop(self, "cuts_amt", expand=True)
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Offset:")
		row.row(align=True).prop(self, "offset_xy", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Size:")
		row.row(align=True).prop(self, "size", text="")
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Rotation:")
		row.row(align=True).prop(self, "rot", text="")
		col.separator(factor=0.5)
		col.prop(self, "rem_doubles")
		if self.rem_doubles:
			col.prop(self, "doubles_dist")
		col.prop(self, "tri_ngons")
		col.prop(self, "remove_singles")

	def invoke(self, context, event):
		obj = context.active_object
		self.tangent_idx = 1
		self.offset_xy = [0.0, 0.0]
		self.rot = 0

		mesh = obj.data
		obj.update_from_editmode()
		self.has_face = faces = [f for f in mesh.polygons if f.select]
		if faces:
			fe = sum((list(f.edge_keys) for f in faces), [])
			fe = undupe(list(chain.from_iterable(fe)))
			self.face_edge_index = ""
			if fe:
				for e in fe:
					self.face_edge_index += str(e) + " "

				flist = self.face_edge_index.split()
				self.idx = flist[0]

			return context.window_manager.invoke_props_popup(self, event)
		else:
			self.report({'WARNING'}, "No faces selected!")
			return {'FINISHED'}

class MESH_OT_auto_smooth(Operator):
	'''Smooth shade active object and use auto-smooth. Ctrl+click for popup menu'''
	bl_idname = 'auto_smooth.rflow'
	bl_label = 'Auto Smooth'
	bl_options = {'REGISTER', 'UNDO'}

	deg : FloatProperty(
		name        = "Angle",
		description = "Auto smooth angle",
		default     = radians(30),
		min         = 0,
		max         = radians(180),
		step        = 10,
		precision   = 3,
		subtype     = "ANGLE"
		)
	set_auto_smooth : BoolProperty(
		name        = "Use Auto Smooth",
		description = "Toggle auto smooth on or off",
		default     = True
		)
	clear_cn : BoolProperty(
		name        = "Clear Custom Split Normals Data",
		description = "Remove the custom split normals layer, if it exists",
		default     = False
		)

	@classmethod
	def poll(cls, context):
		return context.active_object is not None and context.active_object.type == 'MESH'

	def execute(self, context):
		objs = context.selected_objects

		for o in objs:
			mesh = o.data

			auto_smooth(o, self.deg, self.set_auto_smooth)
			o.update_from_editmode()

			if self.clear_cn and mesh.has_custom_normals:
				bpy.ops.mesh.customdata_custom_splitnormals_clear()

			mesh.update()

		return {"FINISHED"}

	def draw(self, context):
		layout = self.layout
		col = layout.column()
		col.prop(self, "deg")
		col.separator()
		col.prop(self, "set_auto_smooth")
		col.prop(self, "clear_cn")

	def invoke(self, context, event):
		self.set_auto_smooth = True
		self.clear_cn = False

		if event.ctrl:
			return context.window_manager.invoke_props_dialog(self)
		else:
			return self.execute(context)

class MESH_OT_auto_mirror(Operator):
	'''Add mirror modifier on selected objects'''
	bl_idname = 'auto_mirror.rflow'
	bl_label = 'Auto Mirror'
	bl_options = {'REGISTER', 'UNDO'}

	@classmethod
	def poll(cls, context):
		return context.active_object is not None and context.active_object.type == "MESH"

	def create_header(self):

		def enumerate_axis(axes):

			axis_text = ""
			for i in "XYZ":
				if axes[str(i)]: axis_text += i

			return axis_text

		axis_text = enumerate_axis(self.axes)

		if self.set_axis:
			header = ", ".join(filter(None,
				[
				"Mirror Axis: " + (axis_text.lstrip() if axis_text else "None"),
				"R: Reset axis toggles",
				"Enter/Space: Confirm",
				"Right Click/Esc: Cancel",
				]))
		else:
			header = ", ".join(filter(None,
				[
				"Left Click area to mirror",
				"Right Click/Esc: Cancel",
				]))

		return header.format()

	def pick_axis(self, context, co):

		hit, normal, face_index, _ = scene_ray_hit(context, co, ray_obj=self.mirror_obj, hit_bounds=True)

		return hit, normal, face_index

	def update_mirror(self, obj, origin):

		mw = obj.matrix_world
		origin = mw.inverted() @ origin

		mesh = obj.data
		bm = bmesh.new()
		bm.from_mesh(mesh)

		pivot = Vector()
		axis = [Vector((1,0,0)), Vector((0,1,0)), Vector((0,0,1))]

		x_dir = axis[0] if origin.x > 0 else -axis[0]
		y_dir = axis[1] if origin.y > 0 else -axis[1]
		z_dir = axis[2] if origin.z > 0 else -axis[2]

		axis_dir = [x_dir if self.axes['X'] else None, \
			y_dir if self.axes['Y'] else None, \
			z_dir if self.axes['Z'] else None]

		for n in axis_dir:
			if n:
				split = bm.verts[:] + bm.edges[:] + bm.faces[:]
				bmesh.ops.bisect_plane(
					bm,
					geom        = split,
					dist        = 0.0001,
					plane_co    = pivot,
					plane_no    = n,
					clear_inner = True,
					clear_outer = False
					)

		bm.to_mesh(mesh)
		mesh.update()

		self.mirror_add(obj, self.axes['X'], self.axes['Y'], self.axes['Z'])

	def create_axis(self, obj, src_obj):

		center = src_obj.matrix_world.translation

		mesh = obj.data
		bm = bmesh.new()

		pivot = src_obj.matrix_world.inverted() @ center
		v0 = bm.verts.new(pivot)

		for co in draw_axis(pivot):
			v = bm.verts.new(co)
			bm.edges.new((v0, v))

		bm.to_mesh(mesh)
		bm.free()

		copy_loc_rot(obj, src_obj)

	def mirror_apply(self):

		mirror = self.mirror_mod
		if mirror:
			name = mirror.name
			bpy.ops.object.modifier_move_to_index(modifier=name, index=0)
			bpy.ops.object.modifier_apply(modifier=name)

	def mirror_add(self, obj, x=False, y=False, z=False):

		mod = obj.modifiers
		md = mod.new("Mirror", "MIRROR")
		md.use_axis[0] = x
		md.use_axis[1] = y
		md.use_axis[2] = z
		md.use_clip = True
		md.use_mirror_merge = True
		md.show_expanded = False
		md.show_in_editmode = True
		md.show_on_cage = False

		bpy.ops.object.modifier_move_to_index(modifier=md.name, index=0)

	def clear_mesh_list(self):

		mirror_set = [
			self.center_axis,
			self.color_axis,
			]

		for o in mirror_set: remove_obj(o)

		for o in bpy.data.meshes:
			if not o in self.mesh_list \
				and o.users == 0: bpy.data.meshes.remove(o)

	def confirm_op(self, context):

		self.clear_mesh_list()
		context.window.cursor_modal_restore()
		bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')

	def modal(self, context, event):
		context.area.tag_redraw()
		self.mouse_co = event.mouse_region_x, event.mouse_region_y

		if event.type in {
			'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE',
			'NUMPAD_1', 'NUMPAD_2', 'NUMPAD_3', 'NUMPAD_4', 'NUMPAD_6',
			'NUMPAD_7', 'NUMPAD_8', 'NUMPAD_9', 'NUMPAD_5'}:
			return {'PASS_THROUGH'}

		if not self.set_axis:
			if event.type == 'LEFTMOUSE':
					if event.value == 'PRESS':
						hit, _, _ = self.pick_axis(context, self.mouse_co)
						if hit:
							self.mirror_axis = hit
							self.create_axis(self.center_axis, self.mirror_obj)
							self.set_axis = True
							context.window.cursor_modal_restore()
						else:
							self.report({'WARNING'}, ("Mirror axis not found!"))
							self.confirm_op(context)
							return {'FINISHED'}
		else:
			if event.type == 'R':
				if event.value == 'RELEASE':
					for axis in self.axes: self.axes[axis] = False

			axis_keys = ['X', 'Y', 'Z']
			if event.type in axis_keys:
				for i, x in enumerate(axis_keys):
					if event.type == x:
						if event.value == 'RELEASE':
							self.axes[axis_keys[i]] = False if self.axes[axis_keys[i]] else True

			if event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'}:
				if event.value == 'PRESS':
					self.mirror_apply()
					if next((axis for i, axis in enumerate(self.axes) \
						if self.axes[axis]), None):
						self.update_mirror(self.mirror_obj, self.mirror_axis)

					self.confirm_op(context)

					return {'FINISHED'}

		if event.type in {'RIGHTMOUSE', 'ESC'}:
			self.confirm_op(context)

			return {'CANCELLED'}

		# context.area.header_text_set(self.create_header())

		return {'RUNNING_MODAL'}

	def invoke(self, context, event):
		args = (self, context)

		self.mesh_list = [o for o in bpy.data.meshes]

		obj = self.mirror_obj = context.active_object
		self.mouse_co = []
		self.mirror_axis = Vector()
		self.set_axis = False
		self.axes = {axis:False for axis in ['X', 'Y', 'Z']}

		self.center_axis = create_temp_obj(context, "Center Axis")
		self.color_axis = create_temp_obj(context, "Color Axis")

		mirror = self.mirror_mod = next((m for m in obj.modifiers if m.type == 'MIRROR'), None)
		axis_toggles = []
		if mirror:
			axis_toggles = [i for i in mirror.use_axis]
			for n, axis in enumerate(self.axes):
				self.axes[axis] = axis_toggles[n]

		context.window.cursor_modal_set("EYEDROPPER")
		self._handle = bpy.types.SpaceView3D.draw_handler_add(draw_callback_px_quick_symm, args, 'WINDOW', 'POST_PIXEL')
		context.window_manager.modal_handler_add(self)

		return {'RUNNING_MODAL'}

class MESH_OT_extract_proxy(Operator):
	'''Extract faces from active object'''
	bl_idname = 'extr_proxy.rflow'
	bl_label = 'Extract Faces'
	bl_options = {'REGISTER', 'UNDO'}

	@classmethod
	def poll(cls, context):
		return context.active_object is not None

	def initialize(self, context, set = True):

		if set:
			self.create_applied_mesh(context, self.orig_obj, self.orig_dup)
			context.window.cursor_modal_set("CROSSHAIR")
		else:
			self.clear_extract()
			context.area.header_text_set(None)
			context.window.cursor_modal_restore()

	def delta_increment(self, event, x, y, dim):

		delta = abs(x - y)
		incr =  dim * (0.01 * (0.1 if event.shift else 1))
		v = delta * incr

		return v

	def move_origin(self, copy_obj, obj):

		new_origin = copy_obj.matrix_world.translation
		pivot = obj.matrix_world.inverted() @ new_origin
		obj.data.transform(Matrix.Translation(pivot))

	def update_extract(self, context, obj):

		obj.data = self.orig_dup.data.copy()
		self.move_origin(self.orig_dup, obj)

		sce = context.scene
		mat = obj.matrix_world

		mesh = obj.data
		bm = bmesh.new()
		bm.from_mesh(mesh)

		bm.faces.ensure_lookup_table()

		remove_faces = []
		inset_faces = []

		for i in range(len(bm.faces)):
			if not bm.faces[i].index in self.extract_faces:
				remove_faces.append(bm.faces[i])
			else:
				inset_faces.append(bm.faces[i])

		if abs(self.inset_val) > 0:
			if self.inset_indv:
				ret = bmesh.ops.inset_individual(bm, faces=inset_faces, use_even_offset=True, thickness=self.inset_val)
			else:
				ret = bmesh.ops.inset_region(bm, faces=inset_faces, use_boundary=True, use_even_offset=True, thickness=self.inset_val)

			remove_faces.extend(ret['faces'])

		bmesh.ops.delete(bm, geom=remove_faces, context='FACES')

		bm.to_mesh(mesh)
		mesh.update()

	def get_origin(self, context, co):

		hit, normal, face_index, _ = scene_ray_hit(context, co, ray_obj=self.orig_dup)

		return hit, normal, face_index

	def select_faces(self, context, event, add=True):

		scene = context.scene
		props = scene.rflow_props

		mouse_pos = event.mouse_region_x, event.mouse_region_y
		hit, normal, index = self.get_origin(context, mouse_pos)

		def assign_extract_faces(add, index):

			undo_index = []

			if add:
				if index not in self.extract_faces:
					self.extract_faces.append(index)
					undo_index.append(index)
			else:
				if index in self.extract_faces:
					self.extract_faces.remove(index)
					undo_index.append(index)

			return undo_index

		if hit:
			if self.select_plus or \
				self.loop_select:

				dim = self.orig_obj.dimensions.copy()
				avg_dim = sum(d for d in dim)/len(dim)

				mw = self.orig_dup.matrix_world
				mesh = self.orig_dup.data

				bm = bmesh.new()
				bm.from_mesh(mesh)

				bm.faces.ensure_lookup_table()

				if self.select_plus:
					pick_normal = bm.faces[index].normal
					active_median = bm.faces[index].calc_center_median() @ self.orig_dup.matrix_world

					result = set()
					traversal_stack = [bm.faces[index]]

					while len(traversal_stack) > 0:
						f_curr = traversal_stack.pop()
						result.add(f_curr)

						for e in f_curr.edges:
							if e.is_contiguous and e.smooth and not e.seam:
								for f_linked in e.link_faces:
									if f_linked not in result:
										if (f_linked.calc_center_median()-active_median).length <= ((avg_dim / 2) * props.select_influence):
											angle = f_curr.normal.angle(f_linked.normal, 0.0)
											if angle < radians(30): traversal_stack.append(f_linked)

					if result:
						undo_list = []
						for f in result:
							plus_selection = assign_extract_faces(add, f.index)
							undo_list.extend(plus_selection)

						if undo_list: self.undo_faces.append(undo_list)

				if self.loop_select:
					face = bm.faces[index]
					loc = None
					closest_edge = None
					start_loop = None

					def face_loop_select(start_loop, limit, reverse=False):

						indices = []
						cancel = False

						for i in range(limit):
							if not reverse:
								next_loop = start_loop.link_loop_next.link_loop_radial_next.link_loop_next
								next_edge = start_loop.link_loop_next.edge
							else:
								next_loop = start_loop.link_loop_prev.link_loop_radial_prev.link_loop_prev
								next_edge = start_loop.link_loop_prev.edge

							if next_loop.face == face or \
								len(next_loop.face.edges) != 4:
								cancel = True

							angle = next_edge.calc_face_angle(None)
							if next_edge.is_boundary or \
								next_edge.smooth == False or next_edge.seam: cancel = True

							if cancel: break

							selection = assign_extract_faces(add, next_loop.face.index)
							if selection: indices.extend(selection)

							start_loop = next_loop

						return indices

					for i, e in enumerate(face.loops):
						vloc = sum([(mw @ v.co) for v in e.edge.verts], Vector()) / len(e.edge.verts)
						coord_2d = v3d_to_v2d(context, [vloc])
						closest_edge = (Vector((mouse_pos)) - Vector((coord_2d[0][0], coord_2d[0][1]))).length

						if not loc: loc = closest_edge
						if closest_edge <= loc:
							loc = closest_edge
							start_loop = face.loops[i-1]

					if start_loop:
						undo_list = []

						first_face = assign_extract_faces(add, start_loop.face.index)
						undo_list.extend(first_face)

						for i in range(0, 2):
							face_loops = face_loop_select(start_loop, len(bm.faces), reverse=False if i else True)
							undo_list.extend(face_loops)

						if undo_list: self.undo_faces.append(undo_list)

				bm.to_mesh(mesh)
				mesh.update()
			else:
				undo_list = assign_extract_faces(add, index)
				if undo_list: self.undo_faces.append(undo_list)

			self.update_extract(context, self.extr_obj)

	def get_closest(self, obj, hit, radius):

		mesh = obj.data

		size = len(mesh.vertices)
		kd = mathutils.kdtree.KDTree(size)

		for i, v in enumerate(mesh.vertices):
			kd.insert(v.co, i)

		kd.balance()

		co_find = obj.matrix_world.inverted() @ hit

		vertices = []
		for (co, index, dist) in kd.find_range(co_find, radius):
			vertices.append(index)

		return vertices

	def create_applied_mesh(self, context, orig_obj, orig_dup):

		def apply_modifier_list(obj, apply_list):

			mod = obj.modifiers

			for m in mod:
				if m.type not in apply_list:
					mod.remove(m)

			obj.data = get_evaluated_mesh(context, obj).copy()
			obj.modifiers.clear()
			clear_customdata(obj)
			bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

		for obj in context.selected_objects:
			obj.select_set(False if obj != orig_dup else True)

		context.view_layer.objects.active = orig_dup
		orig_loc, orig_rot, orig_scale = orig_obj.matrix_world.decompose()
		orig_dup.scale = orig_scale
		orig_dup.rotation_euler = orig_rot.to_euler()
		orig_dup.location = orig_loc

		copy_modifiers([orig_obj, orig_dup])
		apply_modifier_list(orig_dup, ['MIRROR', 'SOLIDIFY', 'BOOLEAN', 'ARRAY', 'SUBSURF'])

		self.orig_dup.hide_set(True)

	def new_mesh_data(self, name):

		new_data = bpy.data.meshes.new(name)
		new_obj = bpy.data.objects.new(name, new_data)

		return new_obj

	def clear_mesh_list(self):

		for o in bpy.data.meshes:
			if o not in self.mesh_list \
				and o.users == 0:
				bpy.data.meshes.remove(o)

	def clear_extract(self):

		remove_obj(self.orig_dup)
		self.clear_mesh_list()

	def cancel_extract(self, context):

		remove_obj(self.extr_obj)
		self.clear_mesh_list()

		self.initialize(context, set=False)
		self.remove_handlers()

		self.orig_obj.select_set(True)
		context.view_layer.objects.active = self.orig_obj

	def remove_handlers(self):

		bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
		bpy.types.SpaceView3D.draw_handler_remove(self._handle1, 'WINDOW')

	def modal(self, context, event):
		context.area.tag_redraw()

		scene = context.scene
		props = scene.rflow_props
		extr_obj = self.extr_obj
		self.mouse_co = event.mouse_region_x, event.mouse_region_y

		if event.type == 'MIDDLEMOUSE':
			self.render_hit = False
		else: self.render_hit = True

		if event.type in {
			'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE',
			'NUMPAD_1', 'NUMPAD_2', 'NUMPAD_3', 'NUMPAD_4', 'NUMPAD_6',
			'NUMPAD_7', 'NUMPAD_8', 'NUMPAD_9', 'NUMPAD_5'}:
			return {'PASS_THROUGH'}

		if event.type in {'TAB'}:
			if event.value == 'RELEASE':
				view = context.region_data.view_perspective
				if view == "PERSP":
					context.region_data.view_perspective = "ORTHO"
				else: context.region_data.view_perspective = "PERSP"

		if event.type == 'H':
			if event.value == 'RELEASE':
				self.help_index += 1
				if self.help_index > 2: self.help_index = 1

		if event.type == 'X':
			if event.value == 'RELEASE':
				self.draw_solid ^= True

		if not event.ctrl:
			self.draw_strips = False
			self.render_hit = True
			context.window.cursor_modal_restore()

		if event.type == 'MOUSEMOVE':
			if self.view_render_hit:
				mouse_pos = event.mouse_region_x, event.mouse_region_y
				hit, normal, index = self.get_origin(context, mouse_pos)

				if hit:
					rv3d = context.region_data
					mesh = self.orig_dup.data
					draw_face = mesh.polygons[index]

					if self.view_mat1 != rv3d.view_matrix:
						vertices = np.empty((len(mesh.vertices), 3), 'f')
						mesh.vertices.foreach_get("co", np.reshape(vertices, len(mesh.vertices) * 3))

						mwv = [self.orig_dup.matrix_world @ Vector(v) for v in vertices]
						self.hit_verts = v3d_to_v2d(context, mwv)

						# refresh?
						self.view_mat1 = rv3d.view_matrix.copy()

					self.hit_indices = [k for i, k in enumerate(draw_face.edge_keys)]

			if self.lmb:
				self.select_faces(context, event, True)

			if self.rmb:
				self.select_faces(context, event, False)

			if event.ctrl:
				self.draw_strips = True
				self.render_hit = False
				context.window.cursor_modal_set("CROSSHAIR")

				if extr_obj.data.polygons:
					self.draw_strips = True

					delta_x, delta_y = get_delta(context, event, extr_obj, local_center(extr_obj))
					v2 = delta_increment(event, delta_x, delta_y, self.avg_dim)

					if delta_x <= delta_y:
						self.inset_val += v2
					else:
						self.inset_val -= v2

					self.update_extract(context, extr_obj)

		if event.type == 'A':
			if event.value == 'PRESS':
				incr = 0.01 if event.shift else 0.1
				props.select_influence -= incr

		if event.type == 'D':
			if event.value == 'PRESS':
				incr = 0.01 if event.shift else 0.1
				props.select_influence += incr

		if event.type == 'LEFTMOUSE':
			if event.shift and \
				not event.alt and \
				not event.ctrl:
				self.select_plus = True
			else: self.select_plus = False

			if event.shift \
				and event.alt \
				and not event.ctrl:
					self.loop_select = True
			else: self.loop_select = False

			self.select_faces(context, event, True)
			self.lmb = event.value == 'PRESS'

		if event.type == 'RIGHTMOUSE':
			if event.shift and \
				not event.alt and \
				not event.ctrl:
				self.select_plus = True
			else: self.select_plus = False

			if event.shift \
				and event.alt \
				and not event.ctrl:
					self.loop_select = True
			else: self.loop_select = False

			self.select_faces(context, event, False)
			self.rmb = event.value == 'PRESS'

		if event.type == 'Z':
			if event.value == 'PRESS':
				select_list = self.undo_faces
				if len(select_list) > 0:
					for i in select_list[-1]:
						if i in self.extract_faces:
							self.extract_faces.remove(i)
						else:
							self.extract_faces.append(i)

					select_list.remove(select_list[-1])
					self.update_extract(context, self.extr_obj)

		if event.type == 'R':
			if event.value == 'RELEASE':
				self.inset_val = 0.0
				self.undo_faces.append([i for i in self.extract_faces])
				self.extract_faces.clear()
				self.update_extract(context, self.extr_obj)

		if event.type == 'T':
			if event.value == 'RELEASE':
				self.inset_val = 0.0
				self.update_extract(context, self.extr_obj)

		if event.type == 'V':
			if event.value == 'RELEASE': self.view_render_hit ^= True

		if event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'}:
			if event.value == 'PRESS':
				if len(self.extract_faces) > 0:
					scene.collection.objects.link(extr_obj)

					extr_obj.select_set(True)
					context.view_layer.objects.active = extr_obj

					move_center_origin(self.orig_obj.matrix_world.translation, extr_obj)
					copy_rotation(self.orig_obj, extr_obj)

					self.initialize(context, set=False)
					self.remove_handlers()

					return {'FINISHED'}
				else:
					self.report({'WARNING'}, "No selected faces.")
					self.cancel_extract(context)

					return {'FINISHED'}

		if event.type in {'ESC'}:
			self.cancel_extract(context)

			return {'CANCELLED'}

		return {'RUNNING_MODAL'}

	def invoke(self, context, event):
		if context.area.type == 'VIEW_3D':
			args = (self, context)

			obj = self.orig_obj = context.active_object

			dim = obj.dimensions.copy()
			self.avg_dim = sum(d for d in dim)/len(dim)
			self.mesh_list = [o for o in bpy.data.meshes]

			self.orig_dup = duplicate_obj('Temp_Mesh', obj, get_eval=False)
			self.orig_dup.data.materials.clear()
			self.extr_obj = self.new_mesh_data("Extr_Proxy")

			self.help_index = 1

			self.extract_faces = []
			self.undo_faces = []
			self.hit_verts = []
			self.hit_indices = []

			self.lmb = False
			self.rmb = False

			self.view_mat1 = Matrix()

			self.render_hit = True
			self.view_render_hit = True
			self.loop_select = False
			self.select_plus = False

			self.draw_strips = False
			self.draw_solid = True

			self.inset_val = 0.0
			self.inset_indv = False

			self.mouse_co = []

			self.initialize(context, set=True)

			self._handle = bpy.types.SpaceView3D.draw_handler_add(draw_callback_px_draw_extract, args, 'WINDOW', 'POST_PIXEL')
			self._handle1 = bpy.types.SpaceView3D.draw_handler_add(draw_callback_px_draw_extract_shade, args, 'WINDOW', 'POST_VIEW')
			context.window_manager.modal_handler_add(self)

			return {'RUNNING_MODAL'}
		else:
			self.report({'WARNING'}, "View3D not found, cannot run operator.")

			return {'CANCELLED'}

class MESH_OT_apply_mesh(Operator):
	'''Apply mirror modifiers on selected objects'''
	bl_idname = 'apply_mesh.rflow'
	bl_label = 'Apply Mesh'
	bl_options = {'REGISTER', 'UNDO'}

	mirror_only : BoolProperty(
		name        = "Apply Mirror Modifier Only",
		description = "Only apply mirror modifier in selected objects",
		default     = False
		)
	join_objs : BoolProperty(
		name        = "Join Objects",
		description = "Join objects as one mesh",
		default     = False
		)
	mesh_to_vg : BoolProperty(
		name        = "Mesh To Vertex Group",
		description = "Assign mesh vertices to vertex group before joining",
		default     = True
		)
	list : StringProperty(
		name        = "Join To",
		description = "Join selected objects to this object"
		)
	meshes : CollectionProperty(type=PropertyGroup)

	@classmethod
	def poll(cls, context):
		return context.active_object is not None and context.active_object.mode == "OBJECT"

	def verts_to_vgroup(self, obj):

		vg = obj.name
		group = obj.vertex_groups.get(vg) or obj.vertex_groups.new(name=vg)
		group_index = group.index

		mesh = obj.data
		bm = bmesh.new()
		bm.from_mesh(mesh)

		deform_layer = bm.verts.layers.deform.active
		if deform_layer is None: deform_layer = bm.verts.layers.deform.new()

		for v in bm.verts:
			v[deform_layer][group_index] = 1.0

		bm.to_mesh(mesh)
		bm.free()

	def execute(self, context):
		actv_obj = bpy.data.objects.get(self.list)
		objs = context.selected_objects

		for o in objs:
			o.select_set(True)
			context.view_layer.objects.active = o

			if o.type == 'MESH':
				if self.join_objs \
					and self.mesh_to_vg:
					o.vertex_groups.clear()
					self.verts_to_vgroup(o)

				mods = o.modifiers
				if self.mirror_only:
					for m in mods:
						if m.type == "MIRROR":
							bpy.ops.object.modifier_move_to_index(modifier=m.name, index=0)
							bpy.ops.object.modifier_apply(modifier=m.name)
				else:
					if o.data.materials:
						for m in mods:
							try:
								bpy.ops.object.modifier_apply(modifier=m.name)
							except:
								o.modifiers.remove(m)
					else:
						o.data = get_evaluated_mesh(context, o).copy()
						o.modifiers.clear()
			elif o.type == 'CURVE':
				bpy.ops.object.convert(target='MESH')
			else:
				o.select_set(False)

		context.view_layer.objects.active = actv_obj
		if self.join_objs: bpy.ops.object.join()

		return {"FINISHED"}

	def draw(self, context):
		layout = self.layout
		col = layout.column()
		col.prop(self, "mirror_only")
		col.prop(self, "join_objs")
		if self.join_objs:
			row = col.row().split(factor=0.27, align=True)
			row.label(text="Join To:")
			row.prop_search(
				self,
				"list",
				self,
				"meshes",
				text="",
				icon = "MESH_DATA"
				)
			col.separator(factor=0.5)
			row = col.row().split(factor=0.27, align=True)
			row.separator()
			row.prop(self, "mesh_to_vg")

	def invoke(self, context, event):
		self.join_objs = False
		self.list = ""
		self.meshes.clear()

		objs = context.selected_objects
		if objs:
			for o in objs:
				if o.type in ['MESH', 'CURVE']:
					newListItem = self.meshes.add()
					newListItem.name = o.name

			if self.meshes: self.list = self.meshes[0].name

			return context.window_manager.invoke_props_dialog(self)
		else:
			self.report({'WARNING'}, "No objects selected.")
			return {"FINISHED"}

class MESH_OT_scatter_origin(Operator):
	'''Sets origin for scatter objects'''
	bl_idname = 'set_origin.rflow'
	bl_label = 'Set Origin'
	bl_options = {'REGISTER', 'UNDO'}

	origin : EnumProperty(
		name = "Origin",
		items = (
			('AXIS', 'Axis','Use axes for new origin'),
			('SELECTED', 'Selected','Use selected verts for new origin')),
		default = 'AXIS')
	space : EnumProperty(
		name = "Space",
		items = (
			('LOCAL', 'Local','Use objects local matrix'),
			('GLOBAL', 'Global','Use global matrix')),
		default = 'LOCAL')
	axis : EnumProperty(
		name = "Origin",
		items = (
			('X', 'X',''),
			('Y', 'Y',''),
			('Z', 'Z','')),
		default = 'Z')
	location : EnumProperty(
		name = "Location",
		items = (
			('NEGATIVE', 'Negative','Find outermost verts in the negative axis direction'),
			('POSITIVE', 'Positive','Find outermost verts in the positive axis direction')),
		default = 'NEGATIVE')
	tolerance : FloatProperty(
		name        = "Tolerance",
		description = "Tolerance threshold for finding verts based on location",
		default     = 1e-5,
		min         = 0.0,
		soft_max    = 1.0,
		step        = 1.0,
		precision   = 5
		)

	@classmethod
	def poll(cls, context):
		return context.active_object is not None and context.active_object.mode == "OBJECT"

	def execute(self, context):
		objs = context.selected_objects

		for o in objs:
			mesh = o.data
			M = o.matrix_world if self.space == 'GLOBAL' else Matrix()

			bm = bmesh.new()
			bm.from_mesh(mesh)

			point = None
			if self.origin == 'AXIS':
				axis = ['X','Y','Z'].index(self.axis)
				verts = sorted(bm.verts, key=lambda v: (M @ v.co)[axis])
				pos = (M @ verts[-1 if self.location == 'POSITIVE' else 0].co)[axis]
				point = [o.matrix_world @ v.co for v in verts if abs((M @ v.co)[axis] - pos) < self.tolerance]
			else:
				point = [o.matrix_world @ v.co for v in bm.verts if v.select]
				if not point:
					self.report({'WARNING'}, "No verts selected.")

			bm.to_mesh(mesh)
			bm.free()

			if point:
				new_origin = sum(point, Vector()) / len(point)
				move_center_origin(new_origin, o)

		return {"FINISHED"}

	def draw(self, context):
		layout = self.layout
		col = layout.column()
		row = col.row().split(factor=0.2, align=True)
		row.label(text="Origin:")
		row.row(align=True).prop(self, "origin", expand=True)
		col1 = col.column()
		col1.enabled = self.origin == 'AXIS'
		row = col1.row().split(factor=0.2, align=True)
		row.label(text="Space:")
		row.row(align=True).prop(self, "space", expand=True)
		row = col1.row().split(factor=0.2, align=True)
		row.label(text="Axis:")
		row.row(align=True).prop(self, "axis", expand=True)
		row = col1.row().split(factor=0.2, align=True)
		row.label(text="Location:")
		row.row(align=True).prop(self, "location", expand=True)
		row = col1.row().split(factor=0.2, align=True)
		row.label(text="Tolerance:")
		row.row(align=True).prop(self, "tolerance", expand=True)

	def invoke(self, context, event):

		return context.window_manager.invoke_props_dialog(self)

class MESH_OT_clean_up(Operator):
	'''Limited dissolve, remove doubles and zero area faces from selected objects'''
	bl_idname = 'clean_up.rflow'
	bl_label = 'Clean Up'
	bl_options = {'REGISTER', 'UNDO'}

	rem_double_faces : BoolProperty(
		name        = "Remove Face Doubles",
		description = "Remove overlapping faces",
		default     = False
		)
	rem_doubles : BoolProperty(
		name        = "Remove Doubles",
		description = "Remove overlapping verts",
		default     = False
		)
	doubles_dist : FloatProperty(
		name        = "Merge Distance",
		description = "Maximum distance between elements to merge",
		default     = 0.0001,
		min         = 0.0,
		soft_max    = 10.0,
		step        = 0.1,
		precision   = 4
		)
	lim_dissolve : BoolProperty(
		name        = "Limited Dissolve",
		description = "Use limited dissolve to unify faces",
		default     = False
		)
	angle : FloatProperty(
		name        = "Max Angle",
		description = "Angle limit",
		default     = radians(5),
		min         = 0,
		max         = radians(180),
		step        = 10,
		precision   = 3,
		subtype     = "ANGLE"
		)
	use_clip : BoolProperty(
		name        = "Clip Center",
		description = "Clip center verts when using mirror modifier",
		default     = False
		)
	clip_dist : FloatProperty(
		name        = "Clip Distance",
		description = "Distance within which center vertices are clipped",
		default     = 0.001,
		min         = 0,
		soft_max    = 1.0,
		step        = 0.1,
		precision   = 4
		)
	clip_axis : BoolVectorProperty(
		name        = "Clip Axis",
		description = "Clip axis toggles",
		default     = (True, True, True),
		size		= 3,
		subtype		= "XYZ"
		)
	offset_range : IntProperty(
		name        = "Offset Range",
		description = "Number of adjacent faces for verts to be offset",
		default     = 2,
		min         = 0,
		max         = 32767,
		step        = 1
		)
	center_offset : FloatProperty(
		name        = "Center Offset",
		description = "Offset of symmetry verts",
		default     = 0.0,
		soft_min    = -1,
		soft_max    = 1,
		step        = 0.01,
		precision   = 4
	)
	offset_axis : BoolVectorProperty(
		name        = "Smooth",
		description = "Clip smooth axis toggles",
		default     = (True, True, True),
		size		= 3,
		subtype		= "XYZ"
		)
	deg_dissolve : BoolProperty(
		name        = "Degenerate Dissolve",
		description = "Remove zero area faces and zero length edges",
		default     = False
		)
	deg_dist : FloatProperty(
		name        = "Merge Distance",
		description = "Maximum distance between elements to merge",
		default     = 0.0001,
		min         = 0.0,
		soft_max    = 10.0,
		step        = 0.1,
		precision   = 4
		)
	remove_singles : BoolProperty(
		name        = "Remove 2EV's",
		description = "Remove verts with only two connecting edges",
		default     = False
		)

	@classmethod
	def poll(cls, context):
		return context.active_object is not None and context.active_object.type == 'MESH'

	def clean_double_faces(self, bm):

		double_faces = []
		f_centers = [tuple(f.calc_center_median()) for f in bm.faces]
		dup_centers = [k for k, v in Counter(f_centers).items() if v > 1]
		for f in bm.faces:
			if tuple(f.calc_center_median()) in dup_centers \
				and not f in double_faces: double_faces.append(f)

		bmesh.ops.delete(bm, geom=double_faces, context='FACES')

	def execute(self, context):
		objs = context.selected_objects

		rem_count = 0
		for o in objs:
			if o.type == 'MESH':
				mesh = o.data
				if mesh.is_editmode:
					bm = bmesh.from_edit_mesh(mesh)
				else:
					bm = bmesh.new()
					bm.from_mesh(mesh)

				vcount = len(bm.verts)

				if self.rem_doubles:
					bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=self.doubles_dist)

				if self.lim_dissolve:
					bmesh.ops.dissolve_limit(bm, angle_limit=self.angle, \
						use_dissolve_boundaries=False, verts=bm.verts, edges=bm.edges, delimit={'NORMAL'})

				if self.use_clip:
					clip_center(bm, o, self.clip_dist, self.clip_axis, self.offset_range, \
						self.center_offset, self.offset_axis)
					remove_axis_faces(bm, o)

				if self.deg_dissolve:
					bmesh.ops.dissolve_degenerate(bm, dist=self.deg_dist, edges=bm.edges)

				if self.remove_singles:
					singles = get_singles(bm.verts)
					bmesh.ops.dissolve_verts(bm, verts=singles)

				new_vcount = vcount - len(bm.verts)
				rem_count += new_vcount

				if mesh.is_editmode:
					bmesh.update_edit_mesh(mesh)
				else:
					bm.to_mesh(mesh)
					mesh.update()
					bm.free()

		self.report({'INFO'}, "Removed " + str(rem_count) + " vertices.")

		return {"FINISHED"}

	def draw(self, context):
		layout = self.layout
		col = layout.column()
		col.prop(self, "rem_doubles")
		if self.rem_doubles:
			col.prop(self, "doubles_dist")
		col.prop(self, "lim_dissolve")
		if self.lim_dissolve:
			col.prop(self, "angle")
		col.prop(self, "deg_dissolve")
		if self.deg_dissolve:
			col.prop(self, "deg_dist")
		col.prop(self, "use_clip")
		if self.use_clip:
			flow = col.column_flow(columns=2, align=True)
			flow.prop(self, "clip_dist")
			flow.row(align=True).prop(self, "clip_axis", text="", expand=True)
			flow = col.column_flow(columns=2, align=True)
			flow.prop(self, "center_offset")
			flow.row(align=True).prop(self, "offset_axis", text="", expand=True)
			col.prop(self, "offset_range")
		col.prop(self, "remove_singles")

	def invoke(self, context, event):

		return context.window_manager.invoke_props_dialog(self, width=300)

class MESH_OT_manage_data(Operator):
	'''Save, use or clear saved mesh data'''
	bl_idname = 'manage_data.rflow'
	bl_label = 'Save/Use/Clear Mesh Data'
	bl_options = {'REGISTER', 'UNDO'}

	mode : EnumProperty(
		items = (
			('SAVE', 'Save','Save mesh data to list'),
			('USE', 'Use','Use mesh data from list'),
			('CLEAR', 'Clear','Clear mesh data list')),
		default = 'SAVE')
	list : StringProperty(
		name        = "Mesh",
		description = "Mesh data"
		)
	meshes : CollectionProperty(type=PropertyGroup)

	@classmethod
	def poll(cls, context):
		return context.active_object is not None

	def use_save_data(self, mesh, save_mesh):

		if mesh.is_editmode:
			bm = bmesh.from_edit_mesh(mesh)
		else:
			bm = bmesh.new()
			bm.from_mesh(mesh)

		bm.clear()
		bm.from_mesh(save_mesh)

		if mesh.is_editmode:
			bmesh.update_edit_mesh(mesh)
		else:
			bm.to_mesh(mesh)
			mesh.update()

	def get_similarity(self, orig, comp_mesh):

		orig.update_from_editmode()
		mesh1 = orig.data
		p1 = np.empty((len(mesh1.vertices), 3), 'f')
		mesh1.vertices.foreach_get("co", np.reshape(p1, len(mesh1.vertices) * 3))

		mesh2 = comp_mesh
		p2 = np.empty((len(mesh2.vertices), 3), 'f')
		mesh2.vertices.foreach_get("co", np.reshape(p2, len(mesh2.vertices) * 3))

		p1 = list(map(tuple, p1))
		p2 = list(map(tuple, p2))

		similarity = len((set(p2).intersection(set(p1))))/len(p2) * 100

		return similarity

	def execute(self, context):
		obj = context.active_object
		suffix = self.suffix

		mesh = obj.data

		if self.mode == 'SAVE':
			if obj: obj.select_set(True)
			for o in context.selected_objects:
				o.update_from_editmode()
				mesh_data = o.data.copy()
				if mesh_data.name.find(suffix) == -1:
					mesh_data.name += suffix
				mesh_data.use_fake_user = True
		if self.mode == "USE":
			if self.list:
				save_mesh = bpy.data.meshes.get(self.list)
				if save_mesh: self.use_save_data(mesh, save_mesh)
		if self.mode == 'CLEAR':
			rem_count = 0
			for m in bpy.data.meshes:
				if m.name.find(suffix) != -1 \
					and m.use_fake_user:
					m.use_fake_user = False
					if m.users < 1:
						rem_count += 1
						bpy.data.meshes.remove(m)

			self.report({'INFO'}, str(rem_count) + " saved mesh data removed.")

		return {"FINISHED"}

	def draw(self, context):
		if self.mode == 'USE':
			layout = self.layout
			col = layout.column()
			row = col.row().split(factor=0.27, align=True)
			row.label(text="Mesh Data:")
			row.prop_search(
				self,
				"list",
				self,
				"meshes",
				text="",
				icon = "MESH_DATA"
				)
			col.separator(factor=0.5)
			col.box().label(text=self.same_info, icon="INFO")
		else: None

	def invoke(self, context, event):
		self.list = ""
		self.meshes.clear()
		self.suffix = suffix = "_rf_data"
		self.same_info = ""

		if self.mode == 'USE':
			saved_meshes = { m: self.get_similarity(context.active_object, m) \
				for m in bpy.data.meshes if m.name.find(suffix) != -1 and m.use_fake_user }
			sorted_meshes = sorted(saved_meshes.items(), key=lambda item: item[1])
			same = 0
			for m in reversed(sorted_meshes):
				newListItem = self.meshes.add()
				newListItem.name = m[0].name
				if m[1] > 0: same += 1

			self.same_info = "Found " + str(same) + " matching geometry."

			return context.window_manager.invoke_props_dialog(self)
		else:
			return self.execute(context)

class MESH_OT_merge_objs(Operator):
	'''Merge selected meshes to an object'''
	bl_idname = 'merge_objs.rflow'
	bl_label = 'Merge Objects'
	bl_options = {'REGISTER', 'UNDO'}

	list : StringProperty(
		name        = "Merge To",
		description = "Merge selected objects to this object"
		)
	meshes : CollectionProperty(type=PropertyGroup)
	offset : FloatProperty(
		name        = "Mesh Offset",
		description = "Offset mesh by this amount to fix overlap",
		default     = 0.0001,
		min			= 0.0,
		soft_min    = 0.0001,
		soft_max    = 1.0,
		step        = 0.01,
		precision   = 4
		)

	@classmethod
	def poll(cls, context):
		return context.active_object is not None and context.active_object.mode == "OBJECT"

	def group_obj(self, obj, coll_name):

		coll = bpy.data.collections.get(coll_name) or bpy.data.collections.new(coll_name)
		coll.objects.link(obj)

	def execute(self, context):
		merge_obj = bpy.data.objects.get(self.list)
		coll_name = "rflow_merge_objs"
		merge_objs = set()

		if merge_obj:

			def list_new_obj(o):

				self.group_obj(o, coll_name)
				merge_objs.add(o)
				o.hide_set(True)

			for obj in context.selected_objects:
				if obj.type == 'MESH' \
					and obj != merge_obj:
					obj.data.materials.clear()
					obj.data = get_evaluated_mesh(context, obj).copy()
					obj.modifiers.clear()

					mesh = obj.data
					lparts = get_islands(obj, None, use_bm=False)

					if len(lparts) > 1:
						for p in lparts:
							bm_loose = bmesh.new()
							temp_mesh = bpy.data.meshes.new(".temp")
							bm_loose.from_mesh(mesh)

							v = [v for v in bm_loose.verts if not v.index in p]
							bmesh.ops.delete(bm_loose, geom=v, context='VERTS')

							e = [e for e in bm_loose.edges if e.is_boundary]
							bmesh.ops.holes_fill(bm_loose, edges=e)

							bm_loose.to_mesh(temp_mesh)
							bm_loose.free()

							if temp_mesh.polygons:
								new_obj = bpy.data.objects.new(obj.name, temp_mesh)
								orig_loc, orig_rot, orig_scale = obj.matrix_world.decompose()
								new_obj.scale = orig_scale
								new_obj.rotation_euler = orig_rot.to_euler()
								new_obj.location = orig_loc
								new_obj.data.use_auto_smooth = obj.data.use_auto_smooth
								new_obj.data.auto_smooth_angle = obj.data.auto_smooth_angle
								bpy.context.scene.collection.objects.link(new_obj)

								list_new_obj(new_obj)

						remove_obj(obj)
					else:
						new_obj = obj
						list_new_obj(new_obj)

			if merge_objs:
				old_data = merge_obj.data.copy()
				merge_obj.data.materials.clear()
				merge_obj.data = get_evaluated_mesh(context, merge_obj).copy()
				merge_obj.modifiers.clear()
				for m in old_data.materials:
					merge_obj.data.materials.append(m)

				mesh = merge_obj.data

				if self.offset > 0:
					bm = bmesh.new()
					bm.from_mesh(mesh)

					for v in bm.verts:
						v.co += v.normal * (v.calc_shell_factor() * self.offset)

					bm.to_mesh(mesh)
					bm.free()

				mod = merge_obj.modifiers
				bool_op = mod.new('Boolean', 'BOOLEAN')
				bool_op.operand_type = 'COLLECTION'
				bool_op.collection = bpy.data.collections[coll_name]
				bool_op.operation = 'UNION'
				bool_op.solver = 'EXACT'
				bool_op.show_expanded = False

				merge_obj.select_set(True)
				context.view_layer.objects.active = merge_obj
				bpy.ops.object.modifier_apply(modifier=bool_op.name)

				for o in merge_objs: remove_obj(o)

		return {"FINISHED"}

	def draw(self, context):
		layout = self.layout
		col = layout.column()
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Merge To:")
		row.prop_search(
			self,
			"list",
			self,
			"meshes",
			text="",
			icon = "MESH_DATA"
			)
		row = col.row().split(factor=0.27, align=True)
		row.label(text="Mesh Offset:")
		row.row(align=True).prop(self, "offset", text="")

	def invoke(self, context, event):
		self.list = ""
		self.meshes.clear()

		for o in context.selected_objects:
			if o.type == 'MESH':
				newListItem = self.meshes.add()
				newListItem.name = o.name

		return context.window_manager.invoke_props_dialog(self)

class MESH_OT_use_info(Operator):
	'''Display important usage information'''
	bl_idname = 'use_info.rflow'
	bl_label = 'Help'

	def execute(self, context): return {"FINISHED"}

	def draw(self, context):
		margin = " " * 9
		layout = self.layout
		col = layout.column(align=False)
		col.label(text="Usage Information:", icon="INFO")
		col.separator(factor=0.5)
		col.label(text=margin + "Select faces in edit mode to use for randomizing.")
		col.label(text=margin + "Go back to object mode to use random operators.")
		col.label(text=margin + "Faces needs to be quads or tris for subdivision to work.")
		col.separator(factor=0.5)
		col.label(text=margin + "Press F9 to bring back redo panel when it disappears.")
		col.label(text=margin + "Performing some commands will finalize last action and scrub the redo panel from history.")
		col.label(text=margin + "In user preferences of the add-on, you can use the confirm type redo panels instead.")
		col.separator(factor=0.5)
		col.label(text="Limitations:", icon="ERROR")
		col.label(text=margin + "Be careful with using higher resolution face selections.")
		col.label(text=margin + "(Most operations run recursively)")
		col.label(text=margin + "(It can run out of memory and take a long time to compute)")

	def invoke(self, context, event):

		return context.window_manager.invoke_props_dialog(self, width=500)

class UI_MT_random_flow(Menu):
	bl_label = "Random FLow"
	bl_idname = "UI_MT_random_flow"

	def draw(self, context):
		obj = context.active_object
		prefs = context.preferences.addons[__name__].preferences

		layout = self.layout
		layout.operator_context = 'INVOKE_REGION_WIN'
		layout.operator("rand_loop_extr.rflow", icon="ORIENTATION_NORMAL")
		layout.operator("rand_panels.rflow", icon="MESH_GRID")
		layout.operator("rand_axis_extr.rflow", icon="SORTBYEXT")
		layout.operator("rand_scatter.rflow", icon="OUTLINER_OB_POINTCLOUD")
		layout.operator("rand_tubes.rflow", icon="IPO_CONSTANT")
		layout.operator("rand_cables.rflow", icon="FORCE_CURVE")
		layout.operator("rand_vcol.rflow", icon="COLORSET_10_VEC")
		layout.separator()
		if obj:
				layout.operator("make_flanges.rflow", icon="GP_ONLY_SELECTED")
				layout.operator("panel_screws.rflow", icon="GRIP")
				layout.separator()
		if obj \
			and obj.type == 'MESH' \
			and obj.data.is_editmode:
			layout.operator("grid_project.rflow", icon="MESH_GRID")
			layout.operator("quad_slice.rflow", icon="GRID")
			layout.operator("tag_verts.rflow", icon="VERTEXSEL")
		if obj \
			and not obj.data.is_editmode:
			if obj.type == 'MESH':
				layout.operator("auto_mirror.rflow", icon="EMPTY_ARROWS")
			layout.operator("extr_proxy.rflow", icon="FACESEL")
			layout.operator("apply_mesh.rflow", icon="MOD_MIRROR")
		layout.operator("auto_smooth.rflow", icon="OUTLINER_OB_MESH")
		layout.separator()
		layout.menu("UI_MT_rflow_mesh_data")
		layout.menu("UI_MT_rflow_extras")
		layout.menu("UI_MT_rflow_settings")
		if prefs.show_helper:
			layout.separator()
			layout.operator("use_info.rflow", text="Usage Info", icon="INFO")

class UI_MT_rflow_mesh_data(Menu):
	bl_label = "Mesh Data"
	bl_idname = "UI_MT_rflow_mesh_data"

	def draw(self, context):
		layout = self.layout
		layout.operator_context = 'INVOKE_REGION_WIN'
		layout.operator("manage_data.rflow", text="Save").mode = 'SAVE'
		layout.operator("manage_data.rflow", text="Use").mode = 'USE'
		layout.operator("manage_data.rflow", text="Clear Saved").mode = 'CLEAR'

class UI_MT_rflow_extras(Menu):
	bl_label = "Extras"
	bl_idname = "UI_MT_rflow_extras"

	def draw(self, context):
		layout = self.layout
		layout.operator_context = 'INVOKE_REGION_WIN'
		layout.operator("set_origin.rflow", icon="OBJECT_ORIGIN")
		layout.operator("clean_up.rflow", icon="MESH_DATA")
		layout.operator("merge_objs.rflow", icon="ERROR")

class UI_MT_rflow_settings(Menu):
	bl_label = "Settings"
	bl_idname = "UI_MT_rflow_settings"

	def draw(self, context):
		sce = context.scene
		rf_props = sce.rflow_props

		layout = self.layout
		layout.prop(rf_props, "select_active")
		layout.prop(rf_props, "all_mods")
		layout.prop(rf_props, "copy_temp_mats")

class UI_PT_rflow_addon_pref(AddonPreferences):
	bl_idname = __name__

	use_confirm : BoolProperty(
		default     = False,
		name        = "Use confirm menu for random operators",
		description = "Use confirm type adjust last action menu for random operators."
		)
	show_helper : BoolProperty(
		default     = True,
		name        = "Show Usage Info button",
		description = "Show usage info button in add-on menu."
		)

	def draw(self, context):
		layout = self.layout
		col = layout.column()
		row = col.row(align=True)
		row.separator(factor=2.0)
		row.prop(self, 'use_confirm')
		row = col.row(align=True)
		row.separator(factor=2.0)
		row.prop(self, 'show_helper')
		col.separator()
		wm = context.window_manager
		kc = wm.keyconfigs.user
		km = kc.keymaps['3D View Generic']
		kmi = get_hotkey_entry_item(km, 'wm.call_menu', 'UI_MT_random_flow')
		if kmi:
			col.context_pointer_set("keymap", km)
			rna_keymap_ui.draw_kmi([], kc, km, kmi, col, 0)
		else:
			col.label(text="No hotkey found!", icon="ERROR")
			col.operator("add_hotkey.rflow", text="Add hotkey")

def get_hotkey_entry_item(km, kmi_name, kmi_value):

	for i, km_item in enumerate(km.keymap_items):
		if km.keymap_items.keys()[i] == kmi_name:
			if km.keymap_items[i].properties.name == kmi_value:
				return km_item
	return None

def add_hotkey():

	addon_prefs = bpy.context.preferences.addons[__name__].preferences

	kc = bpy.context.window_manager.keyconfigs.addon
	if kc:
		km = kc.keymaps.new(name='3D View Generic', space_type='VIEW_3D', region_type='WINDOW')
		kmi = km.keymap_items.new('wm.call_menu', 'Q', 'PRESS', ctrl=False, shift=True, alt=False)
		kmi.properties.name = 'UI_MT_random_flow'
		kmi.active = True
		addon_keymaps.append((km, kmi))

def remove_hotkey():

	for km, kmi in addon_keymaps:
		km.keymap_items.remove(kmi)

	addon_keymaps.clear()

class USERPREF_OT_change_hotkey(Operator):
	'''Add hotkey'''
	bl_idname = "add_hotkey.rflow"
	bl_label = "Add Hotkey"
	bl_options = {'REGISTER', 'INTERNAL'}

	def execute(self, context):
		add_hotkey()
		return {'FINISHED'}

addon_keymaps = []

class RFlow_Props(PropertyGroup):

	select_influence : FloatProperty(
		description = "Select influence value for Extract Proxy",
		name        = "Select Influence",
		default     = 1.0,
		min         = 0,
		max         = 1.0
		)
	select_active : BoolProperty(
		default     = True,
		name        = "Select Active",
		description = "Always select active or source object after operation"
		)
	all_mods : BoolProperty(
		default     = False,
		name        = "Copy All Modifiers",
		description = "Copy all modifiers from source object to random objects"
		)
	copy_temp_mats : BoolProperty(
		default     = False,
		name        = "Copy Temp Mats",
		description = "Copy materials assigned to inner/outer faces from random extrusion and panelling"
		)

classes = (
	MESH_OT_r_loop_extrude,
	MESH_OT_r_panels,
	MESH_OT_r_axis_extrude,
	MESH_OT_r_scatter,
	MESH_OT_r_tubes,
	MESH_OT_r_cables,
	MESH_OT_r_vertex_color,
	MESH_OT_make_flanges,
	MESH_OT_panel_screws,
	MESH_OT_tag_verts,
	MESH_OT_quad_slice,
	MESH_OT_grid_project,
	MESH_OT_auto_smooth,
	MESH_OT_auto_mirror,
	MESH_OT_extract_proxy,
	MESH_OT_apply_mesh,
	MESH_OT_scatter_origin,
	MESH_OT_clean_up,
	MESH_OT_manage_data,
	MESH_OT_merge_objs,
	MESH_OT_use_info,
	UI_MT_random_flow,
	UI_MT_rflow_mesh_data,
	UI_MT_rflow_extras,
	UI_MT_rflow_settings,
	UI_PT_rflow_addon_pref,
	USERPREF_OT_change_hotkey,
	RFlow_Props,
	)

def register():
	for cls in classes:
		bpy.utils.register_class(cls)

	bpy.types.Scene.rflow_props = PointerProperty(
		type        = RFlow_Props,
		name        = "Random Flow Properties",
		description = ""
		)

	add_hotkey()

def unregister():
	for cls in reversed(classes):
		bpy.utils.unregister_class(cls)

	del bpy.types.Scene.rflow_props

	remove_hotkey()

if __name__ == '__main__':
	register()