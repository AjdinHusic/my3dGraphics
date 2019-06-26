# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 10:31:58 2019

@author: Ajdin
Procedure:

1. Object Coordinates (created at object definition)
2. World Coordinates (Modeling Transform)
3. Eye Coordinates (Viewing Transform)
4. Clip Coordinates (Projection Transform)
5. Device Coordinates (Viewport Transform)
"""
import numpy as np
import json
import copy
from transform import rotz_matrix_3
from transform import view_transform
from transform import orth_projection_transform
from transform import pers_projection_transform
from transform import viewport_transform
from transform import get_cam_frame


def getpar(_dict, val, default):
    if _dict is not None:
        return _dict.get(val, default)
    return default


class Polygon:
    def __init__(self, obj_vertices, indices):
        self.indices = indices
        self.vertices = []
        for index in indices:
            self.vertices.append(obj_vertices[index])

    def transform(
            self,
            scaling=(1, 1, 1),
            rotation=0,
            translation=(0, 0, 0)
            ):
        self.vertices = [
                list(np.diag(scaling) @ vertex)
                for vertex
                in self.vertices]
        self.vertices = [
                list(rotz_matrix_3(rotation) @ vertex)
                for vertex
                in self.vertices]
        self.vertices = [
                list(np.add(translation, vertex))
                for vertex
                in self.vertices]

    def get_edges(self):
        pass


class _3dObject:
    def __init__(self, polygons, transformation=None):
        # define object as collection of independent polygons
        self.polygons = polygons
        # default the transformations
        self.scaling = (1, 1, 1)
        self.rotation = 0
        self.translation = (0, 0, 0)
        # copy the polygons for world manipulation
        self.world_polygons = copy.deepcopy(polygons)
        self.transform(transformation)

    def transform(self, transformation):
        if transformation:
            if 'scaling' in transformation:
                self.scaling = transformation['scaling']
            if 'rotation' in transformation:
                self.rotation = transformation['rotation']
            if 'translation' in transformation:
                self.translation = transformation['translation']
            self.world_polygons = copy.deepcopy(self.polygons)
            for polygon in self.world_polygons:
                polygon.transform(
                        self.scaling,
                        self.rotation,
                        self.translation)


class World:
    def __init__(self, scene_objects=[]):
        self.scene_objects = scene_objects

    def add_object(self, scene_object):
        self.scene_objects.append(scene_object)


class Camera:
    DEF_POS = (0, 0, 0)
    DEF_MODE = 'perspective'
    DEF_ASPECT = 1.0
    DEF_RESOLUTION = (512, 512)

    def __init__(self, fov_y, up, look_at, **kwargs):
        # camera properties
        self.fov_y = fov_y  # vertical field of view
        self.up = up  # up (y-) vector as seen from the world
        self.look_at = look_at  # point in world view to look at
        self.position = getpar(kwargs, 'position', self.DEF_POS)
        self.mode = getpar(kwargs, 'mode', self.DEF_MODE)
        self.aspect = getpar(kwargs, 'aspect', self.DEF_ASPECT)
        self.resolution = getpar(kwargs, 'resolution', self.DEF_RESOLUTION)
        self.left = getpar(kwargs, 'left', -2)
        self.right = getpar(kwargs, 'right', 2)
        self.bottom = getpar(kwargs, 'bottom', -2)
        self.top = getpar(kwargs, 'top', 2)
        self.near = getpar(kwargs, 'near', 0.1)
        self.far = getpar(kwargs, 'far', 10)
        # origin and transform matrices
        self.update()

    def update(self):
        self.frame_origin = get_cam_frame(self.position, self.look_at, self.up)
        self.view_transform = view_transform(
                self.position,
                self.look_at,
                self.up)
        if hasattr(self, 'world_view'):
            del self.world_view
        if hasattr(self, 'eye_view'):
            del self.eye_view
        if hasattr(self, 'clip_view'):
            del self.clip_view

    def view_world(self, world):
        edges = []
        for obj in world.scene_objects:
            for poly in obj.world_polygons:
                for i in range(len(poly.vertices)):
                    p1 = np.append(poly.vertices[i], 1)
                    p2 = np.append(poly.vertices[i-1], 1)
                    edge = (list(p1), list(p2))
                    edges.append(edge)
        self.world_view = edges
        return self.world_view

    def view_eye(self, world):
        edges = []  # add the lines to draw
        if not hasattr(self, 'world_view'):
            self.view_world(world)
        for edge in self.world_view:
            p1 = list(self.view_transform @ edge[0])
            p2 = list(self.view_transform @ edge[1])
            edges.append((p1, p2))
        self.eye_view = edges
        return self.eye_view

    def view_clip(self, world):
        edges = []
        if not hasattr(self, 'eye_view'):
            self.view_eye(world)
        if self.mode == 'orthographic':
            args = (self.left, self.right, self.bottom,
                    self.top, self.near, self.far)
            self.clip_transform = orth_projection_transform(*args)
        else:
            args = (self.fov_y, self.aspect, self.near, self.far)
            self.clip_transform = pers_projection_transform(*args)
        for edge in self.eye_view:
            p1 = self.clip_transform @ edge[0]
            p2 = self.clip_transform @ edge[1]
            # divide by w
            p1 = list(p1 / p1[-1])
            p2 = list(p2 / p2[-1])
            edges.append((p1, p2))
        self.clip_view = edges
        return self.clip_view

    def view_device(self, world):
        edges = []
        if not hasattr(self, 'clip_view'):
            self.view_clip(world)
        self.device_transform = viewport_transform(self.resolution)
        for edge in self.clip_view:
            p1 = list(self.device_transform @ edge[0])
            p2 = list(self.device_transform @ edge[1])
            edges.append((p1, p2))
        self.device_view = edges
        return self.device_view


def load_cube(transformation=None):
    with open('CubeObject.json', 'r') as read_file:
        cube_dict = json.load(read_file)
        all_verts = cube_dict['vertices']
        all_indices = cube_dict['polygons']
    polygons = [Polygon(all_verts, indices) for indices in all_indices]
    return _3dObject(polygons, transformation)
