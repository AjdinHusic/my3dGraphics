# -*- coding: utf-8 -*-
"""
Created on Mon May 27 09:43:14 2019

@author: Ajdin

Procedure:

1. Object Coordinates (created at object definition)
2. World Coordinates (Modeling Transform)
3. Eye Coordinates (Viewing Transform)
4. Clip Coordinates (Projection Transform)
5. Device Coordinates (Viewport Transform)
"""
import pygame
import sys
import graphics
from transform import rotz_matrix_3
from transform import rotx_matrix_3


if __name__ == '__main__':
    # Camera args
    field_of_view_y = 45
    up_vector = (0, 0, 1)
    pos_target = (0, 0, 0)
    pos_camera = (0, 4, 0)
    kwargs = dict(position=pos_camera, mode='perspective')
    # Define all world objects
    cam = graphics.Camera(field_of_view_y, up_vector, pos_target, **kwargs)
    # Place Identity Cube at Origin of World
    transformation = {'scaling': (1, 1, 1),
                      'rotation': 0,
                      'translation': (0, 0, 0)}
    obj = graphics.load_cube(transformation=transformation)
    # Create the world
    world = graphics.World(scene_objects=[obj])
    # define colors
    black = (0, 0, 0)
    # run the game
    pygame.init()
    w, h = 512, 512
    screen = pygame.display.set_mode((w, h))
    clock = pygame.time.Clock()
    drag = False
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    drag = True
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    drag = False
            if event.type == pygame.MOUSEMOTION:
                if drag:
                    mouse_x, mouse_y = pygame.mouse.get_rel()
                    angle_x = -mouse_x*2
                    angle_y = -mouse_y*2
                    # pos_camera = tuple(rotx_matrix_3(angle_y) @ pos_camera)
                    # up_vector = tuple(rotx_matrix_3(angle_y) @ up_vector)
                    cam.up = up_vector
                    pos_camera = tuple(rotz_matrix_3(angle_x) @ pos_camera)
                    cam.position = pos_camera
                    cam.update()

        screen.fill((255, 255, 255))
        edges = cam.view_device(world)
        for edge in edges:
            pygame.draw.line(
                    screen,
                    black,
                    (int(edge[0][0]), int(edge[0][1])),
                    (int(edge[1][0]), int(edge[1][1])),
                    1)
        # test
        # pygame.draw.line(screen, black, (1, 1), (150, 150), 1)
        pygame.display.flip()
