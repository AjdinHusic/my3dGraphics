# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 11:59:01 2019

@author: Ajdin
"""
import numpy as np
import matplotlib.pyplot as plt
# import graphics
from mpl_toolkits.mplot3d import Axes3D


def to_4D(vec):
    vec = np.array(vec)
    narr = np.ones(4)
    narr[0:len(vec)] = vec
    return narr


def rotz_matrix_3(degrees):
    rads = np.deg2rad(degrees)
    R = np.array([[np.cos(rads), -np.sin(rads), 0],
                  [np.sin(rads), np.cos(rads), 0],
                  [0, 0, 1]])
    return R


def rotx_matrix_3(degrees):
    rads = np.deg2rad(degrees)
    R = np.array([[1, 0, 0],
                  [0, np.cos(rads), -np.sin(rads)],
                  [0, np.sin(rads), np.cos(rads)]])
    return R


def get_cam_frame(eye_pos, look_at, up):
    looking_vector = np.subtract(look_at, eye_pos)
    e3 = -looking_vector / np.linalg.norm(looking_vector)
    v = up - np.dot(up, e3)*e3
    e2 = v / np.linalg.norm(v)
    e1 = np.cross(e2, e3)
    return e1, e2, e3


def view_transform(eye_pos, look_at, up=(0, 0, 1)):
    e1, e2, e3 = get_cam_frame(eye_pos, look_at, up)
    M = np.array([e1, e2, e3, eye_pos]).T
    conc = np.array([[0, 0, 0, 1]])
    M = np.concatenate((M, conc))
    return np.linalg.inv(M)


def orth_projection_transform(left, right, bottom, top, near, far):
    P = np.array([
            [2/(right-left), 0, 0, -(right+left)/(right-left)],
            [0, 2/(top-bottom), 0, -(top+bottom)/(top-bottom)],
            [0, 0, -2/(far-near), -(far+near)/(far-near)],
            [0, 0, 0, 1]
            ])
    return P


def pers_projection_transform(fov, aspect, near, far):
    f = 1/np.tan(np.deg2rad(fov/2))
    P = np.array([
            [f/aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far+near)/(near-far), (2*near*far)/(near-far)],
            [0, 0, -1, 0]
            ])
    return P


def viewport_transform(resolution):
    w = resolution[0]
    h = resolution[1]
    V = np.array([
            [w/2, 0, 0, w/2],
            [0, h/2, 0, h/2],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
            ])
    return V

# %% matplotlib tests


def plot_plane(ax, normal):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    xl, xh = x_limits
    yl, yh = y_limits
    x_1 = np.ceil(xl)
    x_2 = np.floor(xh)
    y_1 = np.ceil(yl)
    y_2 = np.floor(yh)
    X = [xl] + list(np.arange(x_1, x_2+1, 1)) + [xh]
    Y = [yl] + list(np.arange(y_1, y_2+1, 1)) + [yh]
    X, Y = np.meshgrid(X, Y)
    Z = -(normal[0]*X + normal[1]*Y) / normal[2]
    ax.plot_wireframe(X, Y, Z, alpha=0.5)


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


if __name__ == '__main__':
    import graphics
    # define object
    obj = graphics.load_cube()
    # define points and cam
    cam_pos = (-3, -2, 0)
    look_at = (0, 0, 0)
    up = (0, 0, 1)
    kwargs = dict(position=cam_pos, mode='orthographic')
    cam = graphics.Camera(100, up, look_at, **kwargs)
    # define world
    world = graphics.World([obj])
    # creat figure
    fig = plt.figure()
    fig.set_size_inches([10, 8])
    World = fig.add_subplot(221, projection=Axes3D.name)
    Eye = fig.add_subplot(222, projection=Axes3D.name)
    Clip = fig.add_subplot(223, projection=Axes3D.name)
    Persp = fig.add_subplot(224, projection=Axes3D.name)
    # draw
    obj_col = 'r'
    vert_col = 'b'
    cam_col = 'g'
    # create edges to draw
    world_edges = cam.view_world(world)
    eye_edges = cam.view_eye(world)
    clip_edges = cam.view_clip(world)
    # draw world origin
    origin = [0, 0, 0]
    unit_x = [1, 0, 0]
    unit_y = [0, 1, 0]
    unit_z = [0, 0, 1]
    color = (0.2, 0.2, 0.2)
    World.quiver(*[origin]*3, unit_x, unit_y, unit_z, colors=[color])
    World.quiver(*cam_pos, *cam.frame_origin[0], colors=[cam_col])
    World.quiver(*cam_pos, *cam.frame_origin[1], colors=[cam_col])
    World.quiver(*cam_pos, *cam.frame_origin[2], colors=[cam_col])
    Eye.quiver(*[origin]*3, unit_x, unit_y, unit_z, colors=[cam_col])
    Clip.quiver(*[origin]*3, unit_x, unit_y, unit_z, colors=['b'])
    Persp.quiver(*[origin]*3, unit_x, unit_y, unit_z, colors=['b'])
    # draw the edges in World view
    for edge in world_edges:
        vert1 = edge[0]
        vert2 = edge[1]
        World.scatter(vert1[0], vert1[1], vert1[2], color=vert_col)
        World.scatter(vert2[0], vert2[1], vert2[2], color=vert_col)
        edge_x = [vert1[0], vert2[0]]
        edge_y = [vert1[1], vert2[1]]
        edge_z = [vert1[2], vert2[2]]
        World.plot3D(edge_x, edge_y, edge_z, color=obj_col)
    # draw the edge in Eye view
    for edge in eye_edges:
        vert1 = edge[0]
        vert2 = edge[1]
        Eye.scatter(vert1[0], vert1[1], vert1[2], color=vert_col)
        Eye.scatter(vert2[0], vert2[1], vert2[2], color=vert_col)
        edge_x = [vert1[0], vert2[0]]
        edge_y = [vert1[1], vert2[1]]
        edge_z = [vert1[2], vert2[2]]
        Eye.plot3D(edge_x, edge_y, edge_z, color=obj_col)
    # draw the edges in Clip View
    for edge in clip_edges:
        vert1 = edge[0]
        vert2 = edge[1]
        Clip.scatter(vert1[0], vert1[1], vert1[2], color=vert_col)
        Clip.scatter(vert2[0], vert2[1], vert2[2], color=vert_col)
        edge_x = [vert1[0], vert2[0]]
        edge_y = [vert1[1], vert2[1]]
        edge_z = [vert1[2], vert2[2]]
        Clip.plot3D(edge_x, edge_y, edge_z, color=obj_col)
    # draw the edges in perspective
    cam.mode = 'perspective'
    persp_edges = cam.view_clip(world)
    for edge in persp_edges:
        vert1 = edge[0]
        vert2 = edge[1]
        Persp.scatter(vert1[0], vert1[1], vert1[2], color=vert_col)
        Persp.scatter(vert2[0], vert2[1], vert2[2], color=vert_col)
        edge_x = [vert1[0], vert2[0]]
        edge_y = [vert1[1], vert2[1]]
        edge_z = [vert1[2], vert2[2]]
        Persp.plot3D(edge_x, edge_y, edge_z, color=obj_col)
    # settings
    World.set_aspect('equal')
    Eye.set_aspect('equal')
    Clip.set_aspect('equal')
    set_axes_equal(World)
    set_axes_equal(Eye)
    set_axes_equal(Clip)
    World.set_title('World View')
    Eye.set_title('Camera / Eye View')
    Clip.set_title('Clip View (Orthographic)')
    Persp.set_title('Clip View (Perspective)')
    # World text
    World.text(*unit_x, 'x')
    World.text(*unit_y, 'y')
    World.text(*unit_z, 'z')
    World.text(*origin, 'world', color='b')
    # Cam text
    World.text(*(cam_pos+cam.frame_origin[0]), 'x')
    World.text(*(cam_pos+cam.frame_origin[1]), 'y')
    World.text(*(cam_pos+cam.frame_origin[2]), 'z')
    World.text(*cam_pos, 'camera', color='b')
    Eye.text(*unit_x, 'x')
    Eye.text(*unit_y, 'y')
    Eye.text(*unit_z, 'z')
    Eye.text(*origin, 'camera', color='b')
    # Clip text
    Clip.text(*unit_x, 'x', color='b')
    Clip.text(*unit_y, 'y', color='b')
    Clip.text(*unit_z, 'z', color='b')
    pass
    # label text
    World.set_xlabel('x')
    World.set_ylabel('y')
    Eye.set_xlabel('x')
    Eye.set_ylabel('y')
    Eye.set_zlabel('z')
    Clip.set_xlabel('x')
    Clip.set_ylabel('y')
    Persp.set_xlabel('x')
    Persp.set_ylabel('y')
    # plot planes
    plot_plane(World, (0, 0, 1))
    plot_plane(Eye, (0, 0, 1))
