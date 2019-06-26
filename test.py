# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:30:56 2019

@author: Ajdin
"""
import numpy as np
import transform

aspect = 1.0
height = 512
kwargs = dict(look_at=(0, 0, 0),
              eye_pos=(0, 4, 0),
              up=(0, 0, 1),
              fov_y=45,
              aspect=aspect,
              near=0.1,
              far=10,
              height=height,
              width=aspect*height)


def world_to_device(point, forward=True, **kwargs):
    # target to look at
    t = kwargs.get('look_at')
    # camera position
    p = kwargs.get('eye_pos')
    # up pointing vector
    u = kwargs.get('up')
    # vertical field of view
    fov_y = kwargs.get('fov_y')
    # aspect ratio widht/height
    a = kwargs.get('aspect')
    # near clipping plane
    n = kwargs.get('near')
    # far clipping plane
    f = kwargs.get('far')
    # device pixel width
    w = kwargs.get('width')
    # device pixel height
    h = kwargs.get('height')

    # Model-View-Transform-Matrix
    M = transform.view_transform(p, t, u)
    # Projection Matrix
    P = transform.pers_projection_transform(fov_y, a, n, f)
    # Viewport/Device Transform
    V = transform.viewport_transform((w, h))
    if forward:
        # world coordinate
        w_point = point
        if len(w_point) == 3:
            w_point = np.append(w_point, [1])

        # eye coordinate from world coordinate
        e = M @ w_point
        # un-normalized clip coordinate
        cn = P @ e
        # normalized clip coordinate (divide-by-w)
        c = cn / cn[-1]
        # device coordinates
        d = V @ c
        return d
    else:
        # if not forward, it becomes device_to_world func. (backward)
        d = point
        # clip coordinate
        c = np.linalg.inv(V) @ d
        # un-normalized eye coordinate
        en = np.linalg.inv(P) @ c
        # eye coordinate
        e = en / en[-1]
        # world coordinate
        w_point = np.linalg.inv(M) @ e
        return w_point


if __name__ == '__main__':
    # play around with params
    kwargs['look_at'] = (0, 0, 0)
    kwargs['eye_pos'] = (0, 40, 0)
    kwargs['up'] = (0, 0, 1)
    kwargs['near'] = 0.15
    kwargs['far'] = 100

    # compute for sample points
    w_points = [(10, 10, 10), (10, -10, 10), (-10, 10, 10)]
    d_points = []
    for point in w_points:
        d = world_to_device(point, forward=True, **kwargs)
        d_points.append(d)
        print('d: ', d)
    w_recons = []
    for point in d_points:
        wp = world_to_device(point, forward=False, **kwargs)
        w_recons.append(wp)
        print('w: ', wp)
