import numpy as np
from get_grasp_ori import GetGraspOri
from scipy import linalg

def grasprec_mask(grec, shape):
    # 3-D mask generater
    boundlines = np.zeros([4, 2])
    for t in range(4):
        yx1 = np.expand_dims(grec[t, :], 0)
        yx2 = np.expand_dims(grec[(t + 1) % 4, :], 0)
        yx = np.concatenate((yx1, yx2), 0)
        x = np.concatenate((np.expand_dims(yx[:, 1], 1), np.ones([2, 1])), 1)
        y = yx[:, 0]
        boundlines[t, :] = np.linalg.solve(x, y)
    mask = np.zeros(shape)
    x = np.arange(shape[1]).reshape(shape[1], 1) + 1
    # x: n x 2
    x = np.concatenate((x, np.ones((shape[1], 1))), axis=1)
    # boundlines: 4 x 2
    # bound: n x 4
    bound = np.dot(x, boundlines.T)
    # y: m x 1
    y = np.arange(shape[0]).reshape(shape[0], 1) + 1
    # totally: m x n points, therefore, totally m x n x 4 bound checking
    # mask: m x n x 4
    mask = (np.expand_dims(bound, 0) > np.expand_dims(y, 1))
    mask = (mask[:, :, 0] & mask[:, :, 2]) | (mask[:, :, 1] & mask[:, :, 3]) | \
           ((1 - mask[:, :, 0]) & (1 - mask[:, :, 2])) | ((1 - mask[:, :, 1]) & (1 - mask[:, :, 3]))
    mask = 1 - mask
    if len(shape) == 2:
        return np.expand_dims(mask, 2)
    elif len(shape) == 3:
        return mask

def kinect_grasp_depth(kg, depth):
    xmin = int(np.max([np.floor(np.min(kg[:, 1])), 1]))
    xmax = int(np.min([np.ceil(np.max(kg[:, 1])), depth.shape[1]]))
    ymin = int(np.max([np.floor(np.min(kg[:, 0])), 1]))
    ymax = int(np.min([np.ceil(np.max(kg[:, 0])), depth.shape[0]]))
    temp_kg = np.copy(kg)
    temp_kg[:, 1] = kg[:, 1] - xmin
    temp_kg[:, 0] = kg[:, 0] - ymin
    h = int(ymax - ymin)
    w = int(xmax - xmin)
    kgmask = grasprec_mask(temp_kg, (h, w))
    graspdepthpatch = depth[ymin:ymax, xmin:xmax]
    xyoffset = {'x': xmin, 'y': ymin}
    return graspdepthpatch, kgmask, xyoffset

def min_ignore_zero(array):
    tarray = np.copy(array)
    infmat = ((tarray == 0) - 0.5) * np.inf
    infmat[infmat < 0] = 0
    tarray = infmat + tarray
    return np.min(tarray), np.where(tarray == np.min(tarray))

def gen_robot_xyz(kgdepth, xyoffset, transmat):
    xoffset = xyoffset['x']
    yoffset = xyoffset['y']
    # h,w,_ = kgdepth.shape
    h, w = kgdepth.shape
    robotpc = np.zeros([h, w, 3])
    pcmask = np.zeros([h, w], dtype=np.int32)
    for y in range(robotpc.shape[0]):
        for x in range(robotpc.shape[1]):
            depth = kgdepth[y, x]
            if depth > 0:
                xori = x + xoffset
                yori = y + yoffset
                robotxyz = (np.dot(transmat, np.array([[xori], [yori], [depth], [1]]))).squeeze()
                robotpc[y, x, :] = robotxyz
                pcmask[y, x] = 1
    return robotpc, pcmask

def fit_plane(points):
    # input: n x 3 points, np.array
    # output: 3-d surface normal, with z > 0

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    xy = np.sum(x * y)
    xz = np.sum(x * z)
    yz = np.sum(y * z)
    xx = np.sum(x * x)
    yy = np.sum(y * y)
    x_sum = np.sum(x)
    y_sum = np.sum(y)
    z_sum = np.sum(z)

    n_points = points.shape[0]
    a = np.array([[xx, xy, x_sum],
                  [xy, yy, y_sum],
                  [x_sum, y_sum, n_points]])
    b = np.array([xz, yz, z_sum])
    # plane: Ax + By + C = z, surf_norm = (A, B, -1)
    ABC = linalg.solve(a, b)
    surf_norm = np.array([-ABC[0], -ABC[1], 1])
    surf_norm = surf_norm / np.linalg.norm(surf_norm)
    return surf_norm

def image_grasp_to_robot(kgrec, transmat, depth):
    kgdepth, kgmask, xyoffset = kinect_grasp_depth(kgrec, depth)

    kgmask = kgmask[:, :, 0]

    mindepth, gpoints = min_ignore_zero(kgdepth * kgmask)
    print(mindepth, gpoints)
    gpoints = np.array([gpoints[0][0], gpoints[1][0]])
    grasp_patch = kgdepth[(gpoints[0] - 2): (gpoints[0] + 3),
                  (gpoints[1] - 2): (gpoints[1] + 3)]
    xyoffset['x'] += gpoints[0] - 2
    xyoffset['y'] += gpoints[1] - 2
    robotpc, pcmask = gen_robot_xyz(grasp_patch, xyoffset, transmat)
    if pcmask.sum() > 5:
        # at least 5 points are required to fit the surface.
        robotgvec = fit_plane(robotpc.reshape(-1, 3)[pcmask.reshape(-1) == 1])
        if np.abs(robotgvec[2]) > 0:
            robotgvec = -robotgvec
        robotgvec = np.array([0., 0., -1.])
    else:
        robotgvec = np.array([0., 0., -1.])

    # grasp rec angle
    gp1 = np.flip(np.mean(kgrec[0:2, :], 0), 0)
    gp2 = np.flip(np.mean(kgrec[2:4, :], 0), 0)
    gdiff = gp2 - gp1
    robotgang = np.arctan2(gdiff[1], -gdiff[0])
    # robot grasp point
    gcent = (gp1 + gp2) / 2
    robotgpoint = np.dot(transmat, np.array([gcent[0], gcent[1], mindepth, 1])).squeeze()
    print(robotgpoint)

    robotori = GetGraspOri(robotgvec, -robotgang)
    return robotgpoint, robotori, robotgvec