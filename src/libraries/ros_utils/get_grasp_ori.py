import numpy as np
import math

def GetRotationMatrix(vec, ang):
    R = alignToVecAndLevel(vec)
    R = np.dot(R, rotAroundGivenAxis(2,ang))
    return R

def alignToVecAndLevel(vec):
    az = math.atan2(vec[0], vec[2])
    el = -math.atan2(vec[1], math.sqrt(math.pow(vec[0],2) + math.pow(vec[2],2)))
    R = np.dot(rotAroundGivenAxis(1, az), rotAroundGivenAxis(0, el))
    R = np.dot(R, rotToLevelYInXYPlane(R))
    return R

def rotToLevelYInXYPlane(R):
    xyslop = -R[0,2]/R[1,2]
    if math.fabs(xyslop) < np.Inf:
        PXY = np.array([[1, xyslop, 0]])
    else:
        PXY = np.array([[0, np.sign(xyslop), 0]])
    PXY = np.dot(R.transpose(),PXY.transpose())
    rollAng = math.atan2(PXY[1,0], PXY[0,0]) + math.pi / 2
    R = rotAroundGivenAxis(2, rollAng)
    return R

def rotAroundGivenAxis(axis, ang):
    cTh = math.cos(ang)
    sTh = math.sin(ang)
    if   axis == 0:
         R = np.array([[1.0, 0.0,  0.0],
                       [0.0, cTh, -sTh],
                       [0.0, sTh, cTh]])
    elif axis == 1:
         R = np.array([[cTh,  0.0, sTh],
                       [0.0,  1.0, 0.0],
                       [-sTh, 0.0, cTh]])
    elif axis ==2:
         R = np.array([[cTh, -sTh, 0.0],
                       [sTh,  cTh, 0.0],
                       [0.0,  0.0, 1.0]])
    else:
        print('ERROR:axis has to be one of [0, 1, 2], referring to [x, y, z]!')
        return

    return R

def RotationMatrixToQuaternion(R):

    r = R.shape[0]
    c = R.shape[1]
    if (r != 3 | c != 3 ):
     print('R must be a 3x3 matrix!')
     return
    w = math.sqrt(np.trace(R) + 1) / 2
    if (w.imag > 0):
        w = 0

    x = math.sqrt(1 + R[0,0] - R[1,1] - R[2,2]) / 2
    y = math.sqrt(1 + R[1,1] - R[0,0] - R[2,2]) / 2
    z = math.sqrt(1 + R[2,2] - R[1,1] - R[0,0]) / 2
    quater = [w, x, y, z]
    i = quater.index(max(quater))

    if (i == 0):
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)

    if (i == 1):
        w = (R[2, 1] - R[1, 2]) / (4 * x)
        y = (R[0, 1] + R[1, 0]) / (4 * x)
        z = (R[2, 0] + R[0, 2]) / (4 * x)

    if (i == 2):
        w = (R[0, 2] - R[2, 0]) / (4 * y)
        x = (R[0, 1] + R[1, 0]) / (4 * y)
        z = (R[1, 2] + R[2, 1]) / (4 * y)

    if (i == 3):
        w = (R[1, 0] - R[0, 1]) / (4 * z)
        x = (R[2, 0] + R[0, 2]) / (4 * z)
        y = (R[1, 2] + R[2, 1]) / (4 * z)

    Q = [x, y, z, w]

    return Q

def GetGraspOri(vec, ang):
    return RotationMatrixToQuaternion(GetRotationMatrix(vec, ang))

if __name__=='__main__':
    vec = [2,3,1]
    ang = -2
    ori = GetGraspOri(vec,ang)
    print (ori)
