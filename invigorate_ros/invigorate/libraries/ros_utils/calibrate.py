#!/usr/bin/env python
import cv2
import numpy as np
import getco


def calibrate(c_Image, c_Robot):
    '''
    c_Image: depth map coordinates (n,3)
    c_Robot: robot coordinates (n,3)
    '''
    c_Image = c_Image.astype(np.float32)
    c_Robot = c_Robot.astype(np.float32)
    c_Image = c_Image.reshape(-1, 1, 3)
    c_Robot = c_Robot.reshape(-1, 1, 3)
    retval, out, inliers = cv2.estimateAffine3D(c_Image, c_Robot)

    return out

def calibrate_kinect(robot_coordinate):
    rgb = 'cali_crgb'
    d = 'cali_cd'
    # kinect1.save_image(rgb, d)
    Img = cv2.imread('output/rob_result/kinectImg/' + rgb + '.jpg')
    Dep = cv2.imread('output/rob_result/kinectImg/' + d + '.png', cv2.IMREAD_UNCHANGED)
    image_coordinate = getco.Get_Image_Co(Img, Dep)
    print(robot_coordinate)
    print(image_coordinate)
    trans_matrix = calibrate(image_coordinate, robot_coordinate)
    print trans_matrix
    matfile = open('output/rob_result/' + 'trans_mat.txt', 'w+')
    for row in range(3):
        for rank in range(4):
            matfile.write(str(trans_matrix[row][rank]) + ' ')
        matfile.write('\n')

if __name__ == '__main__':
    Img = cv2.imread('crgb_1.jpg')
    Dep = cv2.imread('cdepth_1.png', cv2.IMREAD_UNCHANGED)
    image_coordinate = Getco.Get_Image_Co(Img, Dep)
    print(image_coordinate)
    robot_coordinate = np.array([[0.5029423278578329, 0.12722137423998614, -0.07489523443063247],
                                [0.5106979570010648, -0.009187424372038757, -0.07845359211769065],
                                [0.6362082691566375, 0.13554749650288614, -0.0752874743759239],
                                [0.642033826260483, -0.0032112772247053353, 0.06016039267698298]])

    trans_matrix = calibrate(image_coordinate, robot_coordinate)
    print trans_matrix
    matfile = open('trans_mat.txt', 'w+')
    for row in range(3):
       for rank in range(4):
           matfile.write(str(trans_matrix[row][rank])+' ')
       matfile.write('\n')

