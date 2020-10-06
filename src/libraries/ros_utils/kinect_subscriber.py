#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import Image,PointCloud2
import cv2
from cv_bridge import CvBridge
import time
import sys

class kinect_reader():
  def __init__(self, cfg):
    self.img_type = cfg['imgtype']
    self.image_color = Image()
    self.image_depth = Image()
    self.isread = True
    self.bridge = CvBridge()
    self.subscriber_depth = rospy.Subscriber("/kinect2/" + self.img_type + "/image_depth_rect", Image, self.callback_depth)
    self.subscriber_color = rospy.Subscriber("/kinect2/" + self.img_type + "/image_color_rect", Image, self.callback_color)

  def callback_color(self, data):
      if self.isread:
         self.image_color = self.bridge.imgmsg_to_cv2(data, "bgr8")

  def callback_depth(self, data):
      if self.isread:
         self.image_depth = self.bridge.imgmsg_to_cv2(data, "16UC1")

  def show_image(self, option = 0):
      if option == 0:
         cv2.imshow('image_rgb', self.image_color)
         cv2.waitKey(0)
      elif option ==1:
         cv2.imshow('image_depth', self.image_depth/256)
         cv2.waitKey(0)

  def save_image(self, rgbpic_name,dpic_name):
      print('saving images to '+'kinectImg/' + rgbpic_name + '.jpg')
      cv2.imwrite('output/rob_result/kinectImg/' + rgbpic_name + '.jpg', self.image_color)
      print('saving images to '+'kinectImg/' + dpic_name + '.png')
      cv2.imwrite('output/rob_result/kinectImg/' + dpic_name + '.png', self.image_depth)

  def get_image(self):
      return self.image_color, self.image_depth

if __name__ == '__main__':
    cfg = {'imgtype' : 'hd'}
    kinect1 = kinect_reader(cfg)
    time.sleep(2)
    kinect1.save_image('1','1')
