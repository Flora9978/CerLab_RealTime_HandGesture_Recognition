#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 03:32:26 2022

The mask generator class

@author: Xinyang Chen
"""

import cv2 as cv
import numpy as np

class MaskGenerator():
    def __init__(self, lowBound, highBound):
        self.lowBound = lowBound # the low boundary of color mask
        self.highBound = highBound # the high boundary of color mask
        
        self.colorMask = np.zeros([360, 640], dtype = bool)


    def getColorMask(self, lab):
        trim = cv.inRange(lab, self.lowBound, self.highBound)            
        check = np.zeros([trim.shape[0], trim.shape[1], 1])
        check[:,:,0]= trim[:,:]
        self.colorMask = np.all(check == 255,axis=-1)

    