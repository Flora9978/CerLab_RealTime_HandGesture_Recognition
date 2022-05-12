#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 03:32:26 2022

The class of glove solver to preprocess input

@author: Xinyang Chen
"""

import cv2 as cv
import numpy as np
from glove_utils.OpenCVMaskGenerator import MaskGenerator

class GloveSolver():
    def __init__(self, lowBound=np.array([0, 127, 127]), highBound=np.array([255, 137, 137]),
                buffer=np.array([0, 20, 20]), sampleSize=40, img=None, lab=None):
        # check definition of lowBound, highBound, sampleSize and segmentBuffer in MaskGenerator class
        self.maskGenerator = MaskGenerator(lowBound, highBound)
        
        self.sample_size = sampleSize # the size of each sample circle in fine sampling
        self.buffer = buffer # the shifting amount in LAB color space
        self.bound_state = False # the status of shifing amount (self.buffer[0]) initialization
        
        self.img = img # the image on present frame
        self.lab = lab # the image in LAB color space
        
    def setImage(self, img):
        """
        @brief: set img
        param img: the input image
        """
        self.img = img.copy()
        
    def imgBlur(self):
        """
        @brief: blur the img
        """
        kernel = (9,9)
        self.img = cv.blur(self.img, kernel)
        
    def setLAB(self):
        """
        @brief: get lab by convert img into LAB color space
        """
        self.lab = cv.cvtColor(self.img, cv.COLOR_BGR2LAB)
    
    
    def colorChange(self, labSample, LArea, ABArea, amount):   
        """
        @brief: change color inside LAB color space with buffer
        param labSample: the image sample in LAB color space
        param area: the area whose A and B need to be changed
        param amount: the shifting amount in lab color space
        
        return labSample: the image sample after color shifting
        """
        labSample[LArea, 0] += amount[0]
        labSample[ABArea, 1] += amount[1]
        labSample[ABArea, 2] += amount[2]
        
        return labSample

    def bufferFineAdjust(self, landmarks, lab, mask):
        """
        @brief: readjust shifting amount of L by fine sampling
        param landmarks: the hand landmarks corresponding to lab
        param lab: the input of image in LAB color space
        param mask: a mask to calibrate the fine sampled area
        """
        sample_area = np.zeros([lab.shape[1], lab.shape[0]], dtype = bool)
        # generate some circle based on the position of hand landmarks
        for landmark in landmarks:
            temp = np.ones([lab.shape[1], lab.shape[0]], dtype = bool)
                    
            temp[int(landmark[1]) + self.sample_size:, :] = False
            temp[:int(landmark[1]) - self.sample_size, :] = False
            temp[:, int(landmark[2]) + self.sample_size:] = False
            temp[:, :int(landmark[2]) - self.sample_size] = False

            sample_area = sample_area | temp
        
        sample_area = mask & sample_area.transpose()

        self.buffer[0] =  255 - np.max(lab[sample_area, 0])


    def bufferInitialize(self, ABArea, detector):
        """
        @brief: readjust shifting amount of L by fine sampling
        param area: the sample area for buffer initialization
        param detector: a mediapipe hand detector
        """
        MAX_UINT8 = 255
        stepSize = 5
        LArea = np.ones_like(ABArea)
        # generate a copy for present lab color
        labCopy = self.lab.copy()
        
        # initialize shifting amount of l with maximum inside candidate area
        self.buffer[0] =  255 - np.max(labCopy[ABArea, 0])

        # run loop to search the lowest succeed shifting amount
        while True:
            newLab = self.colorChange(labCopy.copy(), LArea, ABArea, self.buffer)
            newImg = cv.cvtColor(newLab, cv.COLOR_LAB2BGR) 
            success = detector.findHands(newImg)
            tmpPositionList = detector.findPosition()
            if success:
                self.bufferFineAdjust(tmpPositionList, labCopy, ABArea)
                self.bound_state = True
                break
            if self.buffer[0] >= MAX_UINT8:
                # shift back when reach 255, fail and break
                break
            self.buffer[0] += stepSize
    
    def frameInitialize(self, img):
        self.setImage(img)
        self.imgBlur()
        self.setLAB()     
        self.maskGenerator.getColorMask(self.lab)  
    
    def baseTrial(self, detector):
        LArea = np.ones_like(self.maskGenerator.colorMask)
        newLab = self.colorChange(self.lab.copy(), LArea, self.maskGenerator.colorMask, self.buffer)
        newImg = cv.cvtColor(newLab, cv.COLOR_LAB2BGR) 
        self.success_detect = detector.findHands(newImg)
    

    
    def solver(self, img, detector):
        """
        @brief: preprocess the image and trace the glove
        param img: input raw image
        param detector: a mediapipe hand detector
        
        return img: visualization result of glove solver
        """
        self.frameInitialize(img)
        
        # before initialization succeed, keep try initialization 
        if not self.bound_state:
            self.bufferInitialize(self.maskGenerator.colorMask, detector)            
        
        # if succeed, preprocess the image and tracing the hand
        else:
            self.baseTrial(detector)   
