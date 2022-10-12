#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 03:32:26 2022

The class of hand detector based on mediapipe

@author: Xinyang Chen
"""

import mediapipe as mp

class HandDetector():
    def __init__(self, wid=640, hei=360, mode=False, maxHands=1, model_complexity=1, detectionCon=0.3, trackCon=0.3):
        self.mode = mode# hand tracking mode
        self.maxHands = maxHands# number of hands
        self.model_complexity = model_complexity# complexity of model
        self.detectionCon = detectionCon# detection confidence(palm detector)
        self.trackCon = trackCon# tracking confidence(landmark model)

        self.mpHands = mp.solutions.hands# hand tracking solution
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils# draw
        self.wid = wid # width of the input image
        self.hei = hei # height of the input image
        self.results = None

    def findHands(self, img):
        """
        @brief: get the hand tracing based on mediapipe
        param img: the input image
        param drawOn: draw tracing result on drawOn
        param draw: enable drawing

        return success: the status of tracing
        return drawOn: output after drawing
        """
        self.results = self.hands.process(img)
        
        return self.results.multi_hand_landmarks
    
    def findPosition(self):
        """
        @brief: get the hand tracing based on mediapipe
        param wid: the width of the input image
        param hei: the height of the input image

        return lmList: the list of hand landmarks, elements are id, x position and y position
        """
        
        lmList = [] # list of the landmarks
        

        
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handlms.landmark):
                    # x and y are normalized to [0.0, 1.0] by the image width and height respectively. 
                    lmList.append([id, lm.x * self.wid, lm.y * self.hei])

                    
        return lmList