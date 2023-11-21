import numpy as np
EAST_OUTPUT_LAYERS = ["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]
'''
Implementation of EAST Module (Efficient and Accurate Scene Text detection pipeline).
Program based on:
OCR with OpenCV, Tesseract, and Python Intro to OCR - 1st Edition (version 1.0)
Chapter 18: Rotated Text Bounding Box Localization with OpenCV
'''
def predictions(scores,geometry, minConf = 0.5):
    r,c = scores.shape[:2]
    boxes =[]
    confidences =[]
    
    for y in range(r):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        
        for x in range(c):
            score = float(scoresData[x])
            if score < minConf:
                continue
        
        offsetX = x*4
        offsetY = y*4
        
        angle = anglesData[x]
        cos,sin = np.cos(angle), np.sin(angle)
        
        curr_height = xData0[x]+xData2[x]
        curr_width = xData1[x]+xData3[x]
        
        offset = ([
            
        ])