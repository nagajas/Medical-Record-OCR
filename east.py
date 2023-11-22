import numpy as np
import numpy as np
import cv2
import matplotlib.pyplot as plt

EAST_OUTPUT_LAYERS = ["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]
'''
    Implementation of EAST Module (Efficient and Accurate Scene Text detection pipeline):
        east.py serves as helper program to find bounding boxes to extract ROI (regions of interest)
    Program based on:
        OCR with OpenCV, Tesseract, and Python Intro to OCR - 1st Edition (version 1.0)
        Chapter 18: Rotated Text Bounding Box Localization with OpenCV
'''
def predictions(scores,geometry, minConf = 0.5):
    r,c = scores.shape[:2]
    boxes =[]
    confidences =[]
    
    for y in range(r):
        # Create temporary box; coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        
        for x in range(c):
            # grab the confidence score for the current detection for each column and continue if there is micConf
            score = float(scoresData[x])

            if score < minConf:
                continue
        #print(score)
        offsetX = x*4
        offsetY = y*4
        
        #Compute the boundary rectangle using angle and offset
        angle = anglesData[x]
        cos,sin = np.cos(angle), np.sin(angle)
        
        curr_height = xData0[x]+xData2[x]
        curr_width = xData1[x]+xData3[x]
        # compute the offset factor as our resulting feature
        offset = ([
            offsetX + (cos * xData1[x]) + (sin * xData2[x]),
            offsetY - (sin * xData1[x]) + (cos * xData2[x])
            ])
        
        topL= (-sin*curr_height+offset[0],-cos*curr_height + offset[1])
        botR= (-cos*curr_width+offset[0],sin*curr_width+ offset[1])
        
        cX = 0.5 * (topL[0] + botR[0])
        cY = 0.5 * (topL[1] + botR[1])
        box = ((cX, cY), (curr_width, curr_height), -1 * angle * 180.0 / np.pi)
        
        #Store the rectangle and confidence
        boxes.append(box)
        confidences.append(score)
    #print(boxes,confidences)
    return (boxes, confidences)

def localize(image,east,confidence=0.1,nms=0.3):
    origH,origW = image.shape[:2]
    newW, newH = 640,640
    rW,rH = origW/float(newW),origH/float(newH)
    net = cv2.dnn.readNet(east)
    
    blob = cv2.dnn.blobFromImage(image, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    
    scores, geometry = net.forward(EAST_OUTPUT_LAYERS)
    print(scores)
    rects, confidences = predictions(scores, geometry)
   
    bboxes = [(
    int(coord[0][0]), 
    int(coord[0][1]), 
    int(coord[1][0]),
    int(coord[1][1])  
) for coord in rects]
    
    print(confidences)
    idxs = cv2.dnn.NMSBoxesRotated(rects,scores= confidences,score_threshold=confidence,nms_threshold=nms)

    if len(idxs) > 0:

        for i in idxs.flatten():
            box = cv2.boxPoints(rects[i])
            box[:, 0] *= rW
            box[:, 1] *= rH
            box = np.int0(box)
            
            cv2.polylines(image, [box], True, (0, 255, 0), 2)
    
    return image

def main():
    file = "Project/Walmart.jpeg"
    img = cv2.imread(file)
    img2 = localize(img,east = "Project/frozen_east_text_detection.pb")
    # cv2.imshow('Text Detection',img2)
    # cv2.waitKey(0)
    
if __name__ == "__main__":
    main()