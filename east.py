import numpy as np
import numpy as np
import cv2
import matplotlib.pyplot as plt

'''
    Implementation of EAST Module (Efficient and Accurate Scene Text detection pipeline):
        east.py serves as helper program to find bounding boxes to extract ROI (regions of interest)
    Program based on:
        OCR with OpenCV, Tesseract, and Python Intro to OCR - 1st Edition (version 1.0)
        Chapter 18: Rotated Text Bounding Box Localization with OpenCV
'''
def near32(x):
    '''
        To compute nearest multiple of 32 of x
    '''
    return int(32 * round(float(x) / 32))

def predictions(scores,geometry, minConf = 0.3):
    r,c = scores.shape[2:4]
    boxes =[]
    confidences =[]
    
    for y in range(r):
        # Create temporary box (= coordinates that surround text)
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        
        for x in range(c):
            # Grab the confidence score for the current detection for each column and continue if there is micConf
            score = float(scoresData[x])
            if score < minConf:
                continue
            #print(score)
            offsetX = x*4
            offsetY = y*4
            
            # Compute the boundary rectangle using angle and offset
            angle = anglesData[x]
            cos,sin = np.cos(angle), np.sin(angle)
            
            curr_height = xData0[x]+xData2[x]
            curr_width = xData1[x]+xData3[x]
            # Compute the offset factor as our resulting feature
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

def localize(image,east,nms=0.3):
    origH,origW = image.shape[:2]
    #newW, newH = near32(origW),near32(origH)
    newW,newH = 320,320
    rW,rH = origW/float(newW),origH/float(newH)
    net = cv2.dnn.readNet(east)
    
    blob = cv2.dnn.blobFromImage(image, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]
    
    scores, geometry = net.forward(layerNames)
    rects, confidences = predictions(scores, geometry)
   
    bboxes = [(
    int(coord[0][0]), 
    int(coord[0][1]), 
    int(coord[1][0]),
    int(coord[1][1])  
) for coord in rects]
    

    #print(confidences)
    idxs = cv2.dnn.NMSBoxesRotated(rects,scores= confidences,score_threshold=0.6,nms_threshold=nms)

    #print(idxs)
    if len(idxs) > 0:

        for i in idxs.flatten():
            box = cv2.boxPoints(rects[i])
            box[:, 0] *= rW
            box[:, 1] *= rH
            box = np.intp(box)
            
            t=int(0.005*origH)
            cv2.polylines(image, [box], True, (0, 255, 0), t if t else 2)
    
    return image

def main():
    pass
    
if __name__ == "__main__":
    main()