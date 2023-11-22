from east import predictions
import cv2
import pytesseract
import numpy as np


def detect_localized_text(image,east,nms=0.4,padding = 0):
    origH,origW = image.shape[:2]
    newW,newH = 640,640
    rW,rH = origW/float(newW),origH/float(newH)
    net = cv2.dnn.readNet(east)
    
    blob = cv2.dnn.blobFromImage(image, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]
    
    scores, geometry = net.forward(layerNames)
    rects, confidences = predictions(scores, geometry)
    
    idxs = cv2.dnn.NMSBoxesRotated(rects,scores= confidences,score_threshold=0.6,nms_threshold=nms)

    results=[]
    words=[]
    for i in idxs.flatten():
        box = cv2.boxPoints(rects[i])
        box[:, 0] *= rW
        box[:, 1] *= rH
        box = np.intp(box)
        
        (x, y, w, h) = cv2.boundingRect(box)
        
        #Enable padding for obtaining better results
        dX = int(w*padding)
        dY = int(h*padding)
        
        startX = max(0, x - dX)
        startY = max(0, y - dY)
        endX=min(origW,x+w+(dX*2))
        endY=min(origH,y+h+(dY*2))
        
        paddedROI = image[startY:endY, startX:endX]
        
        options = "--psm 7"
        text = pytesseract.image_to_string(paddedROI, config=options)
        
        words.append(text.strip())
        results.append((box, text))
        
    results = sorted(results, key=lambda x: x[0][0][0])
    for (box, text) in results:
        #print("{}\n".format(text))
        output = image.copy()
        cv2.polylines(output, [box], True, (0, 255, 0), 2)
        
        (x, y, w, h) = cv2.boundingRect(box)
        cv2.putText(output, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX,1.2, (0, 0, 255), 3)
        
    return output,words

def main():
    # file = 'Project/Sample/kohli.jpeg'
    # im = cv2.imread(file)
    # img = detect_localized_text(im,east='Project/frozen_east_text_detection.pb')
    # print(img[1])
    # cv2.imshow('Text Detection',img)
    # cv2.waitKey(0)
    pass

if __name__ == '__main__':
    main()