import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pytesseract
#import tensorflow as tf

def convert_to_portrait(image):
    height, width, _ = image.shape

    if width > height:
        transposed_image = cv2.transpose(image)
        portrait_image = cv2.flip(transposed_image, 1)
        return portrait_image
    else:
        return image


def resize_image(image,height=800):
    if (image.shape[0] > height):
        rat = height / image.shape[0]
        image_temp =cv2.resize(image, (int(rat * image.shape[1]), height))
    else:
        image_temp = image
    return image_temp


def edge_detection(image,mn=200,mx=250,height=800):
    image_temp = resize_image(image)
    image_gray = cv2.cvtColor(image_temp, cv2.COLOR_BGR2GRAY)
    
    image_blur = cv2.bilateralFilter(image_gray, 9, 75, 75)
    
    image_th = cv2.adaptiveThreshold(image_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)
    #plt.imshow(image_th,cmap ='gray')
    
    image_med = cv2.medianBlur(image_th, 11)
    
    image_border = cv2.copyMakeBorder(image_med, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    #plt.imshow(image_border, 'gray')
    
    return cv2.Canny(image_border, mn, mx)
    
    
def get_contour(image):
    binary_image = edge_detection(image)
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))
    print(contours)
    contour_img = cv2.drawContours(resize_image(image),contours,-1,(0,255,0),thickness=2,lineType=cv2.LINE_AA)
    return contour_img


def four_corners_sort(pts):
    diff = np.diff(pts, axis=1)
    summ = pts.sum(axis=1)
    return np.array([pts[np.argmin(summ)],
                     pts[np.argmax(diff)],
                     pts[np.argmax(summ)],
                     pts[np.argmin(diff)]])


def contour_offset(cnt, offset):
    cnt += offset
    cnt[cnt < 0] = 0
    return cnt


def find_page_contours(edges, img):
    # Getting contours  
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Finding biggest rectangle otherwise return original corners
    height = edges.shape[0]
    width = edges.shape[1]
    MIN_COUNTOUR_AREA = height * width * 0.5
    MAX_COUNTOUR_AREA = (width - 10) * (height - 10)

    max_area = MIN_COUNTOUR_AREA
    page_contour = np.array([[0, 0],
                            [0, height-5],
                            [width-5, height-5],
                            [width-5, 0]])

    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)

        # Page has 4 corners and it is convex
        if (len(approx) == 4 and
                cv2.isContourConvex(approx) and
                max_area < cv2.contourArea(approx) < MAX_COUNTOUR_AREA):
            
            max_area = cv2.contourArea(approx)
            page_contour = approx[:, 0]

    # Sort corners and offset them
    page_contour = four_corners_sort(page_contour)
    return contour_offset(page_contour, (-5, -5))


def persp_transform(img, s_points):
    """ Transform perspective from start points to target points """
    # Euclidean distance - calculate maximum height and width
    height = max(np.linalg.norm(s_points[0] - s_points[1]),
                 np.linalg.norm(s_points[2] - s_points[3]))
    width = max(np.linalg.norm(s_points[1] - s_points[2]),
                 np.linalg.norm(s_points[3] - s_points[0]))
    
    # Create target points
    t_points = np.array([[0, 0],
                        [0, height],
                        [width, height],
                        [width, 0]], np.float32)
    
    # getPerspectiveTransform() needs float32
    if s_points.dtype != np.float32:
        s_points = s_points.astype(np.float32)
    
    M = cv2.getPerspectiveTransform(s_points, t_points) 
    return cv2.warpPerspective(img, M, (int(width), int(height)))
    

def cleaned_image(file):
    image = cv2.imread(file)
    image = convert_to_portrait(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    small = resize_image(image)
    
    edges_image = edge_detection(small, 200, 250)
    edges_image = cv2.morphologyEx(edges_image, cv2.MORPH_CLOSE, np.ones((5, 11)))
    
    page_contour = find_page_contours(edges_image, small)
    page_contour = page_contour.dot(image.shape[0]/small.shape[0])
    
    newImage = persp_transform(image, page_contour)
    
    return newImage

def main():
    pass
if __name__ == '__main__':
    main()