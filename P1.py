#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
#     return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def houghspace_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
#     points = cv2.HoughLines(img, rho, theta, threshold, np.array([]))
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    lines = two_significant_lines(img, lines)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, thickness = 5)
    return line_img

def two_significant_lines(img, lines):
    negative_slope_lines = []
    positive_slope_lines = []
    significant_lines = [] 
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        m,b = np.polyfit((x1, x2), (y1, y2), 1)
        
        if(m<0):
            negative_slope_lines.append([m,b])
        else:
            positive_slope_lines.append([m,b])
    if(len(negative_slope_lines)):
        m_negative_slope_lines,b_negative_slope_lines = np.average(negative_slope_lines, axis=0)
#     print(np.mean(negative_slope_lines[:][0]))
    if(len(positive_slope_lines)):
        m_positive_slope_lines,b_positive_slope_lines =  np.average(positive_slope_lines, axis=0)
    
#     significant_lines.append(getLinefromRhoTheta(rho_negative_slope_lines,theta_negative_slope_lines))
#     significant_lines.append(getLinefromRhoTheta(rho_positive_slope_lines,theta_positive_slope_lines))
    significant_lines.append(getLine(img, m_negative_slope_lines,b_negative_slope_lines))
    significant_lines.append(getLine(img, m_positive_slope_lines,b_positive_slope_lines))
    return significant_lines


def getLine(img, m, b):
#     print("m,b",m,b)
    y1 = img.shape[0]

    y2 = int(y1-300)

    x1 = int((y1-b)/m)

    x2 = int((y2-b)/m)

    return [[x1,y1,x2,y2]]
            
# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

import os
image_list = os.listdir("test_images/")

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 1 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 1     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 10 #minimum number of pixels making up a line
max_line_gap = 1    # maximum gap in pixels between connectable line segments



def on_trackbar(val):
    global image_list
    image_name = "whiteCarLaneSwitch.jpg"
    img = cv2.imread(str("test_images/"+image_name))
    imshape = img.shape
    img_gray = grayscale(img)
    img_blur = gaussian_blur(img_gray, 7)
    img_canny = canny(img_blur,65,val)
    roi_vertices = np.array([[(0,imshape[0]),(imshape[1]/2 - 50, imshape[0]/2 + 50 ), (imshape[1]/2 + 50, imshape[0]/2 +50), (imshape[1],imshape[0])]], dtype=np.int32)
    img_roi = region_of_interest(img_canny, roi_vertices)

    line_image = houghspace_lines(img_roi, rho, theta, threshold, min_line_length, max_line_gap)

    line_image = region_of_interest(line_image, roi_vertices)

    out_image = weighted_img(line_image, img)
    cv2.imshow("Window", out_image)


cv2.namedWindow("Window")
cv2.createTrackbar("Trackbar","Window",65,255,on_trackbar)

# Show some stuff
on_trackbar(1)
# Wait until user press some key
cv2.waitKey()

def process_image(img):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    line_image = np.copy(img)*0 # creating a blank to draw lines on
    imshape = img.shape
    img_gray = grayscale(img)
    img_blur = gaussian_blur(img_gray, 3)
    img_canny = canny(img_blur,50,150)
    roi_vertices = np.array([[(0,imshape[0]),(imshape[1]/2 - 50, imshape[0]/2 + 50 ), (imshape[1]/2 + 50, imshape[0]/2 +50), (imshape[1],imshape[0])]], dtype=np.int32)
    img_roi = region_of_interest(img_canny, roi_vertices)

    line_image = houghspace_lines(img_roi, rho, theta, threshold, min_line_length, max_line_gap)
    
    line_image = region_of_interest(line_image, roi_vertices)
    
    out_image = weighted_img(line_image, img)
    # cv2.imshow("output",out_image)
    return out_image

