# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

[//]: # (Image References)

[image1]: ./output.png "Output of lane detection"



### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 6 steps.

1. I converted the images to grayscale.
2. Applied a gaussian noise kernel on the image to remove traces of noise. 
3. Canny edge detection with fine-tuned thresholds
4. Extract ROI with using vertices (0,imshape[0]),(imshape[1]/2 - 50, imshape[0]/2 + 50 ), (imshape[1]/2 + 50, imshape[0]/2 +50), (imshape[1],imshape[0])
5. Use hough transform to find all the lines in the frame and use those to find/draw the two significant lines with positive and negative slopes. 
6. Place the lines detected from previous on the original image

In order to draw a single line on the left and right lanes, I modified the draw_lines() function to find the find the average positive and negative slopes of lines. Also, to reduce noise, the slopes are filtered.

The image below shows the output on test images. 
Column 1 - input image
Column 2 - Canny edge detection 
Column 3 - Lines detected from Hough transform on ROI
Column 4 - Extrapolated lines on input image

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline

The pipeline does work on two of the test videos pretty well except on the challenge video. The filtering of slopes worked really well to reduce the nosie in detection. The algorithm does not perform well under the following cases:
1. Curvy lanes
2. Lanes on comparatively white-ish roads

### 3. Suggest possible improvements to your pipeline

A possible improvement would be to use k means clustering to find the mean of two clusters. This would remove the dependency of filter coefficients to filter outliers in slopes.
