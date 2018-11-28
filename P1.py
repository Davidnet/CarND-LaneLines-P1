from __future__ import division, print_function
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

import os
import math

REGION_SELECTOR_CONFIGS = {"left_bottom":   [0, 539], "right_bottom": [900, 539], "apex": [475, 250]}



def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
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
    # breakpoint()
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
    # breakpoint()

    # y_size = img.shape[0]
    original_image = img.copy()
    y_size = int(img.shape[0])
    x_size = img.shape[1]

    #min_y = y_size * (3/5)
    min_y = int(y_size * (3/5))

    def _return_slope(x0, y0, x1, y1):
        return ((y1 - y0)/(x1 - x0))

    right_lines, left_lines = [], []

    for line in lines:
        # breakpoint()
        line_slope = _return_slope(*line.T)
        target = left_lines if line_slope > 0 else right_lines
        target.append(line)

    if (not left_lines) or (not right_lines):
        return img

    def _break_into_x_y(lines):
        x_coordinates = []
        y_coordinate = []
        for line in lines:
            for x0, y0, x1, y1 in line:
                x_coordinates.extend([x0, x1])
                y_coordinate.extend([y0, y1])

        return x_coordinates, y_coordinate

    right_x_coordinates, right_y_coordinates = _break_into_x_y(right_lines)
    left_x_coordinates, left_y_coordinates = _break_into_x_y(left_lines)

    line_right = np.poly1d(np.polyfit(right_y_coordinates, right_x_coordinates, deg=1))
    line_left = np.poly1d(np.polyfit(left_y_coordinates, left_x_coordinates, deg=1))

    # Regression evaluation
    left_x0_coordinate = int(line_left(y_size))
    left_x1_coordinate = int(line_left(min_y))


    right_x0_coordinate = int(line_right(y_size))
    right_x1_coordinate = int(line_right(min_y))


    # breakpoint()
    cv2.line(img, (left_x0_coordinate, y_size), (left_x1_coordinate, min_y), color, thickness)
    cv2.line(img, (right_x0_coordinate, y_size), (right_x1_coordinate, min_y), color, thickness)

    weighted_img(img, original_image)




    return None

    
    
    # for line in lines:
    #     for x1,y1,x2,y2 in line:
    #         cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

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


def image_pipeline(image_path):
    image = mpimg.imread(image_path)
    processed_image = image.copy()
    processed_image = grayscale(processed_image)
    # processed_image = cv2.medianBlur(processed_image, 5)
    processed_image = gaussian_blur(processed_image, 5)
    processed_image = cv2.Canny(processed_image, 120, 220)
    vertices = np.array([[ cartesian_point for  cartesian_point in REGION_SELECTOR_CONFIGS.values() ]], dtype=np.int32)
    processed_image = region_of_interest(processed_image, vertices)
    # processed_image = hough_lines(processed_image, 2, 1, 15, 40, 20)
    processed_image = hough_lines(processed_image, 2, np.pi/180, 90, 40, 20)
    processed_image = weighted_img(processed_image, image, )
    return processed_image


if __name__ == "__main__": 

    #reading in an image
    image = mpimg.imread('test_images/solidWhiteRight.jpg')

    #printing out some stats and plotting
    print('This image is:', type(image), 'with dimensions:', image.shape)
    plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

    TEST_IMAGES = "test_images"
    test_imagepaths = os.scandir(TEST_IMAGES)
    prototype_path = next(test_imagepaths)
    original_image = mpimg.imread(prototype_path.path)
    image_prototype = image_pipeline(prototype_path.path)

    plt.imshow(image_prototype, cmap='gray')
    
    plt.show()










