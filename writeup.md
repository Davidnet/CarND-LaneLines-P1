# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline is divided in 3 parts, the first part **preprocessing** consist on getting the image to be read as a numpy array, and working on a copy of the image. It also consists on modifying the image into grayscale and doing a smooth filter to reduce the noise of the image signal.

the **processing** stage, consists on running the standard canny algorithm on the greyscale image, and then create a `region of interest` on where to find the pathlines,when I designed my pipeline, I decided to have the region of interest before running the canny detector, but there were lines on the the vertices lines, finally I run the hugh lines detector to get the lines that are included on the area defined by `REGION_SELECTOR_CONFIGS`.

In this setting, I modify the `draw_lines` function, the method to get the lines is the following: I calculate the slopes defined by each pair of x,y points and separate into left and right parts with the condition of positive and negative slope, I still do not know why there are certain image frames that do no return either lines on the right hand side or the left hand side. After that I separate the x and y coordinates for each line in left and right side and creare a linear polynomial that is the line that is going to be drawn and after evaluate that line at the bottom and horizon points of the image.


### 2. Identify potential shortcomings with your current pipeline

I still have frames that do not have either left or right lines, and it clearly very depandant on having images that have a steady camera, there are many parameters to set up, that I am worried that I have chosen wrong all the parameters.

### 3. Suggest possible improvements to your pipeline

A possible improvement would be to have a more robust procedure to choose parameters for each one the functions defined in the notebook.
