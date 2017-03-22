## Writeup Template



**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

**Project files**

Main files:

* `cam_cal.py` `cal_pickle.p` - contain cameral calibration code and data
* `P4_Advanced_Finding_Lane.py` - main file includes `pipeline()`
* `./output_images/` - folder contains the processed images
* `P4_Advanced_Lane_Finding.mp4` - processed video
* `Writeup.md`

Supplementary file:

* `P4_test_images.ipynb` - iPython notebook to show the process step by step

[//]: # (Image References)

[image1]: ./output_images/01.calibration.jpg "Undistorted"
[image2]: ./output_images/02.undistortion.jpg "Road Transformed"
[image3]: ./output_images/03.binary.jpg "Binary Example"
[image4]: ./output_images/04.perspective_transform.jpg "Warp Example"
[image5]: ./output_images/05.draw_lane_warped.jpg "Fit Visual"
[image6]: ./output_images/06.result.jpg "Output"
[video1]: ./P4_Advanced_Lane_Finding.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup
You're reading it!
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the separated file called `cam_cal.py`, and the result is saved in the `cal_pickle.p`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the _**camera calibration**_ and _**distortion coefficients**_ using the `cv2.calibrateCamera()` function.  I applied this distortion correction to one chessboard image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
As applying to chessboard image, I used `cv2.undistort()` to demonstrate the distortion correction to one test images (at line 89 in `P4_Advanced_Finding_Lane.py`) like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I built 4 gradients thresholded functions to help execute pipeline  (at lines 15 through 50 in `P4_Advanced_Finding_Lane.py`). For achieving a lane-line highlighted image, I build a `binary()` function to combined colour and gradient thresholds to generate a binary image ( at lines 52 through 76 and line 91 in `P4_Advanced_Finding_Lane.py`). The details of this step like these:

1. Separate R channel from RGB image.
2. Based on R channel image, combine 'x' directed Sobel, magnitude and direction gradients to highlight the lines (white lines) in the picture.
3. Convert image to HLS format and separate S channel.
4. Apply thresholding to highlight the lines (yellow) in the S channel image.
5. Combine both gradients and S colour thresholded to create a thresholded binary image.

Here's an example of my output for this step:
![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
The code for my perspective transform appears in lines 93 through 110 in the file `P4_Advanced_Finding_Lane.py`. I use `cv2.warpPerspective` function, as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 46, img_size[1] / 2 + 90],
    [((img_size[0] / 6) - 20), img_size[1]],
    [(img_size[0] * 5 / 6) + 20, img_size[1]],
    [(img_size[0] / 2 + 46), img_size[1] / 2 + 90]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 590, 467      | 320, 0        |
| 205, 700      | 320, 720      |
| 1082, 700     | 960, 720      |
| 722, 467      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a undistorted straight-line test image and its warped counterpart to verify that the lines appear parallel in the warped image.
![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
I chose minimal pixels_ to sliding the window. For a new video or image, I firstly searched the maximal pixels in the histogram of the image to ensure the start positions of the lines, and use `margin` of the windows to decide the positions of sliding windows (at lines 52 through 76 and line 195 in `P4_Advanced_Finding_Lane.py`). If the number of finding over minimal pixels, the position of next window will be set to the centre of founded pixels.

Then applying `np.polyfit()` to fit the pixels positions with a polynomial.  

If there is existed positions of **hot** pixels and the polynomials, the next positions could be found based those previous data. The output of this step like this:
![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
I did this in lines 197 through 216 in my code in `P4_Advanced_Finding_Lane.py`. I used course references and formula to calculate both side lanes curvatures and then transformed unit from pixels to the meters.

_Assuming the camera is settled at the centre of the car_, the car's position of the image is calculated by length of the image divided by 2. The road's centre is in the middle of the two lanes (two lanes' intersecting points with X). The difference of the car position and the road's centre is the car's relative position respect to road centre.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
I implemented  lines 87 through 240 in my code in `P4_Advanced_Finding_Lane.py` in the function `pipeline()`.  Here is an example of my result on a test image:
![alt text][image6]

---

### Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./P4_Advanced_Lane_Finding.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?


As the performance of the video, there are two visible wobbly lines at 24s and 39s. Also, at some frame, the difference between right and left curvatures is too big to increase the risk of the car to turn sharply (even though it looks good by human's eyes, the car actually can only read the **number**~). These problems might cause by bad-performanced binary thresholded because main of these frames are located in slow bright/dark ratio parts of the road. It implied that to use tune colour thresholding may improve the recognition result of the lane-lines. There is a discussion that a student used a max-min ratio of the colour channel to adjust to the different environment brightness. Covolution technique deserves to try to find lanes. I'll try those techniques later.

Also, to use global variable is always a dangerous and low-efficiency method in the code. I plan to use _class_ to substitute it in my future improvement.

Lastly, to challenge videos are challenging. I'll always to face challenge~
