# Load camera calibration and distortion coefficients
import pickle

dist_pickle = pickle.load(open("./cal_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Helpful functions

def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    if orient == 'x':
        sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)

    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    abs_sobely = np.absolute(sobely)
    abs_sobelx = np.absolute(sobelx)
    dir = np.arctan2(abs_sobely, abs_sobelx)

    dir_binary = np.zeros_like(dir)
    dir_binary[(dir > thresh[0]) & (dir < thresh[1])] = 1
    return dir_binary

def binary(img, s_thresh=(140, 255), ksize = 3):

    # Create binary image by applying colour and gradient thresholding
    img = np.copy(img)
    R = img[:,:,0]
    # Convert to HSV color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    S = hls[:,:,2]

    # Gradient
    gradx = abs_sobel_thresh(R, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(R, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(R, sobel_kernel=ksize, thresh=(0.7, 1.3))
    gradient = np.zeros_like(dir_binary)
    gradient[(gradx == 1) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    # Threshold color channel
    s_binary = np.zeros_like(S)
    s_binary[(S >= s_thresh[0]) & (S <= s_thresh[1])] = 1

    # Combine binaries
    combined_binary = np.zeros_like(gradient)
    combined_binary[(s_binary == 1) | (gradient == 1)] = 1

    return combined_binary

# For accelerating the speed, I creat four global variables
# to store the infor of lane positions. Thus, exclude the first
# time, *pipeline()* doesn't need to search the windows positions each time.
# Create empty lists to receive left and right lane
left_lane_inds = []
right_lane_inds = []
left_fit = []
right_fit = []

def pipeline(image):
    # Undistort image
    preprocess = cv2.undistort(image, mtx, dist, None, mtx)
    # Combined colour thresholding and gradient thresholding
    preprocess = binary(preprocess,s_thresh=(80, 255), ksize = 11)
    # Transform image to a bird-eye perspective
    # Calculate image size
    img_size = (preprocess.shape[1], preprocess.shape[0])
    # Source and destination location
    src = np.float32(
        [[(img_size[0] / 2) - 46, (img_size[1] / 2) + 90],
        [((img_size[0] / 6) - 20), (img_size[1])],
        [(img_size[0] * 5 / 6) + 20, img_size[1]],
        [(img_size[0] / 2 + 46), img_size[1] / 2 + 90]])
    dst = np.float32(
        [[(img_size[0] / 4), 0],
        [(img_size[0] / 4), img_size[1]],
        [(img_size[0] * 3 / 4), img_size[1]],
        [(img_size[0] * 3 / 4), 0]])
    # Paramenters to transform and inverse perspective
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Perspective transform
    warped = cv2.warpPerspective(preprocess, M, img_size, flags=cv2.INTER_NEAREST)
    # Claim global variables
    ## an individual class seems a better choice to store values
    global left_lane_inds
    global right_lane_inds
    global left_fit
    global right_fit
    # Set the width of the windows +/- margin
    margin = 60
    # Choose the number of sliding windows
    nwindows = 9
    # Set minimum number of pixels found to recenter window
    minpix = 100

    ## If there is no previous lane positions infro, search
    ## windows position based on pixel peak of histogram
    if len(left_lane_inds) == 0:
        histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = (np.dstack((warped, warped, warped))*255.999).astype(np.uint8)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        # Set height of windows
        window_height = np.int(warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = warped.shape[0] - (window+1)*window_height
            win_y_high = warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
    else: ## the previous position of windows exist
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Find the bottom of y by search the maxim of y
    y_eval = np.max(ploty)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # Calculate the car position
    ## Assumption: the camera is in the centre of the car
    camera_centre = (left_fitx[-1] + right_fitx[-1])/2
    centre_diff = (camera_centre-warped.shape[1]/2)*xm_per_pix
    side = 'left'
    if centre_diff <= 0:
        side = 'right'

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, img_size)
    # Combine the result with the original image
    output = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    # Draw radius and position texts on the image
    cv2.putText(output, 'Radius of Left Curvature = '+str(round(left_curverad, 3))+'(m)', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255),2)
    cv2.putText(output, 'Radius of Right Curvature = '+str(round(right_curverad, 3))+'(m)', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255),2)
    cv2.putText(output, 'Vehicle is '+str(abs(round(centre_diff, 3)))+'m '+side+' of centre',(50,150), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255),2)

    return output

# Output process the project video and save a lan_founded video
from moviepy.editor import VideoFileClip

output = 'P4_Advanced_Lane_Finding.mp4'
clip1 = VideoFileClip("project_video.mp4")
clip = clip1.fl_image(pipeline)
clip.write_videofile(output, audio=False)
