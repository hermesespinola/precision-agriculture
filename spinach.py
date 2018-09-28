import cv2
import numpy as np
import os, glob, sys
from pprint import pprint
from matplotlib import pyplot as plt

# Settings to calculate real values
focal_length = 53.3
spinach_dist = 500.0 # mm
size = (600, 480)

# Asumming no distortion nor lense imperfections
def calculate_height(px_height):
    return (px_height * spinach_dist) / focal_length

isVideo = len(sys.argv) > 1 and sys.argv[1] == 'video'

roi_width = 100
floor_level = 220
floor_diff = 130
day_diff = 1785//2 if isVideo else 27
seed_radius = 5
seeds = [
    (58, 254),
    # (120, 254),
    # (194, 250),
    # (262, 252),
    # (325, 252),
    (400, 252),
    # (473, 252),
    (543, 252)
]

def process(im, day, wait_press=False, out_org=None, out_markers=None):
    im_resized = cv2.resize(im, size)
    floor = floor_level+floor_diff if day >= day_diff else floor_level

    # convert to HSV and find green areas
    green_lower = np.array([23, 80, 80], dtype=np.uint8)
    green_upper = np.array([113, 255, 255], dtype=np.uint8)
    hsv = cv2.cvtColor(im_resized, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    # Close gaps in green mask
    kernel = np.ones((2, 2),np.uint8)
    hsv_closing = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    # Find edges
    gauss = cv2.GaussianBlur(im_resized, (5, 5), 1)
    canny = cv2.Canny(gauss, 80, 250)

    # Close gaps in edges image
    canny_closing = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

    # merge both results
    clossing_with_canny = hsv_closing | canny_closing

    # Ignore all bellow the floor
    clossing_with_canny[floor:,:] = 0

    # Fill holes
    im_floodfill = clossing_with_canny.copy()
    h, w = canny.shape[:2]
    flood_mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, flood_mask, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    flooded = clossing_with_canny | im_floodfill_inv

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(flooded, cv2.MORPH_OPEN, kernel)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=2)

    # sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.2*dist_transform.max(), 255, 0)

    # unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling (add one because background is 0)
    _, markers = cv2.connectedComponents(opening)
    # mark unknown region with zero
    markers = markers + 1
    markers[unknown==255] = 0
    u8markers = np.uint8(markers)
    colored = cv2.applyColorMap(u8markers, cv2.COLORMAP_PINK)
    markers = cv2.watershed(im_resized, markers)

    # color image bounds
    contours = markers == -1
    c_w, c_h = markers.shape
    contours[0,:] = False
    contours[c_w-1,:] = False
    contours[:,0] = False
    contours[:,c_h-1] = False
    im_resized[contours] = [255, 255, 0]

    heights = np.zeros(len(seeds))
    for n_seed, seed in enumerate(seeds):
        l, r = seed[0]-roi_width//2, seed[0]+roi_width//2
        roi = contours[:,l:r]
        # find upper extreme contour
        roi_w, roi_h = roi.shape
        upper = 0
        for i in range(roi_w):
            for j in range(roi_h):
                if roi[i, j]:
                    upper = (j+seed[0]-roi_width//2, i)
                    break
            if upper:
                # Draw max value
                cv2.circle(im_resized, upper, 5, (0, 128, 128), cv2.FILLED)
                heights[n_seed] = im_resized.shape[1] / im_resized.shape[1] * (floor - upper[1])
                # Add height measure to result
                break

        # draw roi delimiters
        cv2.line(im_resized, (l, 0), (l, roi_w), (0, 205, 0))
        cv2.line(im_resized, (r, 0), (r, roi_w), (0, 205, 0))

        # draw some points of interest
        if day >= day_diff:
            seed = seed[0], seed[1] + floor_diff
        cv2.circle(im_resized, seed, seed_radius, (0, 100, 235), cv2.FILLED)

    # draw floor line
    if day >= day_diff:
        im_resized[floor_level+floor_diff] = [255, 0, 0]
    else:
        im_resized[floor_level] = [255, 0, 0]

    # Display images
    cv2.imshow('flooded without noise', opening)
    cv2.imshow('markers', colored)
    cv2.imshow('original', im_resized)
    out_org.write(im_resized)
    out_markers.write(colored)

    if wait_press:
        if cv2.waitKey(0) & 0xFF == ord('q'):
            return 1
    else:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return 1
    return heights

measures = []
if isVideo:
    cap = cv2.VideoCapture('spinach.mp4')

    # To write video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out_original = cv2.VideoWriter('result.mp4', fourcc, 20.0, size)
    out_markers = cv2.VideoWriter('markers.mp4', fourcc, 20.0, size)
    n_frame = 0
    try:
        while cap.isOpened():
            # If hit 'q' then quit
            cap.read()
            _, im = cap.read()
            spinach_heights = process(im, n_frame, False, out_original, out_markers)
            if type(spinach_heights) is int and spinach_heights is -1:
                break
            measures.append(spinach_heights)
            n_frame += 1
    finally:
        cap.release()
        out_original.release()
        out_markers.release()
else:
    os.chdir('data')
    images = sorted(glob.glob('*.jpeg'))
    for day, image in enumerate(images):
        # If hit 'q' then quit
        spinach_heights = process(cv2.imread(image), day, wait_press=True)
        if type(spinach_heights) is int and spinach_heights is -1:
            break
        measures.append([calculate_height(px_height) / 100 for px_height in spinach_heights])

# plot heights
_, ax = plt.subplots()
ax.set_color_cycle(['red', 'blue', 'green'])
plt.plot(measures)
plt.xlabel('frame' if isVideo else 'day')
plt.ylabel('height (cm)')
plt.pause(0.1)

cv2.waitKey(0)
cv2.destroyAllWindows()
