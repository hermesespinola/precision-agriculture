import cv2
from scipy.spatial import distance as dist
import numpy as np
import sys, os, glob

os.chdir('data')
images = sorted(glob.glob('*.jpeg'))
rois = [cv2.selectROIs(
    'select spinaches #{}'.format(i),
    cv2.resize(cv2.imread(image), (600, 480))
    ) for i, image in enumerate([images[1], images[-1]])]
print(rois)
