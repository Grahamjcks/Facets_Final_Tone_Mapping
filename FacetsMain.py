from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import os
def loadExposureSeq(path):
    images = []
    times = []
    with open(os.path.join(path, 'list.txt')) as f:
        content = f.readlines()
    for line in content:
        tokens = line.split()
        images.append(cv.imread(os.path.join(path, tokens[0])))
        times.append(1 / float(tokens[1]))
    return images, np.asarray(times, dtype=np.float32)

images, times = loadExposureSeq('./exposureseq')

calibrateDebevec = cv.createCalibrateDebevec()
responseDebevec = calibrateDebevec.process(images, times)

# Merge images into an HDR linear image
mergeDebevec = cv.createMergeDebevec()
hdrDebevec = mergeDebevec.process(images, times, responseDebevec)
# Save HDR image.
cv.imwrite("hdrDebevec.hdr", hdrDebevec)

# Tonemap using Mantiuk's method obtain 24-bit color image
tonemapDurand = cv.createTonemapMantiuk(1.8,0.9,0.9)
ldrMantiuk = tonemapDurand.process(hdrDebevec)
ldrMantiuk = 3 * ldrMantiuk
cv.imwrite("ldr-Mantiuk.jpg", ldrMantiuk * 255)

# Tonemap using Drago's method to obtain 24-bit color image
tonemapDrago = cv.createTonemapDrago(1, 0.7, 0.75)
ldrDrago = tonemapDrago.process(hdrDebevec)
ldrDrago = 3 * ldrDrago
cv.imwrite("ldr-Drago.jpg", ldrDrago * 255)

# Tonemap using Reinhard's method to obtain 24-bit color image
tonemapReinhard = cv.createTonemapReinhard(1.5, 0,0,0)
ldrReinhard = tonemapReinhard.process(hdrDebevec)
cv.imwrite("ldr-Reinhard.jpg", ldrReinhard * 255)
