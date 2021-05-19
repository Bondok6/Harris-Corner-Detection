import numpy as np
import cv2 as cv

img = cv.imread('chess_board.png')
img = cv.resize(img, (400, 400))
cv.imshow('Input', img)

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_gray = np.float32(img_gray)

# (img, blockSize, k-size, k)
# img -> must be gray and float32
# block size -> size of Window (corner detection field size to be considered)
# k size -> Aperture parameter of sobel derivative used
# k -> Harris corner detection of free parameters in the equation, the value for the parameter [0,04,0.06].
R = cv.cornerHarris(img_gray, 2, 3, 0.04)

# to enhance subsequent expansion code is marked clear image corners optional accuracy may be commented (NOt Important )
R = cv.dilate(R, None)

# Threshold for an optimal value, it may vary depending on the image. (interest point)
img[R > 0.01 * R.max()] = [0, 0, 255]

cv.imshow('Output', img)

if cv.waitKey(0):
    cv.destroyAllWindows()
