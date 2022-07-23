import cv2
import numpy as np

def find_max_contour(img):
    edges = cv2.Canny(img, 100, 200, apertureSize=5)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
    return sorted_contours[0]


def find_corners(contour):
    # find approx points of the contour
    # repeat approx points until we have 4 corner points
    c_points =None
    d = 0
    while True:
        d += 1
        c_points = cv2.approxPolyDP(contour, d, 1)
        if len(c_points) == 4:
            break
    c_points = np.squeeze(c_points)
    # sort points
    center = np.mean(c_points, axis=0)
    sorted_c_points = [
        c_points[np.where(np.logical_and(c_points[:, 0] < center[0], c_points[:, 1] < center[1]))[0][0], :],
        c_points[np.where(np.logical_and(c_points[:, 0] > center[0], c_points[:, 1] < center[1]))[0][0], :],
        c_points[np.where(np.logical_and(c_points[:, 0] > center[0], c_points[:, 1] > center[1]))[0][0], :],
        c_points[np.where(np.logical_and(c_points[:, 0] < center[0], c_points[:, 1] > center[1]))[0][0], :]
    ]
    return sorted_c_points


def transform(src_points, dst_points, original, preprocessed, w, h):
    # Get the transform
    m = cv2.getPerspectiveTransform(np.float32(src_points), np.float32(dst_points))
    # Transform the image
    o_out = cv2.warpPerspective(original, m, (int(w), int(h)))
    p_out = cv2.warpPerspective(preprocessed, m, (int(w), int(h)))
    return o_out, p_out


def extract_sudoku(preprocessed, original):
    grid = find_max_contour(preprocessed)
    c_points = find_corners(grid)
    # find width and height of grid
    (_, _), (w, h), _ = cv2.minAreaRect(grid)
    dst_points = [[0, 0], [w, 0], [w, h], [0, h]]
    o_out, p_out = transform(c_points, dst_points, original, preprocessed, w, h)
    return o_out, p_out


def preprocess_digit(img):
    pre = cv2.resize(img, (28, 28))
    pre = pre.reshape(28, 28, 1)
    return pre/255


def extract_digits(img):
    digits = {}
    # get all connected components from image
    cnt, _, stats, centroids = cv2.connectedComponentsWithStats(img)
    # create dst image to work with
    # dst = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # calculate cell size so that we do not add anything that is bigger than cell size
    cell_size = img.shape[0]*img.shape[1]//81
    # loop through connected components
    digits = []
    coors = []
    for i in range(0, cnt):
        (x, y, w, h, area) = stats[i]
        # if too small, ignore
        if area < 20:
            continue
        # calculate x_coor & y_coor from centroid
        x_coor = int((centroids[i][0]/img.shape[1]*9-0.5).round())
        y_coor = int((centroids[i][1]/img.shape[0]*9-0.5).round())
        # cv2.rectangle(dst, (x, y, w, h), (0, 255, 255))
        # crop digit
        digit = img[y:y+h, x:x+w]
        digit_size = digit.shape[0]*digit.shape[1]
        # if img is bigger than cellsize, its not a digit
        if digit_size <= cell_size:
            # pad image
            top = (img.shape[0]//9 - digit.shape[0]) // 2
            bottom = (img.shape[0]//9 - digit.shape[0]) // 2
            left = (img.shape[1]//9 - digit.shape[1]) // 2
            right = (img.shape[1]//9 - digit.shape[1]) // 2
            digit = cv2.copyMakeBorder(digit, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 0)
            digits.append(preprocess_digit(digit))
            coors.append((x_coor, y_coor))
    # cv2.imwrite('digit.png', dst)
    return digits, coors