import cv2
import numpy as np
from functools import reduce
import math
import operator


def sort_points(coords):
    coords = sorted(coords, key=lambda x:x[0])
    l, r = coords[:2], coords[2:]
    l = sorted(l, key=lambda x:x[1])
    r = sorted(r, key=lambda x:x[1])
    return np.array([l[0], r[0], r[1], l[1]])


def find_corners(contour, img):
    es = cv2.arcLength(contour, True)*0.04
    approx = cv2.approxPolyDP(contour, es, True)
    coords = np.squeeze(approx)
    coords = sort_points(coords)
    for coor in coords:
        x, y = coor[0], coor[1]
        # image = cv2.circle(img, (x,y), 0, (255,0,0), 6)
    return coords


def find_sudoku(img, ori):
    w_ori = img.shape[1]
    h_ori = img.shape[0]
    thresh_area = (w_ori*h_ori)/3
    edges = cv2.Canny(img, 100, 200, apertureSize=5)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    i = 0
    c_list = []
    for c in contours:
        _,_,w,h = cv2.boundingRect(c)
        if w*h < thresh_area:
            continue
        if abs((w-h)/h) > 0.15:
            continue
        es = cv2.arcLength(c, True)*0.04
        approx = cv2.approxPolyDP(c, es, True)
        if len(approx != 4):
            continue
        c_list.append(c)
    if len(c_list) == 0:
        return None, None, None
    
    max_area = cv2.contourArea(c_list[0])
    max_contour = c_list[0]
    for i in range(1, len(c_list)):
        if cv2.contourArea(c_list[i]) > max_area:
            max_contour = c_list[i]
    
    (_, _), (w, h), _ = cv2.minAreaRect(max_contour)
    src_pts = find_corners(max_contour, ori.copy())
    dst_points = [[0, 0], [w, 0], [w, 
    h], [0, h]]
    M = cv2.getPerspectiveTransform(np.float32(src_pts), np.float32(dst_points))
    # Transform the image
    o_out = cv2.warpPerspective(ori, M, (int(w), int(h)))
    p_out = cv2.warpPerspective(img, M, (int(w), int(h)))
    return o_out, p_out, M


def preprocess_digit(img):

    pre = cv2.resize(img, (28, 28))
    pre = pre.reshape(28, 28, 1)
    return pre


def extract_digits(img):
    # get all connected components from image
    cnt, _, stats, centroids = cv2.connectedComponentsWithStats(img)
    # create dst image to work with
    # dst = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # calculate cell size so that we do not add anything that is bigger than cell size
    cell_area = img.shape[0]*img.shape[1]//81
    # loop through connected components
    digits = []
    coors = []
    for i in range(0, cnt):
        (x, y, w, h, area) = stats[i]
        # if too small, ignore
        if w*h < cell_area*0.05:
            continue
        # calculate x_coor & y_coor from centroid
        x_coor = int((centroids[i][0]/img.shape[1]*9-0.5).round())
        y_coor = int((centroids[i][1]/img.shape[0]*9-0.5).round())
        # cv2.rectangle(dst, (x, y, w, h), (0, 255, 255))
        # crop digit
        digit = img[y:y+h, x:x+w]
        digit_area = digit.shape[0]*digit.shape[1]*0.7
        if digit_area > cell_area:
            continue
        # if img is bigger than cellsize, its not a digit
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