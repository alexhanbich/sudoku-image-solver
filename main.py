import cv2
from cv2 import bitwise_not
import numpy as np

def preprocess(img):
    # grayscale, blurr image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gb = cv2.GaussianBlur(gray, (9, 9), 0)
    # binarize & invert image
    binary = cv2.threshold(gb, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU )[1]
    # remove noise with morphology
    se = cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
    bg = cv2.morphologyEx(binary, cv2.MORPH_DILATE, se)
    out = cv2.divide(binary, bg, scale=255)
    return out


def extract_sudoku(img, original):
    # find 4 corners using edges a d contours
    edges = cv2.Canny(img, 100, 200, apertureSize=5)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
    max_contour = sorted_contours[0]

    # find approx points of the contour
    # repeat approx points until we have 4 corner points
    d = 0
    while True:
        d += 1
        c_points = cv2.approxPolyDP(max_contour, d, 1)
        if len(c_points) == 4:
            break
    
    # remove 1 dimension with np.squeeze
    c_points = np.squeeze(c_points)

    # cv2.drawContours(original, max_contour, -1, (0,255,0),10)
    # cv2.imshow('contour', original)
    # cv2.waitKey(0)

    # find smallest fitting rectangle
    r_rect = cv2.minAreaRect(max_contour)
    (x, y), (w, h), angle = r_rect
    r_points = cv2.boxPoints(r_rect)
    r_points = np.int0(r_points)
    # cv2.drawContours(original, [r_points], -1, (255,0,0),10)
    # cv2.imshow('rpoints', original)
    # cv2.waitKey(0)

    # sort points
    center = np.mean(c_points, axis=0)
    sorted_c_points = [
        c_points[np.where(np.logical_and(c_points[:, 0] < center[0], c_points[:, 1] < center[1]))[0][0], :],
        c_points[np.where(np.logical_and(c_points[:, 0] > center[0], c_points[:, 1] < center[1]))[0][0], :],
        c_points[np.where(np.logical_and(c_points[:, 0] > center[0], c_points[:, 1] > center[1]))[0][0], :],
        c_points[np.where(np.logical_and(c_points[:, 0] < center[0], c_points[:, 1] > center[1]))[0][0], :]
    ]

    dst_points = [[0, 0], [w, 0], [w, h], [0, h]]
    # Get the transform
    m = cv2.getPerspectiveTransform(np.float32(sorted_c_points), np.float32(dst_points))
    # Transform the image
    o_out = cv2.warpPerspective(original, m, (int(w), int(h)))
    p_out = cv2.warpPerspective(img, m, (int(w), int(h)))
    return o_out, p_out


# def clean_up(img):
#     # make image 75% from center so lines are removed
#     w = int(img.shape[1] * 0.75)
#     h = int(img.shape[0] * 0.75)
#     center_x = img.shape[1] // 2
#     center_y = img.shape[0] // 2
#     x = center_x - w//2
#     y = center_y - h//2
#     return img[y:y+h, x:x+w]

# def split_board(board):
#     cells = []
#     rows = np.array_split(board, 9)
#     for row in rows:
#         row = [x.T for x in np.array_split(row.T, 9)]
#         for cell in row:
#             cell = clean_up(cell)
#             white = cv2.countNonZero(cell)
#             print(white/cell.size, white, cell.size) 
#             cells.append(cell)
#     return cells


img = cv2.imread('sudoku.png')
pre = preprocess(img)
o_out, p_out = extract_sudoku(pre, img)

cv2.imwrite('out.png', p_out)



def extract_digits(img):
    digits = {}
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
    dst = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # calculate cell size so that we do not add anything that is bigger than cell size
    cell_size = img.shape[0]*img.shape[1]//81
    for i in range(0, cnt):
        (x, y, w, h, area) = stats[i]
        if area < 20:
            continue
        # calculate x_coor & y_coor from centroid
        x_coor = int((centroids[i][0]/len(img)*9-0.5).round())
        y_coor = int((centroids[i][1]/len(img)*9-0.5).round())

        cv2.rectangle(dst, (x, y, w, h), (0, 255, 255))
        digit = img[y:y+h, x:x+w]
        digit_size = digit.shape[0]*digit.shape[1]
        print()
        # if img is bigger than cellsize, its not a digit
        if digit_size <= cell_size:
            top = (img.shape[0]//9 - digit.shape[0]) // 2
            bottom = (img.shape[0]//9 - digit.shape[0]) // 2
            left = (img.shape[1]//9 - digit.shape[1]) // 2
            right = (img.shape[1]//9 - digit.shape[1]) // 2
            print(top, bottom, left, right)
            digit = cv2.copyMakeBorder(digit, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 0)
            digits[(x_coor, y_coor)] = digit
    cv2.imwrite('digit.png', dst)
    return digits

digits = extract_digits(p_out)
print(len(digits))
sudoku_digits = np.zeros((9,9), dtype=int)
print(sudoku_digits)
for i in range(9):
    for j in range(9):
        key = (i,j)
        if key in digits.keys():
            sudoku_digits[i][j] = 1
            cv2.imwrite(f'out/digit{i}{j}.png', digits[key])
print(sudoku_digits)