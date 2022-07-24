import cv2
from cv2 import WARP_INVERSE_MAP
from cv2 import BORDER_REPLICATE
import numpy as np

def get_font_size(text, shape):
    w = int(shape[1]/9*0.4)
    for scale in reversed(range(0, 60, 1)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, scale/10, 2)
        font_w = text_size[0][0]
        font_h = text_size[0][1]
        if font_w <= w:
            return scale/10, font_w, font_h 
    return 1


def insert_digits(img, s_grid, u_grid):
    c_i = 0
    s_grid = s_grid.T
    u_grid = u_grid.T
    for i in range(9):
        for j in range(9):
            if u_grid[i][j] == 0:
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = str(s_grid[i][j])
                font_size, font_w, font_h = get_font_size(text, img.shape)
                x_coor = int(round((i+0.5)/9*img.shape[1])) - font_w//2
                y_coor = int(round((j+0.5)/9*img.shape[0])) +  font_h//2
                cv2.putText(img, text, (int(x_coor), int(y_coor)), font, font_size, (255, 0, 0), 2, cv2.LINE_AA)
                c_i += 1
    return img


def merge_images(img1, img2):
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    thresh2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    result1 = cv2.bitwise_and(img1, img1, mask=thresh2)
    result2 = cv2.bitwise_and(img2, img2, mask=255-thresh2)
    result = cv2.add(result1, result2)
    return result


def undo_transform(src, dst, c_points):
    c_points = np.array(c_points)
    w, h = src.shape[1], src.shape[0]
    src_pts = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, w-1]], dtype='float32')
    ho, _ = cv2.findHomography(src_pts, c_points)
    warped = cv2.warpPerspective(src, ho, (dst.shape[1], dst.shape[0]), flags=BORDER_REPLICATE)
    cv2.fillConvexPoly(dst, c_points, 0, 16)
    dst_img = cv2.add(dst, warped)
    return dst_img