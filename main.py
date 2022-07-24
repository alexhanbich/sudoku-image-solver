from unicodedata import digit
import cv2
from cv2 import threshold
from insert import insert_digits
from preprocess import dilate_image, threshold_image
from extract import find_sudoku, extract_digits
from predict import predict_digits
from solver.main import SolveSudoku
import numpy as np

img = cv2.imread('resources/sudoku-rotate.jpg')
img2 = img.copy()


# binary: for OCR, dilated: for grid recognition
bin = threshold_image(img)             
dl = dilate_image(bin)
o_out, p_out, M = find_sudoku(dl, img2)

bin2 = threshold_image(o_out)



digits, coors = extract_digits(p_out)

for i in range(len(digits)):
    cv2.imwrite(f'out/digit{i}.png', digits[i]) 


vals = predict_digits(digits)



solver = SolveSudoku(vals, coors)
unsolved_grid = solver.get_grid().copy()
solver.solve()
solved_grid = solver.get_grid()

print(unsolved_grid)
print(solved_grid)

o_out = insert_digits(o_out, solved_grid, unsolved_grid)

next_warp = cv2.warpPerspective(o_out, M, (img2.shape[1], img2.shape[0]), flags=cv2.WARP_INVERSE_MAP)
cv2.imshow('next', next_warp)
cv2.waitKey(0)



