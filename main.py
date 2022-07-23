from unicodedata import digit
import cv2
from preprocess import preprocess
from extract import extract_sudoku, extract_digits
from predict import predict_digits
from solver.main import SolveSudoku
from insert import insert_digits
import numpy as np

img = cv2.imread('resources/sudoku.png')
pre = preprocess(img)
o_out, p_out = extract_sudoku(pre, img)
digits, coors = extract_digits(p_out)
vals = predict_digits(digits)
solver = SolveSudoku(vals, coors)
unsolved_grid = solver.get_grid().copy()
solver.solve()
solved_grid = solver.get_grid()
print(unsolved_grid)
print(solved_grid)
mask = blank_image = np.zeros(o_out.shape, np.uint8)
solved = insert_digits(mask, solved_grid, unsolved_grid)
cv2.imshow('solved', solved)
cv2.waitKey(0)