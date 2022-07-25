import cv2
from insert import insert_digits
from preprocess import dilate_image, threshold_image
from extract import find_sudoku, extract_digits
from predict import predict_digits
from solver.main import SolveSudoku
import numpy as np
  
# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    _, frame = vid.read()
    o_frame = frame.copy()
    # binary: for OCR, dilated: for grid recognition
    bn = threshold_image(frame)             
    dl = dilate_image(bn)
    o_out, p_out, M = find_sudoku(dl, o_frame)
    if o_out is None:
        continue
    bn2 = threshold_image(o_out)
    digits, coors = extract_digits(bn2)
    vals = predict_digits(digits)
    solver = SolveSudoku(vals, coors)
    unsolved_grid = solver.get_grid().copy()
    solver.solve()
    solved_grid = solver.get_grid()
    num_zero = solved_grid.size - np.count_nonzero(solved_grid)
    if num_zero > 0:
        continue
    o_out = insert_digits(o_out, solved_grid, unsolved_grid)
    next_warp = cv2.warpPerspective(o_out, M, (o_frame.shape[1], o_frame.shape[0]), flags=cv2.WARP_INVERSE_MAP)  
    cv2.imshow('frame', frame)
      

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()