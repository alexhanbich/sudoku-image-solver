import cv2

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