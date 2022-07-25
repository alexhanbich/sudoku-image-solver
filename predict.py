from tensorflow import keras
# from keras.models import load_model
import numpy as np
import cv2

def predict_digits(digits):
    digits = np.array(digits)
    # for i in range(digits.size):
    #     cv2.imwrite(f'out/digit{i}.png', digits[i])
    model = keras.models.load_model('model/model.h5')
    res = model.predict(digits)
    vals = []
    for x in res:
        vals.append(x.argmax())
    return vals