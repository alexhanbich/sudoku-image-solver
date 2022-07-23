from keras.models import load_model
import numpy as np

def predict_digits(digits):
    digits = np.array(digits)
    model = load_model('model/model.h5')
    res = model.predict(digits)
    vals = []
    for x in res:
        vals.append(x.argmax())
    return vals