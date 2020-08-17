import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import os


def process():
    np.set_printoptions(suppress=True)

    model = tensorflow.keras.models.load_model(os.getcwd()+'\\model.h5')

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    image = Image.open(os.getcwd()+'\\media\\input.jpg')

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    data[0] = normalized_image_array

    prediction = model.predict(data)
    print(np.argmax(prediction[0]))

    return 'Person '+str(np.argmax(prediction[0])+1)
