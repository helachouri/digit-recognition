from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
import PIL.ImageOps 
from collections import Counter
import argparse as ap

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-i", "--image", help="Path to Image", required="True")
args = vars(parser.parse_args())

# load JSON and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# function to convert image to MNIST format
def imageprepare(argv):
    img = Image.open(argv).convert('L')
    width = float(img.size[0])
    height = float(img.size[1])
    new_image = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
        # resize and sharpen
        img = img.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        new_image.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
        # resize and sharpen
        img = img.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        new_image.paste(img, (wleft, 4))  # paste resized image on white canvas
        new_image = PIL.ImageOps.invert(new_image)

    tv = list(new_image.getdata())  # get pixel values
    pixels = np.array(tv).reshape(1,784)
    return pixels

# load weights into new model
loaded_model.load_weights('model.h5')
print('Loaded model from disk')

# convert image to MNIST format
test = imageprepare(args['image'])

# normalize data from 0-255 to 0-1
test = test/255

# make and print prediction
prediction = loaded_model.predict_classes(test)
prediction = Counter(prediction).most_common(1)[0][0]
print ("Prediction is: ", prediction)
