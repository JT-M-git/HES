import keras
import numpy as np
from PIL import Image
from itertools import groupby
image = Image.open("calc1.jpg").convert("L")
height = image.size[1]
width = image.size[0]
ratio = width / height
width1 = int(ratio * 28)
height1 = 28
image1 = image.resize((width1, height1))
arrimage1 = np.array(image1)
invertedimage = 255 - arrimage1
finalimage = invertedimage / 255.0
m = finalimage.any(0)
out = [finalimage[:, [*g]] for k, g in groupby(np.arange(len(m)), lambda x: m[x] != 0) if k]
num_of_elements = len(out)
elements_list = []
for x in range(0, num_of_elements):
    img = out[x]
    width = img.shape[1]
    filler = (finalimage.shape[0] - width) / 2
    if filler.is_integer() == False:
        filler_l = int(filler)
        filler_r = int(filler) + 1
    else:
        filler_l = int(filler)
        filler_r = int(filler)
    leftfiller = np.zeros((finalimage.shape[0], filler_l))
    rightfiller = np.zeros((finalimage.shape[0], filler_r))
    concate = np.concatenate((leftfiller, img), axis=1)
    elementsarray = np.concatenate((concate, rightfiller), axis=1)
    elementsarray.resize(28, 28, 1)
    elements_list.append(elementsarray)

elements_array = np.array(elements_list)
elements_array = elements_array.reshape(-1, 28, 28, 1)
model = keras.models.load_model("pro.h5")
elements_pred = model.predict(elements_array)
elements_pred = np.argmax(elements_pred, axis=1)
print(elements_pred)

parsestring = ''
for i in range(len(elements_pred)):
    if (elements_pred[i] == 10):
        parsestring = parsestring + '/'
    if (elements_pred[i] == 11):
        parsestring = parsestring + '+'
    if (elements_pred[i] == 12):
        parsestring = parsestring + '-'
    if (elements_pred[i] == 13):
        parsestring = parsestring + '*'
    if (elements_pred[i] == 0):
        parsestring = parsestring + '0'
    if (elements_pred[i] == 1):
        parsestring = parsestring + '1'
    if (elements_pred[i] == 2):
        parsestring = parsestring + '2'
    if (elements_pred[i] == 3):
        parsestring = parsestring + '3'
    if (elements_pred[i] == 4):
        parsestring = parsestring + '4'
    if (elements_pred[i] == 5):
        parsestring = parsestring + '5'
    if (elements_pred[i] == 6):
        parsestring = parsestring + '6'
    if (elements_pred[i] == 7):
        parsestring = parsestring + '7'
    if (elements_pred[i] == 8):
        parsestring = parsestring + '8'
    if (elements_pred[i] == 9):
        parsestring = parsestring + '9'

print(parsestring)
print(eval(parsestring))
