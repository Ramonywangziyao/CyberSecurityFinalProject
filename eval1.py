from tensorflow import keras
import sys
import h5py
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import initializers
import cv2

data_filename = str(sys.argv[1])
clean_data_filename = "clean_data/clean_data.h5"
model_filename = "models/sunglasses_bd_net.h5"
# entropy threshold
ethd = 0.28
np.seterr(divide='ignore', invalid='ignore')

def data_loader(path):
    data = h5py.File(path, 'r')
    X = np.array(data['data'])
    y = np.array(data['label'])
    X = X.transpose((0,2,3,1))
    X /= 255
    return X, y

def superimpose(img1, img2):
    result = cv2.addWeighted(img1, 1, img2, 1, 0, dtype = cv2.CV_32F)
    return result

def getEntropy(img, cleanX, model, entropyRange):
    overlaid_x = [0] * entropyRange
    random_int = np.random.randint(len(cleanX),size=entropyRange)

    for i in range(entropyRange):
        overlaid_x[i] = (superimpose(img, cleanX[random_int[i]]))

    overlaid_y = model.predict(np.array(overlaid_x))
    entropySum = -np.nansum(overlaid_y * np.log2(overlaid_y))
    entropy = entropySum/entropyRange
    return entropy

# data preparation
X, y = data_loader(data_filename)
n = len(X)
cleanX, cleanY = data_loader(clean_data_filename)
entropyRange = int(len(cleanX) * 0.1)
model = keras.models.load_model(model_filename)

# processing
predicts = []
correctCount = 0
poison_cnt = 0
for i in range(n):
    if i % 20 == 0:
        print("Processing: ", i, "/", n)
    x = X[i]
    entropyX = getEntropy(x, cleanX, model, entropyRange)
    if entropyX < ethd:
      predicts.append(n + 1)
      poison_cnt += 1
    else:
      predicts.append(np.argmax(model.predict(np.array([x])), axis = 1)[0])

accu = np.mean(np.equal(predicts, y))*100
print("Classification accuracy: ",accu,"%")
print("Poison data detect: ",poison_cnt,"/",n,", ",poison_cnt/n*100,"%")
