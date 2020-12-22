from tensorflow import keras
import sys
import h5py
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import initializers
import cv2

data_filename = str(sys.argv[1])
clean_data_filename = str(sys.argv[2])
model_filename = "models/sunglasses_bd_net.h5"
# entropy threshold
ethd = 0.3
np.seterr(divide = 'ignore')

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
entropyRange = int(len(cleanX) * 0.001)
model = keras.models.load_model(model_filename)

# processing
predicts = []
correctCount = 0
for i in range(n):
    if i % 20 == 0:
        print("Processing: ", i, "/", n)
    x = X[i]
    entropyX = getEntropy(x, cleanX, model, entropyRange)
    label = n + 1 if entropyX < ethd else np.argmax(model.predict(np.array([x])), axis = 1)[0]
    if label == int(y[i]) or n+1:
        correctCount += 1
    predicts.append(label)

print("Accuracy: ", round(correctCount / n * 100, 2), "%")
