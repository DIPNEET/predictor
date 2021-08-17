import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

# load the data
x = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")["labels"]

# Classifying the data
classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I","J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "W", "X", "Y","Z"]
numClasses = len(classes)

# Training, testing, and scaling the data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state = 9, train_size = 7500, test_size = 2500)
xtrainScale = xtrain/255.0
xtestScale = xtest/255.0

# Fitting the data
cls = LogisticRegression(solver = "saga", multi_class = "multinomial").fit(xtrainScale, ytrain)
 
# Function to predict the image
def getPrediction():
    img = Image.open()
    #Converting it into BLack and White img
    bw = img.convert('L')
    # Resizing
    r = bw.resize((28,28), Image.ANTIALIAS)
    # Pixel Filtering
    pf = 20
    # Scaling
    minp = np.percentile(r, pf)
    scale = np.clip(r - minp, 0, 255)
    maxp = np.max(r)
    scale = np.asarray(scale)/maxp
    # taking the sample and reshaping
    sample = np.array(scale).reshape(1, 784)
    #Predicted
    pred = cls.predict(sample)
    return pred[0]