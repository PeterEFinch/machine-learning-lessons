##################################################
# We include the libraries
##################################################

# Imports matplot lib
# pip3 install -U matplotlib
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
# pip3 install -U scikit-learn
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split

# Imports tensorflow which is a library for doing machine learnings
# pip3 install tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Imports numpy which is useful for arrays
# pip3 install -U numpy
import numpy as np

print("starting")

##################################################
# The data must be prepared
##################################################

# Loading the digest from the datbase
digits = datasets.load_digits()

# Showing the training data
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

fig.suptitle("Training Data")

# Coverts the data to something that can be trained
n_samples = len(digits.images)
x_data = digits.images.reshape((n_samples, -1))
y_data = np.zeros((n_samples, 10))
for i, val in enumerate(digits.target):
    y_data[i][val] = 1

# Split data into 50% train and 50% test subsets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.5, shuffle=False)

plt.show()

print("finished")