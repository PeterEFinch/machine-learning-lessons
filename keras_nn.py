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

##################################################
# We create and train model
##################################################

# Define the keras model
model = Sequential()
model.add(Dense(50, input_shape=(64,), activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse']) # Try different loss functions binary_crossentropy or mse

# Learn the digits on the train subset
model.fit(x_train, y_train)

##################################################
# We analyse the results
##################################################

# Predicts the out array for the test datas
y_predicted = model.predict(x_test) 

# Converts the data to something that can be analysed
# For each image the output node with bigest value is found
predicted = y_predicted.argmax(axis=1) 
actual = y_test.argmax(axis=1) 

# Printing the data to see what it looks like
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(y_predicted[0], predicted[0])
print(y_test[0], actual[0])

# Success rate
success = 0
for i in range(len(predicted)):
    if predicted[i] == actual[i]:
        success += 1

print(f"Correctly predicted {success} out of {len(predicted)}")

# Confusion matrix
disp = metrics.ConfusionMatrixDisplay.from_predictions(actual, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")


plt.show()

print("finished")