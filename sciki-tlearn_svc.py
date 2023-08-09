# Imports matplot lib
# pip3 install -U matplotlib
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
# pip3 install -U scikit-learn
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

print("starting")

# Loading the digest from the datbase
digits = datasets.load_digits()

# Showing the training data
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

fig.suptitle("Training Data")

# Flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Split data into 50% train and 50% test subsets
x_train, x_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# Learn the digits on the train subset
clf.fit(x_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(x_test)

# Showing the predictions
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, x_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

fig.suptitle("Predictions")

# Success rate
success = 0
for i in range(len(predicted)):
    if predicted[i] == y_test[i]:
        success += 1

print(f"Correctly predicted {success} out of {len(predicted)}")

# Confusion matrix
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

# plt.show()

print("finished")