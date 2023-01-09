Here is another Python program for a pattern recognition project, this time using the k-nearest neighbors (k-NN) algorithm to classify handwritten digits:

Copy code

```
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, neighbors
from sklearn.model_selection import train_test_split

# Load the MNIST dataset
mnist = datasets.load_digits()

# Extract the features and labels
features = mnist.data
labels = mnist.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)

# Create a k-NN classifier
classifier = neighbors.KNeighborsClassifier()

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions on the test set
predictions = classifier.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = metrics.accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
```

This code is similar to the previous example, but it uses the k-NN classifier instead of an SVM. It also uses scikit-learn's train_test_split function to split the dataset into training and testing sets. The rest of the code is similar, with the classifier being trained on the training set and its performance being evaluated on the test set.
