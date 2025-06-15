# Step 1: Import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Step 2: Load the Iris dataset
iris = load_iris()

# Step 3: Create a DataFrame for better readability
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['species'] = iris.target  # Add species (target) as a column

# Step 4: Check for missing values
print("Missing values:\n", data.isnull().sum())  # Should be 0 for all

# Step 5: Split data into input (X) and output (y)
X = data[iris.feature_names]   # Features (input)
y = data['species']            # Labels (output)

# Step 6: Split data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Step 7: Create and train a Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)  # Train the model

# Step 8: Make predictions
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')  # Use macro for multi-class
recall = recall_score(y_test, y_pred, average='macro')

# Step 10: Print the evaluation results
print("\nModel Evaluation:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
