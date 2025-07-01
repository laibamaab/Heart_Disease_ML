from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from data_cleaning import X, y

# SPLIT DATA INTO TRAIN AND TEST SETS
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# NITIALIZE AND TRAIN LOGISTIC REGRESSION MODEL
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)
# MAKE PREDICTIONS ON TEST SET
y_pred = model.predict(X_test)

# STEP 15: EVALUATE MODEL
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Plot heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix using LogisticRegression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# STEP 16: DISPLAY RESULTS
print("LogisticRegression Accuracy:", accuracy)
print("\nLogisticRegression Confusion Matrix:\n", conf_matrix)
print("\nLogisticRegression Classification Report:\n", report)

# INITIALIZE AND TRAIN RANDOM FOREST MODEL
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# MAKE PREDICTIONS ON TEST SET
rf_y_pred = rf_model.predict(X_test)

# EVALUATE RANDOM FOREST MODEL
rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_conf_matrix = confusion_matrix(y_test, rf_y_pred)
rf_report = classification_report(y_test, rf_y_pred)

# Plot heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(rf_conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix using RandomForest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# DISPLAY RANDOM FOREST RESULTS
print("Random Forest Accuracy:", rf_accuracy)
print("\nRandom Forest Confusion Matrix:\n", rf_conf_matrix)
print("\nRandom Forest Classification Report:\n", rf_report)


model_names = ['Logistic Regression', 'Random Forest']
accuracies = [accuracy, rf_accuracy]

plt.figure(figsize=(6, 4))
sns.barplot(x=model_names, y=accuracies)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0.0, 1.1)
plt.grid(axis='y')

for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.02, f"{acc:.2f}", ha='center', va='bottom', fontweight='bold')

plt.show()
