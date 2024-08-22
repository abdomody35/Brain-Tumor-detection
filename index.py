import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt

# Function to load images from a folder
def load_images_from_folder(folder):
    images = []
    labels = []
    for label, subfolder in enumerate(['no', 'yes']):
        path = os.path.join(folder, subfolder)
        for filename in os.listdir(path):
            img_path = os.path.join(path, filename)
            try:
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img = img.resize((64, 64))  # Resize to a fixed size (64x64 in this case)
                img_array = np.array(img).flatten()  # Flatten the image
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return np.array(images), np.array(labels)

# Load the dataset
data_folder = './brain_tumor_dataset'
X, y = load_images_from_folder(data_folder)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=90)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Data loading and preprocessing complete.")

# Initialize the SVM model
svm_model = SVC(kernel='rbf')

# Train the SVM model
svm_model.fit(X_train, y_train)

print("Model initialization and training complete.")

# Save the trained SVM model
model_filename = 'brain_tumor_detection_svm_model.sav'
joblib.dump(svm_model, model_filename)

print(f"Trained SVM model saved to {model_filename}")

# Make predictions using the model
y_pred = svm_model.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Data Visualization

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred)

plt.figure()
plt.plot(recall, precision, lw=2, color='blue')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Misclassified Images
misclassified_indices = np.where(y_test != y_pred)[0]

for index in misclassified_indices:
    plt.imshow(X_test[index].reshape(64, 64), cmap='gray')
    plt.title(f"True Label: {y_test[index]}, Predicted: {y_pred[index]}")
    plt.show()
