import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam
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
                img = img.resize((64, 64))  # Resize to a fixed size (64x64)
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

# Reshape data for RNN (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], 64, 64))
X_test = X_test.reshape((X_test.shape[0], 64, 64))

print("Data loading and preprocessing complete.")

# Initialize the RNN model
rnn_model = Sequential([
    SimpleRNN(64, activation='relu', input_shape=(64, 64), return_sequences=True),
    Dropout(0.3),
    SimpleRNN(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
rnn_model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])

# Train the RNN model
history = rnn_model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=32, 
    validation_split=0.2,
    verbose=1
)

print("Model initialization and training complete.")

# Save the trained RNN model
model_filename = 'brain_tumor_detection_rnn_model.h5'
rnn_model.save(model_filename)

# Save the scaler
scaler_filename = 'brain_tumor_scaler_rnn.joblib'
joblib.dump(scaler, scaler_filename)

print(f"Trained RNN model saved to {model_filename}")
print(f"Scaler saved to {scaler_filename}")

# Make predictions using the model
y_pred = (rnn_model.predict(X_test) > 0.5).astype(int).flatten()

# Evaluate the model
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Data Visualization

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve and AUC
y_pred_proba = rnn_model.predict(X_test).flatten()
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
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
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

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