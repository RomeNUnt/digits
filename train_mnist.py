import numpy as np
import cv2
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# === STEP 1: LOAD MNIST DATASET ===
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# NOTE: No filtering — use all digits (0–9)

# Reshape for CNN input
X_train = X_train.reshape(-1,28,28,1) / 255.0
X_test  = X_test.reshape(-1,28,28,1) / 255.0

# Convert labels to categorical (10 classes: 0–9)
y_train_cat = to_categorical(y_train, num_classes=10)
y_test_cat  = to_categorical(y_test, num_classes=10)

# === STEP 2: BUILD CNN MODEL ===
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 10 classes: 0–9
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === STEP 3: TRAIN MODEL ===
model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=5, batch_size=32)

# === STEP 3.5: SAVE THE MODEL ===
model.save("digit_identify.keras")

# === STEP 4: TEST ON IMAGE ===
def predict_digit(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image not found!")
        return
    # Resize to 28x28
    img_resized = cv2.resize(img, (28,28))
    # Invert colors if background is white
    if np.mean(img_resized) > 127:
        img_resized = 255 - img_resized
    # Normalize
    img_norm = img_resized / 255.0
    img_input = img_norm.reshape(1,28,28,1)
    
    # Optional: Show processed image for debugging
    plt.imshow(img_input[0,:,:,0], cmap="gray")
    plt.title("Processed Input")
    plt.show()

    # Prediction
    pred = model.predict(img_input)
    digit = np.argmax(pred)  # 0–9 directly
    print(f"Predicted digit: {digit}")

# Example usage:
predict_digit(r'C:\Users\ASUS\Desktop\promethean\ws\regression models\no_4_1.png')