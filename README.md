# Ageing signs detection
This project uses Convolutional Neural Networks (CNNs) to detect visible signs of aging in facial images such as wrinkles, puffy eyes, dark spots, or a clear face. It includes training a model on labeled data, performing data augmentation, and testing using Haar-cascade-based face detection and class activation predictions.

## Workflow Overview

### 1. Data Loading & Labeling

- Loads images using OpenCV from four folders.

**Labels:**
- `0 = clear face`
- `1 = dark spots`
- `2 = puffy eyes`
- `3 = wrinkles`

---

### 2. Preprocessing

- Converts RGB images to grayscale.
- Normalizes pixel values (divided by 255).
- Reshapes to match CNN input: `(244, 244, 1)`

---

### 3. Data Augmentation

Augmentation is performed using:

- Rotation
- Width/Height Shifting
- Zoom
- Shearing

---

### 4. Model Architecture

Uses **EfficientNetB0** pretrained on ImageNet.

**Model Structure:**

```python
model = Sequential([
    EfficientNetB0(weights='imagenet', include_top=False, input_shape=(244,244,3)),
    MaxPooling2D((3,3)),
    Dropout(0.5),
    Flatten(),
    Dense(4, activation="sigmoid")
])
```

### 5. Training Configuration

- **Loss**: `binary_crossentropy`  
- **Optimizer**: `Adam`  
- **Metric**: `accuracy`

---

## 🧪 Testing & Inference

The model uses **Haar Cascade** to detect faces in test images and then predicts aging signs using the trained CNN model.

### 🔍 Steps:

1. **Load test image**
2. **Detect face using Haar cascade**
3. **Resize and preprocess the face**
4. **Predict signs using the trained CNN model**
5. **Display bounding box + class labels + probabilities on the image**


