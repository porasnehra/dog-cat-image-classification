# 🐶🐱 Dog vs Cat Image Classifier
### Binary CNN · TensorFlow/Keras · Confidence Scores on Every Prediction

---

> A Convolutional Neural Network that classifies images as **Dog** or **Cat** — with confidence scores.  
> Trained end-to-end in TensorFlow/Keras. Visualizes predictions on a 5×2 test grid.

---

## RESULTS

| Sample | True Label | Predicted | Confidence |
|--------|-----------|-----------|------------|
| Image 1 | 🐱 Cat | 🐱 Cat | **91.3%** |
| Image 2 | 🐶 Dog | 🐶 Dog | **87.0%** |
| Image 3 | 🐶 Dog | 🐶 Dog | **92.1%** |
| Image 4 | 🐱 Cat | 🐱 Cat | **88.7%** |

---

## MODEL ARCHITECTURE

```
Input Image (RGB)
      ↓
Conv2D → ReLU → MaxPooling2D
      ↓
Conv2D → ReLU → MaxPooling2D
      ↓
Conv2D → ReLU → MaxPooling2D
      ↓
Flatten
      ↓
Dense → ReLU → Dropout
      ↓
Dense (1) → Sigmoid
      ↓
Output: Dog (>0.5) / Cat (≤0.5) + Confidence %
```

---

## STACK

```
Python · TensorFlow · Keras · NumPy · Matplotlib
```

---

## INSTALL

```bash
# Clone
git clone https://github.com/porasnehra/dog-cat-image-classification.git
cd dog-cat-image-classification

# Install dependencies
pip install tensorflow numpy matplotlib
```

---

## USAGE

### Run the notebook
```bash
jupyter notebook dog_cat_classification.ipynb
```

### Predict on a single image
```python
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("dog_cat_model.h5")

def predict_image(img_path, img_size=150):
    img = image.load_img(img_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_batch, verbose=0)[0][0]

    if prediction > 0.5:
        label = "Dog"
        confidence = prediction * 100
    else:
        label = "Cat"
        confidence = (1 - prediction) * 100

    return f"{label} ({confidence:.1f}%)"

# Example
print(predict_image("test_images/my_pet.jpg"))
# Output: Dog (92.1%)
```

### Visualize test predictions
```python
import matplotlib.pyplot as plt
import numpy as np

class_names = ['cat', 'dog']

plt.figure(figsize=(12, 12))
for images, labels in test_ds.take(1):
    for i in range(10):
        img = images[i]
        true_label = labels[i].numpy()
        img_batch = np.expand_dims(img, axis=0)
        prediction = model.predict(img_batch, verbose=0)[0][0]

        if prediction > 0.5:
            predicted_class_index = 1
            confidence = prediction * 100
        else:
            predicted_class_index = 0
            confidence = (1 - prediction) * 100

        predicted_class_name = class_names[predicted_class_index]
        true_class_name = class_names[true_label]

        plt.subplot(5, 2, i + 1)
        plt.imshow(img.numpy())
        plt.title(f"True: {true_class_name}\nPred: {predicted_class_name} ({confidence:.1f}%)")
        plt.axis("off")

plt.tight_layout()
plt.show()
```

---

## FEATURES

- **Binary classification** with sigmoid activation — clean and efficient
- **Confidence scores** — tells you not just *what* but *how sure* the model is
- **Visual test grid** — 5×2 matplotlib plot showing true vs predicted labels
- **Custom prediction function** — plug in any image path and get an instant result

---

## DATASET

Uses the classic **Dogs vs. Cats** dataset.  
Download from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data) and place in:

```
dog-cat-image-classification/
├── dataset/
│   ├── train/
│   │   ├── cat/
│   │   └── dog/
│   └── test/
│       ├── cat/
│       └── dog/
└── dog_cat_classification.ipynb
```

---

## AUTHOR

**Poras Nehra** · B.Tech CS @ MMEC, Mullana  
[GitHub](https://github.com/porasnehra) · [LinkedIn](https://linkedin.com/in/poras-nehra-142170367)

---

⭐ Star this repo if it helped you!
