# AgroGuard AI: Crop Disease Identification System
s
AgroGuard AI is a computer vision application designed to help farmers identify plant diseases early. Using a Deep Learning model (MobileNetV2), it classifies leaf images and provides treatment recommendations.

## 🚀 Getting Started

### 1. Installation
Clone the project and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Loading Pre-trained Weights
The current implementation initializes a MobileNetV2 model with ImageNet weights. To use your custom trained weights, modify `model.py`:
```python
# In model.py
model.load_weights('path_to_your_model_weights.h5')jjmm
```

### 3. Running the App
Start the Streamlit development server:
```bash
streamlit run app.py
```

## 📁 Dataset Structure for Training
To train the model on a dataset like 'PlantVillage', structure your folders as follows:

```text
/dataset
    /Apple___Apple_scab
        image1.jpg
        image2.jpg
    /Apple___Black_rot
        ...
    /Corn___Common_rust
        ...
    /Healthyn
        ...
```

### Data Augmentation
The `utils.py` file includes an augmentation pipeline. Use this during training to improve robustness:
- Random Horizontal/Vertical Flip
- Random Rotation (0.2)
- Random Zoom (0.2)

## 🛠️ Technology Stack
- **Frontend:** Streamlit
- **Deep Learning:** TensorFlow/Keras
- **Image Processing:** OpenCV, PIL
- **Data Handling:** NumPy

## 🛡️ Preprocessing Pipeline
Every uploaded image undergoes:
1. **Resizing:** 224x224 (std for MobileNetV2)
2. **Noise Reduction:** Gaussian Blur
3. **Normalization:** Pixel values scaled to [0, 1]
