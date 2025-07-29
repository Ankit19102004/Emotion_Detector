# Emotion Detection System

A real-time emotion detection system using deep learning and computer vision. This project can detect and classify human emotions from facial expressions in real-time using a webcam.

## 🎯 Features

- **Real-time Emotion Detection**: Live emotion recognition using webcam feed
- **7 Emotion Classes**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Face Detection**: Automatic face detection using Haar Cascade
- **Deep Learning Model**: Pre-trained CNN model for emotion classification
- **Visual Feedback**: Real-time bounding boxes and emotion labels

## 📁 Project Structure

```
emotion detection/
├── README.md                 # This file
├── realtime.py              # Main application for real-time emotion detection
├── trainmodel.ipynb         # Jupyter notebook for training the model
├── emotiondetector.h5       # Pre-trained model weights
├── emotiondetector.json     # Model architecture
├── requirement.txt          # Python dependencies
├── images/                  # Dataset directory
│   ├── train/              # Training images
│   │   ├── angry/          # Angry emotion images
│   │   ├── disgust/        # Disgust emotion images
│   │   ├── fear/           # Fear emotion images
│   │   ├── happy/          # Happy emotion images
│   │   ├── neutral/        # Neutral emotion images
│   │   ├── sad/            # Sad emotion images
│   │   └── surprise/       # Surprise emotion images
│   └── test/               # Test images (same structure as train)
└── tempCodeRunnerFile.py   # Temporary file
```

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- Webcam
- Sufficient lighting for face detection

### Setup Instructions

1. **Clone or download the project**
   ```bash
   # If using git
   git clone <repository-url>
   cd emotion-detection
   ```

2. **Install required dependencies**
   ```bash
   pip install -r requirement.txt
   ```

3. **Verify installation**
   - Ensure all files are in the correct directory
   - Check that `emotiondetector.h5` and `emotiondetector.json` are present

## 📋 Dependencies

The main dependencies include:
- `opencv-python` - Computer vision and webcam handling
- `tensorflow` / `keras` - Deep learning framework
- `numpy` - Numerical computations
- `matplotlib` - Visualization (for training)
- `jupyter` - Jupyter notebook support

## 🎮 Usage

### Real-time Emotion Detection

1. **Run the main application**
   ```bash
   python realtime.py
   ```

2. **Using the application**
   - Position yourself in front of the webcam
   - Ensure good lighting for better face detection
   - The system will automatically detect faces and classify emotions
   - Press `ESC` key to exit the application

### Training Your Own Model

1. **Open the training notebook**
   ```bash
   jupyter notebook trainmodel.ipynb
   ```

2. **Follow the notebook instructions**
   - The notebook contains the complete training pipeline
   - You can modify hyperparameters and model architecture
   - Training will save the model as `emotiondetector.h5` and `emotiondetector.json`

## 🔧 How It Works

### 1. Face Detection
- Uses Haar Cascade classifier to detect faces in the video frame
- Converts the frame to grayscale for processing

### 2. Image Preprocessing
- Extracts the detected face region
- Resizes the face to 48x48 pixels (model input size)
- Normalizes pixel values to range [0, 1]

### 3. Emotion Classification
- Feeds the preprocessed image to the trained CNN model
- Model outputs probabilities for 7 emotion classes
- Selects the emotion with highest probability

### 4. Visual Output
- Draws bounding box around detected face
- Displays predicted emotion label above the face
- Shows real-time video feed with annotations

## 📊 Model Information

- **Architecture**: Convolutional Neural Network (CNN)
- **Input Size**: 48x48 grayscale images
- **Output Classes**: 7 emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)
- **Training Data**: ~28,000 images across 7 emotion categories
- **Model Files**: 
  - `emotiondetector.h5` - Model weights
  - `emotiondetector.json` - Model architecture

## 🎯 Performance Tips

1. **Lighting**: Ensure good, even lighting for better face detection
2. **Distance**: Maintain appropriate distance from the camera (1-3 feet)
3. **Face Angle**: Face the camera directly for best results
4. **Background**: Use a simple, uncluttered background
5. **Camera Quality**: Higher resolution cameras provide better results

## 🐛 Troubleshooting

### Common Issues

1. **Webcam not detected**
   - Check if webcam is connected and not in use by another application
   - Try changing the camera index in `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`

2. **Face not detected**
   - Improve lighting conditions
   - Ensure face is clearly visible and not obstructed
   - Try adjusting distance from camera

3. **Model loading errors**
   - Verify `emotiondetector.h5` and `emotiondetector.json` are in the same directory
   - Check file permissions

4. **Import errors**
   - Ensure all dependencies are installed: `pip install -r requirement.txt`
   - Check Python version compatibility

### Error Messages

- **"No module named 'cv2'"**: Install OpenCV: `pip install opencv-python`
- **"No module named 'keras'"**: Install Keras: `pip install keras tensorflow`
- **"Camera index out of range"**: Try different camera indices (0, 1, 2)

## 🔄 Customization

### Adding New Emotions
1. Collect images for the new emotion
2. Add the emotion to the labels dictionary in `realtime.py`
3. Retrain the model using `trainmodel.ipynb`

### Modifying Model Architecture
1. Edit the model architecture in `trainmodel.ipynb`
2. Retrain the model
3. Update the model files

### Changing Detection Parameters
- Adjust `detectMultiScale` parameters for different face detection sensitivity
- Modify the confidence threshold for emotion classification

## 📈 Future Enhancements

- [ ] Support for multiple faces simultaneously
- [ ] Emotion intensity measurement
- [ ] Age and gender detection
- [ ] Mobile app integration
- [ ] API for web applications
- [ ] Improved model accuracy with larger datasets

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Dataset: FER2013 or similar emotion recognition dataset
- OpenCV for computer vision capabilities
- Keras/TensorFlow for deep learning framework
- Haar Cascade for face detection

---

**Note**: This emotion detection system is for educational and research purposes. The accuracy may vary based on lighting conditions, face angles, and image quality. 