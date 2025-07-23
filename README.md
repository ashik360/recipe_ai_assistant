
# Visual Recipe Assistant

The **Visual Recipe Assistant** is a Flutter-based mobile application that uses machine learning to recognize ingredients from images and suggests relevant recipes.

## 📱 Features

- **Ingredient Detection**: Upload or capture a photo of an ingredient and identify it with confidence level.
- **Recipe Suggestions**: Get tailored recipe recommendations based on the detected ingredient.
- **Image Input**: Choose an image from the gallery or take one using the camera.
- **On-device ML**: Uses TensorFlow Lite for fast and private inference (Works on Offline).

## 🧰 Tech Stack

- **Flutter (Dart)** – UI framework for building cross-platform apps
- **TensorFlow Lite** – for running machine learning model on-device
- **image_picker** – plugin to access camera and gallery
- **tflite_flutter** – to run TFLite model
- **tflite_flutter_helper_plus** – for image preprocessing and output processing

## 🚀 Getting Started

### Prerequisites

- Flutter SDK installed
- Android Studio or VS Code
- A physical device or emulator

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ashik360/visual_recipe_assistant.git
   cd visual_recipe_assistant
   ```

2. Get dependencies:
   ```bash
   flutter pub get
   ```

3. Run the app:
   ```bash
   flutter run
   ```

### Assets

- Place your TensorFlow Lite model file in `assets/model.tflite`
- Place your label file in `assets/labels.txt`
- Add to `pubspec.yaml`:
   ```yaml
   assets:
     - assets/model.tflite
     - assets/labels.txt
   ```

## 📄 License

This project is licensed under the MIT License.
