# Scribble to Digital Converter

An AI-powered web application that converts hand-drawn sketches into clean digital images using image processing and machine learning.

## Project Overview

**Scribble to Digital Converter** is a full-stack Flask application that lets users upload hand-drawn sketches or draw on a digital canvas, then processes them through OpenCV-based image processing and a scikit-learn shape classifier to generate clean digital outputs.

## Technology Stack

- **Frontend:** HTML5, CSS3, JavaScript, Bootstrap 5
- **Backend:** Python, Flask
- **Machine Learning:** OpenCV, NumPy, Pillow, scikit-learn (RandomForest shape classifier)
- **Database:** SQLite

## Project Structure

```
scribble-to-digital/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── templates/                  # Jinja2 HTML templates
│   ├── base.html
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── upload.html
│   ├── draw.html
│   ├── convert.html
│   └── history.html
├── static/
│   ├── css/style.css           # Custom styles
│   ├── js/canvas.js            # Drawing canvas logic
│   ├── js/upload.js
│   ├── js/history.js
│   ├── uploads/                # User-uploaded images
│   └── outputs/                # Converted outputs + processing stages
├── models/
│   ├── shape_classifier.pkl    # Trained scikit-learn shape classifier
│   └── train_shape_model.py    # Model training script
└── database/
    ├── __init__.py
    ├── init_db.py              # Database schema initialization
    └── scribble.db             # SQLite database
```

## Features

- **User Authentication:** Registration, login, logout, user dashboard
- **Drawing Canvas:** Pencil/eraser tools, brush sizes, color picker, undo, clear, download
- **Image Upload:** JPG/JPEG/PNG with preview and validation
- **AI Image Processing:** Grayscale conversion, Gaussian/median denoising, adaptive thresholding, Canny edge detection, contour analysis
- **ML Shape Classification:** RandomForest classifier trained on synthetic shapes (circle, square, rectangle, triangle, star)
- **Digital Conversion:** Clean line-drawing output generation
- **Before & After Comparison:** Side-by-side original vs converted view
- **Download & History:** Download converted images, view/delete conversion history
- **Processing Stages:** View intermediate preprocessing steps

## Machine Learning vs Image Processing

This project clearly separates **image processing** (OpenCV filters: grayscale, denoising, thresholding, edges) from **machine learning** (scikit-learn RandomForest shape classifier trained on synthetic shape data). The ML model predicts the shape category; OpenCV handles the pixel-level transformation.

## Installation & Setup

### 1. Clone / Extract
```bash
cd "D:\scribble to digital"
```

### 2. Create Virtual Environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the ML Model
```bash
python models/train_shape_model.py
```

### 5. Initialize the Database
```bash
python database/init_db.py
```

### 6. Run the Application
```bash
python app.py
```

Then open: `http://127.0.0.1:5000`

## Usage

1. **Register** a new account or **Login**
2. **Upload** a hand-drawn sketch image (JPG/PNG) OR use the **Draw Canvas**
3. Click **Upload & Convert** to process the image
4. View the **Before & After** comparison with detected shape and ML confidence
5. **Download** the converted digital image
6. View/manage your **Conversion History**

## Database Schema

- `users` — id, username, email, password_hash, created_at
- `conversions` — id, user_id, original_filename, original_path, converted_path, detected_shape, confidence, created_at

## Testing

1. Register a user account
2. Upload a simple hand-drawn shape image
3. Verify the conversion output and detected shape label
4. Download the converted image
5. Check History page for the saved record
6. Try the Draw Canvas to draw and download directly

## Notes

- The SECRET_KEY should be changed in production (`app.py` line)
- Uploads are limited to 16MB
- The shape classifier is a lightweight scikit-learn RandomForest, suitable for this academic project scope