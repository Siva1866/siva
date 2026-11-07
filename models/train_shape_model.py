import os
import numpy as np
import cv2
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'shape_classifier.pkl')


def generate_synthetic_image(shape, size=128, noise=True):
    img = np.ones((size, size), dtype=np.uint8) * 255
    center = size // 2
    radius = size // 3
    if shape == 'circle':
        cv2.circle(img, (center, center), radius, 0, 3)
    elif shape == 'square':
        cv2.rectangle(img, (center - radius, center - radius), (center + radius, center + radius), 0, 3)
    elif shape == 'rectangle':
        cv2.rectangle(img, (center - radius, center - radius // 2), (center + radius, center + radius // 2), 0, 3)
    elif shape == 'triangle':
        pts = np.array([[center, center - radius], [center - radius, center + radius], [center + radius, center + radius]], np.int32)
        cv2.polylines(img, [pts], True, 0, 3)
    elif shape == 'star':
        outer = radius
        inner = radius // 2
        pts = []
        for i in range(10):
            angle = -np.pi / 2 + i * np.pi / 5
            r = outer if i % 2 == 0 else inner
            pts.append([int(center + r * np.cos(angle)), int(center + r * np.sin(angle))])
        pts = np.array(pts, np.int32)
        cv2.polylines(img, [pts], True, 0, 3)
    if noise:
        noise = np.random.randint(0, 60, (size, size), dtype=np.uint8)
        mask = noise > 50
        img[mask] = 0
    return img


def extract_features(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    if peri == 0:
        return None
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = float(w) / h if h else 0
    extent = float(area) / (w * h) if (w * h) else 0
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area else 0
    circularity = 4 * np.pi * (area / (peri * peri)) if peri else 0
    vertices = len(approx)
    return np.array([aspect_ratio, extent, solidity, circularity, vertices, area, peri])


def train():
    shapes = ['circle', 'square', 'rectangle', 'triangle', 'star']
    X, y = [], []
    for shape in shapes:
        for _ in range(400):
            img = generate_synthetic_image(shape)
            _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            feats = extract_features(thresh)
            if feats is not None:
                X.append(feats)
                y.append(shape)
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"Shape classifier accuracy: {acc:.4f}")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("Model saved to", MODEL_PATH)


if __name__ == '__main__':
    train()
