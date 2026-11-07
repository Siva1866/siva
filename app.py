import os
import cv2
import numpy as np
from PIL import Image
import joblib
import sqlite3
import uuid
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify, flash

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'static', 'outputs')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'shape_classifier.pkl')
DB_PATH = os.path.join(BASE_DIR, 'database', 'scribble.db')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

shape_model = None
if os.path.exists(MODEL_PATH):
    try:
        shape_model = joblib.load(MODEL_PATH)
    except Exception:
        shape_model = None


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    denoised = cv2.medianBlur(denoised, 3)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    edges = cv2.Canny(denoised, 50, 150)
    return {'original': img, 'gray': gray, 'denoised': denoised, 'thresh': thresh, 'edges': edges}


def extract_shape_features(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    if peri == 0:
        return None, None
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = float(w) / h if h else 0
    extent = float(area) / (w * h) if (w * h) else 0
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area else 0
    circularity = 4 * np.pi * (area / (peri * peri)) if peri else 0
    vertices = len(approx)
    features = np.array([[aspect_ratio, extent, solidity, circularity, vertices, area, peri]])
    return features, c


def classify_shape(features):
    if shape_model is not None and features is not None:
        try:
            pred = shape_model.predict(features)[0]
            prob = float(shape_model.predict_proba(features)[0].max()) if hasattr(shape_model, 'predict_proba') else 1.0
            return str(pred), prob
        except Exception:
            pass
    if features is None:
        return 'unknown', 0.0
    aspect_ratio, extent, solidity, circularity, vertices, area, peri = features[0]
    if circularity > 0.75:
        return 'circle', circularity
    if vertices == 3:
        return 'triangle', 0.8
    if vertices == 4:
        if 0.9 < aspect_ratio < 1.1:
            return 'square', 0.85
        return 'rectangle', 0.85
    if vertices > 4:
        return 'polygon', 0.7
    return 'unknown', 0.5


def generate_clean_output(img, thresh):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    clean = np.ones_like(gray) * 255
    clean[binary > 0] = 0
    kernel = np.ones((2, 2), np.uint8)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)
    return clean


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        if not username or not email or not password:
            flash('All fields are required.', 'error')
            return render_template('register.html')
        conn = get_db()
        cur = conn.cursor()
        if cur.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone():
            flash('Username already exists.', 'error')
            conn.close()
            return render_template('register.html')
        if cur.execute('SELECT id FROM users WHERE email = ?', (email,)).fetchone():
            flash('Email already exists.', 'error')
            conn.close()
            return render_template('register.html')
        pw_hash = generate_password_hash(password)
        cur.execute('INSERT INTO users (username, email, password_hash, created_at) VALUES (?, ?, ?, ?)',
                    (username, email, pw_hash, datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
        flash('Registration successful. Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        conn = get_db()
        cur = conn.cursor()
        user = cur.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Logged in successfully.', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid username or password.', 'error')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out.', 'info')
    return redirect(url_for('index'))


@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    conn = get_db()
    conversions = conn.execute('''SELECT * FROM conversions WHERE user_id = ? ORDER BY created_at DESC''',
                               (session['user_id'],)).fetchall()
    conn.close()
    return render_template('dashboard.html', conversions=conversions)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part.', 'error')
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash('No selected file.', 'error')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique = f"{uuid.uuid4().hex}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique)
            file.save(filepath)
            session['last_upload'] = filepath
            session['last_original_name'] = filename
            flash('Image uploaded successfully.', 'success')
            return redirect(url_for('convert', file_id=unique))
        flash('Invalid file type. Allowed: JPG, JPEG, PNG.', 'error')
    return render_template('upload.html')


@app.route('/draw')
def draw():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('draw.html')


@app.route('/convert')
def convert():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    file_id = request.args.get('file_id')
    filepath = session.get('last_upload')
    if not filepath or not os.path.exists(filepath):
        flash('No image found to convert.', 'error')
        return redirect(url_for('upload'))
    stages = preprocess_image(filepath)
    if stages is None:
        flash('Failed to process image.', 'error')
        return redirect(url_for('upload'))
    clean = generate_clean_output(stages['original'], stages['thresh'])
    out_name = f"conv_{os.path.basename(filepath)}"
    out_path = os.path.join(app.config['OUTPUT_FOLDER'], out_name)
    cv2.imwrite(out_path, clean)
    features, _ = extract_shape_features(stages['thresh'])
    shape, confidence = classify_shape(features)
    conn = get_db()
    conn.execute('''INSERT INTO conversions (user_id, original_filename, original_path, converted_path, detected_shape, confidence, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)''',
                 (session['user_id'], session.get('last_original_name', os.path.basename(filepath)),
                  filepath.replace('\\', '/'), out_path.replace('\\', '/'), shape, confidence,
                  datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()
    session['last_converted'] = out_path
    session['last_shape'] = shape
    session['last_confidence'] = confidence
    stages_rel = {}
    for k, v in stages.items():
        fname = f"stage_{k}_{os.path.basename(filepath)}"
        fpath = os.path.join(app.config['OUTPUT_FOLDER'], fname)
        cv2.imwrite(fpath, v)
        stages_rel[k] = fpath.replace('\\', '/')
    return render_template('convert.html', original=filepath.replace('\\', '/'),
                           converted=out_path.replace('\\', '/'), shape=shape,
                           confidence=round(confidence * 100, 2), stages=stages_rel)


@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    conn = get_db()
    conversions = conn.execute('''SELECT * FROM conversions WHERE user_id = ? ORDER BY created_at DESC''',
                               (session['user_id'],)).fetchall()
    conn.close()
    return render_template('history.html', conversions=conversions)


@app.route('/download/<int:conv_id>')
def download(conv_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    conn = get_db()
    row = conn.execute('SELECT * FROM conversions WHERE id = ? AND user_id = ?', (conv_id, session['user_id'])).fetchone()
    conn.close()
    if not row:
        flash('Conversion not found.', 'error')
        return redirect(url_for('history'))
    path = row['converted_path']
    if not os.path.exists(path):
        flash('Converted file not found on disk.', 'error')
        return redirect(url_for('history'))
    return send_file(path, as_attachment=True)


@app.route('/delete/<int:conv_id>', methods=['POST'])
def delete(conv_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    conn = get_db()
    row = conn.execute('SELECT * FROM conversions WHERE id = ? AND user_id = ?', (conv_id, session['user_id'])).fetchone()
    if row:
        try:
            if os.path.exists(row['converted_path']):
                os.remove(row['converted_path'])
            if os.path.exists(row['original_path']):
                os.remove(row['original_path'])
        except OSError:
            pass
        conn.execute('DELETE FROM conversions WHERE id = ?', (conv_id,))
        conn.commit()
        flash('Conversion deleted.', 'success')
    conn.close()
    return redirect(url_for('history'))


@app.route('/api/shape-detect', methods=['POST'])
def api_shape_detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    tmp = os.path.join(UPLOAD_FOLDER, 'api_tmp_' + secure_filename(file.filename))
    file.save(tmp)
    stages = preprocess_image(tmp)
    if stages is None:
        os.remove(tmp)
        return jsonify({'error': 'Failed to process image'}), 500
    features, _ = extract_shape_features(stages['thresh'])
    shape, confidence = classify_shape(features)
    os.remove(tmp)
    return jsonify({'shape': shape, 'confidence': round(confidence * 100, 2)})


if __name__ == '__main__':
    app.run(debug=True)
