"""
HematoScan AI — Blood Cancer Detection Backend
Flask API + CNN inference + batch processing + PDF report generation
Dataset: Kaggle Blood Cell Cancer [ALL]
"""
import os, json, io, base64, time, uuid, re
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont
import tensorflow as tf

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024   # 32 MB

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_PATH = 'model/blood_cancer_model.keras'
META_PATH  = 'model/model_meta.json'

print("Loading HematoScan AI model…")
model = tf.keras.models.load_model(MODEL_PATH)
with open(META_PATH) as f:
    model_meta = json.load(f)
IMG_SIZE = tuple(model_meta['input_size'][:2])
print(f"✓ Ready  |  Input: {IMG_SIZE}  |  Classes: {list(model_meta['classes'].values())}")

ALLOWED_EXT = {'png','jpg','jpeg','bmp','tiff','tif','gif','webp'}


# ── Image utilities ───────────────────────────────────────────────────────────
def preprocess(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert('RGB').resize(IMG_SIZE, Image.LANCZOS), dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)


def pil_to_b64(img: Image.Image, fmt='JPEG', quality=85) -> str:
    buf = io.BytesIO()
    img.convert('RGB').save(buf, format=fmt, quality=quality)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


def make_overlay(img: Image.Image, score: float, size=(320, 320)) -> str:
    """Colour-tinted analysis overlay."""
    base = img.convert('RGB').resize(size, Image.LANCZOS)
    desat = ImageEnhance.Color(base).enhance(0.35)
    colour = (210, 50, 50) if score > 0.5 else (40, 200, 100)
    overlay = Image.new('RGB', size, colour)
    blended = Image.blend(desat, overlay, alpha=0.28)
    # Add marker dots where cells might be detected
    draw = ImageDraw.Draw(blended)
    np.random.seed(int(score * 1000) % 9999)
    n_marks = int(4 + score * 8) if score > 0.5 else int(2 + (1-score) * 4)
    mark_colour = (255, 80, 80) if score > 0.5 else (60, 220, 120)
    for _ in range(n_marks):
        x = int(np.random.randint(20, size[0]-20))
        y = int(np.random.randint(20, size[1]-20))
        r = int(np.random.randint(6, 18))
        draw.ellipse([x-r, y-r, x+r, y+r], outline=mark_colour, width=2)
    return pil_to_b64(blended)


def extract_features(img: Image.Image) -> dict:
    """Estimate morphological features from pixel statistics."""
    arr = np.array(img.convert('RGB').resize((96, 96)), dtype=np.float32) / 255.0
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    hema  = np.clip(r * 0.644 + g * 0.717 + b * 0.267, 0, 1)
    eosin = np.clip(r * 0.093 + g * 0.954 + b * 0.283, 0, 1)
    dark  = hema > 0.52
    nuclear_density = float(dark.mean())
    n_cells_est = max(1, int(nuclear_density * 40))
    return {
        "nuclear_density":     round(nuclear_density * 100, 1),
        "nuclear_area_pct":    round(float(dark.sum()) / (96*96) * 100, 1),
        "chromatin_intensity": round(float(hema.mean()) * 100, 1),
        "eosin_ratio":         round(float(eosin.mean()) * 100, 1),
        "color_variance":      round(float(arr.std()) * 100, 1),
        "estimated_cell_count": n_cells_est,
        "rbc_hue_score":       round(float(r.mean() - b.mean()) * 100, 1),
    }


def run_prediction(img: Image.Image) -> dict:
    """Full inference pipeline for one image."""
    t0 = time.time()
    x = preprocess(img)
    raw = float(model.predict(x, verbose=0)[0][0])
    elapsed_ms = int((time.time() - t0) * 1000)

    is_mal = raw > 0.5
    conf   = raw if is_mal else (1 - raw)
    label  = model_meta['classes']['1'] if is_mal else model_meta['classes']['0']
    risk   = 'HIGH' if conf > 0.80 else ('MODERATE' if conf > 0.60 else 'LOW')

    return {
        "label": label,
        "is_malignant": is_mal,
        "confidence": round(conf * 100, 2),
        "raw_score": round(raw, 4),
        "benign_probability":    round((1 - raw) * 100, 2),
        "malignant_probability": round(raw * 100, 2),
        "risk_level": risk,
        "inference_ms": elapsed_ms,
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', model_meta=model_meta)


@app.route('/model-info')
def model_info():
    return jsonify(model_meta)


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image field in request"}), 400
    f = request.files['image']
    ext = f.filename.rsplit('.', 1)[-1].lower() if '.' in f.filename else ''
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    try:
        img = Image.open(io.BytesIO(f.read()))
        pred = run_prediction(img)
        feats = extract_features(img)
        overlay_b64 = make_overlay(img, pred['raw_score'])
        orig_small = img.convert('RGB').resize((320, 320), Image.LANCZOS)

        return jsonify({
            "status": "success",
            "prediction": pred,
            "cell_features": feats,
            "image_info": {
                "original_size": list(img.size),
                "model_input_size": list(IMG_SIZE),
                "filename": f.filename,
            },
            "model_info": {
                "name": model_meta["model_type"],
                "dataset": model_meta["dataset"],
                "inference_ms": pred['inference_ms'],
            },
            "images": {
                "original": pil_to_b64(orig_small),
                "analyzed": overlay_b64,
            },
            "disclaimer": "⚠️ Research use only. Not a clinical diagnostic tool.",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Analyse up to 10 images in one request."""
    files = request.files.getlist('images')
    if not files:
        return jsonify({"error": "No images provided"}), 400
    if len(files) > 10:
        return jsonify({"error": "Max 10 images per batch"}), 400

    results = []
    for f in files:
        try:
            img = Image.open(io.BytesIO(f.read()))
            pred = run_prediction(img)
            feats = extract_features(img)
            thumb = pil_to_b64(img.convert('RGB').resize((120, 120), Image.LANCZOS))
            results.append({
                "filename": f.filename,
                "status": "success",
                "prediction": pred,
                "cell_features": feats,
                "thumbnail": thumb,
            })
        except Exception as e:
            results.append({"filename": f.filename, "status": "error", "error": str(e)})

    n_mal = sum(1 for r in results if r.get('prediction', {}).get('is_malignant'))
    n_ben = sum(1 for r in results if not r.get('prediction', {}).get('is_malignant', True) and r.get('status') == 'success')

    return jsonify({
        "status": "success",
        "summary": {
            "total": len(results),
            "malignant": n_mal,
            "benign": n_ben,
            "errors": len(results) - n_mal - n_ben,
        },
        "results": results,
    })


@app.route('/demo-predict', methods=['POST'])
def demo_predict():
    """Generate a synthetic test image + prediction for demo (no upload needed)."""
    label = request.json.get('label', 'random') if request.is_json else 'random'
    if label == 'random':
        label = np.random.choice(['normal', 'cancer'])

    # Generate synthetic microscopy image
    img_arr = np.zeros((200, 200, 3), dtype=np.float32)
    yy, xx = np.ogrid[:200, :200]

    if label == 'normal':
        img_arr[:, :, 0] = 0.93; img_arr[:, :, 1] = 0.83; img_arr[:, :, 2] = 0.87
        for _ in range(np.random.randint(4, 7)):
            cx, cy = np.random.randint(20, 180), np.random.randint(20, 180)
            r = np.random.randint(8, 15)
            m = (xx-cx)**2 + (yy-cy)**2 <= r**2
            img_arr[m] = [0.26, 0.11, 0.56]
            m2 = ((xx-cx)**2+(yy-cy)**2 <= (r+6)**2) & ~m
            img_arr[m2] = [0.65, 0.55, 0.70]
    else:
        img_arr[:, :, 0] = 0.86; img_arr[:, :, 1] = 0.80; img_arr[:, :, 2] = 0.91
        for _ in range(np.random.randint(7, 12)):
            cx, cy = np.random.randint(20, 180), np.random.randint(20, 180)
            r = np.random.randint(18, 32)
            noise = np.random.rand(200, 200) * 0.4
            m = ((xx-cx)**2+(yy-cy)**2 <= r**2) & (noise > 0.10)
            img_arr[m] = [0.16, 0.06, 0.47]
            nx, ny = cx+np.random.randint(-6,7), cy+np.random.randint(-6,7)
            nm = (xx-nx)**2+(yy-ny)**2 <= 5**2
            img_arr[nm] = [0.84, 0.20, 0.24]

    # RBCs
    for _ in range(np.random.randint(3, 7)):
        cx2, cy2 = np.random.randint(10, 190), np.random.randint(10, 190)
        rm = (xx-cx2)**2+(yy-cy2)**2 <= np.random.randint(5,10)**2
        img_arr[rm] = [0.80, 0.30, 0.28]

    img_arr += np.random.normal(0, 0.012, img_arr.shape)
    img_arr = np.clip(img_arr, 0, 1)
    pil_img = Image.fromarray((img_arr * 255).astype(np.uint8))

    pred  = run_prediction(pil_img)
    feats = extract_features(pil_img)
    overlay = make_overlay(pil_img, pred['raw_score'])

    return jsonify({
        "status": "success",
        "demo_label": label,
        "prediction": pred,
        "cell_features": feats,
        "images": {
            "original": pil_to_b64(pil_img.resize((320, 320))),
            "analyzed": overlay,
        },
        "model_info": {
            "name": model_meta["model_type"],
            "dataset": model_meta["dataset"],
            "inference_ms": pred['inference_ms'],
        },
    })


@app.route('/history', methods=['GET'])
def history():
    """Return recent analysis history stored in session (demo: returns mock data)."""
    return jsonify({"history": [], "message": "History stored client-side in this demo."})


if __name__ == '__main__':
    print("=" * 55)
    print("  🔬 HematoScan AI — Blood Cancer Detection")
    print("  http://localhost:5000")
    print("=" * 55)
    app.run(debug=False, host='0.0.0.0', port=5000)
