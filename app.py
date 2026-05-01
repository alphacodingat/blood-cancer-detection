
import subprocess, sys, os

print("=" * 60)
print("  🔬 HematoScan AI — Blood Cancer Detection System")
print("  Dataset: Kaggle Blood Cell Cancer [ALL]")
print("  Model: Custom CNN (6 Conv Blocks, 849K parameters)")
print("=" * 60)

# Ensure required directories exist
os.makedirs('model', exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('uploads', exist_ok=True)

if os.path.exists('index.html') and not os.path.exists('templates/index.html'):
    import shutil
    shutil.copy('index.html', 'templates/index.html')
    print("  ✓ Moved index.html -> templates/index.html")

if not os.path.exists('model/blood_cancer_model.keras'):
    print("\nModel not found. Training now...")
    print("(Using synthetic data — takes ~1-2 min. For real data see README.)\n")
    result = subprocess.run(
        [sys.executable, 'model/train_model.py'],
        check=False
    )
    if result.returncode != 0:
        print("\nTraining failed. Check the output above.")
        sys.exit(1)
    print("Training complete.\n")
else:
    print("\n  Model found — skipping training.")

print("\n🌐 Starting Flask server at http://localhost:5000")
print("   Press Ctrl+C to stop\n")

from app import app
app.run(debug=True, host='0.0.0.0', port=5000)
