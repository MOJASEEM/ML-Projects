
from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from detector import HybridDetector, FFTAnalyzer
from PIL import Image
import io
import os
from dotenv import load_dotenv
import logging
from functools import lru_cache
import numpy as np

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))
app.config['JSON_SORT_KEYS'] = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GLOBAL SETUP 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f" Using device: {device}")

# Load model
MODEL_PATH = os.getenv('MODEL_PATH', 'models/detector_model.pth')
model = HybridDetector(backbone='resnet18').to(device)

# Try to load saved weights, if they exist
if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        logger.info(f" Model loaded from {MODEL_PATH}")
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                logger.error(f"!!! WEIGHT CRITICAL ERROR: Layer '{name}' contains NaN values !!!")
    except Exception as e:
        logger.warning(f" Could not load model weights: {e}")
        logger.info("Using randomly initialized model")
else:
    logger.warning(f" Model file not found at {MODEL_PATH}")
    logger.info("Using randomly initialized model")

model.eval()

# Preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# FFT Analyzer
fft_analyzer = FFTAnalyzer()

#  UTILITY FUNCTIONS 
def allowed_file(filename):
    """Check if file has allowed extension"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image, use_fft=True):
    """
    Predict if image is AI-generated
    Returns: prediction score (0=Real, 1=AI)
    """
    try:
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Extract FFT features
        fft_features = None
        if use_fft:
            img_np = np.array(image)
            fft_features = fft_analyzer.extract_fft_features(img_np)
            fft_features = torch.tensor(fft_features, dtype=torch.float32, device=device).unsqueeze(0)  
        
        with torch.no_grad():
            logits = model(img_tensor, fft_features)
            prediction = torch.sigmoid(logits).item()
        
        return prediction
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise

def get_prediction_label(score):
    """Convert prediction score to human-readable label"""
    if score > 0.7:
        return " Likely AI-Generated"
    elif score > 0.5:
        return " Possibly AI-Generated"
    elif score > 0.3:
        return " Possibly Real"
    else:
        return " Likely Real Photo"

def get_confidence_level(score):
    """Calculate confidence percentage"""
    if score > 0.5:
        return abs(score - 0.5) * 2 * 100
    else:
        return abs(0.5 - score) * 2 * 100

#  ROUTES 

@app.route('/')
def index():
    """Serve main page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint for image prediction
    Accepts: multipart/form-data with 'file' field
    Returns: JSON with prediction and confidence
    """
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'File type not allowed. Use: PNG, JPG, GIF, BMP, WEBP'
            }), 400
        
        # Load and validate image
        try:
            image = Image.open(io.BytesIO(file.read())).convert('RGB')
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Invalid image file: {str(e)}'
            }), 400
        
        # Get prediction
        prediction_score = predict_image(image)
        
        # Extract metadata
        label = get_prediction_label(prediction_score)
        confidence = get_confidence_level(prediction_score)
        
        logger.info(f"Prediction: {file.filename} - Score: {prediction_score:.4f}")
        
        return jsonify({
            'success': True,
            'filename': file.filename,
            'prediction': {
                'score': round(prediction_score, 4),
                'label': label,
                'confidence': round(confidence, 2),
                'raw_probability_ai': round(prediction_score, 4),
                'raw_probability_real': round(1 - prediction_score, 4)
            },
            'model_info': {
                'name': 'Hybrid CNN + FFT Detector',
                'backbone': 'ResNet-50',
                'device': str(device),
                'uses_frequency_analysis': True
            }
        }), 200
    
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint
    Accepts: JSON array of base64 encoded images
    """
    try:
        if 'images' not in request.json:
            return jsonify({
                'success': False,
                'error': 'No images provided'
            }), 400
        
        images_b64 = request.json['images']
        results = []
        
        for i, img_b64 in enumerate(images_b64):
            try:
                # Decode base64
                import base64
                img_data = base64.b64decode(img_b64)
                image = Image.open(io.BytesIO(img_data)).convert('RGB')
                
                # Predict
                score = predict_image(image)
                results.append({
                    'index': i,
                    'score': round(score, 4),
                    'label': get_prediction_label(score),
                    'confidence': round(get_confidence_level(score), 2)
                })
            except Exception as e:
                results.append({
                    'index': i,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'total': len(images_b64),
            'results': results
        }), 200
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({
            'success': False,
            'error': 'Batch prediction failed'
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': str(device),
        'model_loaded': True,
        'timestamp': str(__import__('datetime').datetime.now())
    }), 200

@app.route('/api/info', methods=['GET'])
def info():
    """Get model information"""
    return jsonify({
        'model_name': 'AI Image Detector',
        'version': '1.0.0',
        'backbone': 'ResNet-50',
        'features': [
            'Spatial feature extraction (CNN)',
            'Frequency domain analysis (FFT)',
            'Multi-model generalization',
            'Batch prediction support'
        ],
        'device': str(device),
        'input_size': [224, 224],
        'output_classes': 2,
        'class_names': ['Real Photo', 'AI-Generated']
    }), 200

# ERROR HANDLERS 

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large"""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size: 16MB'
    }), 413

@app.errorhandler(404)
def not_found(error):
    """Handle 404"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500"""
    logger.error(f"Internal error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

# MAIN 

if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.getenv('PORT', 5000))
    
    logger.info("=" * 60)
    logger.info(" AI Image Detector - Flask Server")
    logger.info("=" * 60)
    logger.info(f"Debug Mode: {debug_mode}")
    logger.info(f"Port: {port}")
    logger.info(f"Device: {device}")
    logger.info("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug_mode,
        use_reloader=False
    )