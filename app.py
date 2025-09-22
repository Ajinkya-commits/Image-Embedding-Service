from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
import io
import base64
import numpy as np
import logging
import os
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

processor = None
model = None
device = None

def initialize_model():
    """Initialize DINOv2 model and processor"""
    global processor, model, device
    
    try:
        logger.info("Initializing DINOv2 model...")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        model_name = "facebook/dinov2-base"
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        model.eval()
        
        logger.info("DINOv2 model initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        return False

@lru_cache(maxsize=1)
def get_model_info():
    """Get model information"""
    return {
        "model_name": "facebook/dinov2-base",
        "embedding_dimension": 768,
        "device": str(device),
        "cuda_available": torch.cuda.is_available()
    }

def download_image(image_url):
    """Download image from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(image_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        image = Image.open(io.BytesIO(response.content))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        logger.error(f"Failed to download image from {image_url}: {str(e)}")
        raise

def process_base64_image(base64_data):
    """Process base64 encoded image"""
    try:
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]
        
        image_data = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        logger.error(f"Failed to process base64 image: {str(e)}")
        raise

def generate_embedding(image):
    """Generate embedding for a PIL image"""
    try:
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0].cpu().numpy()
        
        return embedding[0].tolist()
        
    except Exception as e:
        logger.error(f"Failed to generate embedding: {str(e)}")
        raise

@app.route('/embed/url', methods=['POST'])
def embed_image_url():
    """Generate embedding for image from URL"""
    try:
        data = request.get_json()
        
        if not data or 'image_url' not in data:
            return jsonify({"error": "image_url is required"}), 400
        
        image_url = data['image_url']
        logger.info(f"Processing image from URL: {image_url}")
        
        image = download_image(image_url)
        embedding = generate_embedding(image)
        
        return jsonify({
            "embedding": embedding,
            "dimensions": len(embedding),
            "model": "facebook/dinov2-base",
            "image_url": image_url
        })
        
    except Exception as e:
        logger.error(f"Error processing image URL: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/embed/base64', methods=['POST'])
def embed_image_base64():
    """Generate embedding for base64 encoded image"""
    try:
        data = request.get_json()
        
        if not data or 'image_data' not in data:
            return jsonify({"error": "image_data is required"}), 400
        
        image_data = data['image_data']
        mime_type = data.get('mime_type', 'image/jpeg')
        
        logger.info(f"Processing base64 image, mime_type: {mime_type}")
        
        image = process_base64_image(image_data)
        embedding = generate_embedding(image)
        
        return jsonify({
            "embedding": embedding,
            "dimensions": len(embedding),
            "model": "facebook/dinov2-base",
            "mime_type": mime_type
        })
        
    except Exception as e:
        logger.error(f"Error processing base64 image: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/embed/batch', methods=['POST'])
def embed_batch_images():
    """Generate embeddings for multiple images"""
    try:
        data = request.get_json()
        
        if not data or 'image_urls' not in data:
            return jsonify({"error": "image_urls array is required"}), 400
        
        image_urls = data['image_urls']
        if not isinstance(image_urls, list):
            return jsonify({"error": "image_urls must be an array"}), 400
        
        logger.info(f"Processing batch of {len(image_urls)} images")
        
        results = []
        for url in image_urls:
            try:
                image = download_image(url)
                embedding = generate_embedding(image)
                results.append({
                    "embedding": embedding,
                    "dimensions": len(embedding),
                    "model": "facebook/dinov2-base",
                    "image_url": url
                })
            except Exception as e:
                results.append({
                    "error": str(e),
                    "image_url": url
                })
        
        return jsonify({"results": results})
        
    except Exception as e:
        logger.error(f"Error processing batch images: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        if model is None or processor is None:
            return jsonify({
                "status": "unhealthy",
                "error": "Model not loaded"
            }), 500
        
        return jsonify({
            "status": "healthy",
            "model": "facebook/dinov2-base",
            "device": str(device),
            "service": "Image Embedding Service"
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    return jsonify(get_model_info())

if __name__ == '__main__':
    if initialize_model():
        logger.info("Starting Image Embedding Service...")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        logger.error("Failed to initialize model. Exiting.")
        exit(1)
