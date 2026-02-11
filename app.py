import os
import io
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageEnhance
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
from collections import deque
import base64
import logging
import warnings
import time
from concurrent.futures import ThreadPoolExecutor
import threading

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class OptimizedDeepfakeDetector:
    def __init__(self):
        self.models = {}
        self.score_history = deque(maxlen=5)  # Reduced for less lag
        self.face_cascade = None
        self.model_weights = {}
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.last_analysis_time = 0
        self.min_analysis_interval = 0.5  # Minimum 500ms between analyses
        self.cache = {}  # Simple cache for repeated frames
        self.load_models()
        self.setup_face_detection()
    
    def load_models(self):
        """Load verified working models"""
        models_config = [
            {
                "name": "deepfake_detector_v1",
                "model_id": "dima806/deepfake_vs_real_image_detection",
                "weight": 0.45,
                "threshold": 0.5
            },
            {
                "name": "deepfake_detector_v2", 
                "model_id": "umm-maybe/AI-image-detector",
                "weight": 0.35,
                "threshold": 0.5
            },
            {
                "name": "general_classifier",
                "model_id": "google/vit-base-patch16-224",  # Fallback general classifier
                "weight": 0.2,
                "threshold": 0.5
            }
        ]
        successfully_loaded = 0
        for config in models_config:
            try:
                logger.info(f"Loading {config['name']} model...")
                
                # Load with optimizations for speed
                pipeline_model = pipeline(
                    "image-classification", 
                    model=config["model_id"],
                    device=-1,  # Use CPU
                    trust_remote_code=True
                )
                
                self.models[config["name"]] = {
                    "pipeline": pipeline_model,
                    "weight": config["weight"], 
                    "threshold": config["threshold"]
                }
                
                self.model_weights[config["name"]] = config["weight"]
                successfully_loaded += 1
                logger.info(f"✅ Successfully loaded {config['name']}")
                
            except Exception as e:
                logger.error(f"❌ Failed to load {config['name']}: {e}")
                
                # Try specific backup models for each type
                backup_loaded = False
                
                if config["name"] == "deepfake_detector_v1":
                    # Try alternative deepfake models
                    backup_models = [
                        "Wvolf/ViT_Deepfake_Detection",
                        "prithivMLmods/Deep-Fake-Detector-v2-Model"
                    ]
                    
                    for backup_id in backup_models:
                        try:
                            logger.info(f"Trying backup model: {backup_id}")
                            backup_pipeline = pipeline(
                                "image-classification",
                                model=backup_id,
                                device=-1,
                                trust_remote_code=True
                            )
                            
                            self.models[config["name"]] = {
                                "pipeline": backup_pipeline,
                                "weight": config["weight"],
                                "threshold": config["threshold"]
                            }
                            self.model_weights[config["name"]] = config["weight"]
                            successfully_loaded += 1
                            logger.info(f"✅ Loaded backup model {backup_id} for {config['name']}")
                            backup_loaded = True
                            break
                            
                        except Exception as backup_e:
                            logger.warning(f"Backup model {backup_id} also failed: {backup_e}")
                            continue
                
                elif config["name"] == "deepfake_detector_v2":
                    # Try other AI detection models
                    backup_models = [
                        "microsoft/DinoV2-base",
                        "facebook/deit-base-distilled-patch16-224"
                    ]
                    
                    for backup_id in backup_models:
                        try:
                            logger.info(f"Trying backup model: {backup_id}")
                            backup_pipeline = pipeline(
                                "image-classification",
                                model=backup_id,
                                device=-1,
                                trust_remote_code=True
                            )
                            
                            self.models[config["name"]] = {
                                "pipeline": backup_pipeline,
                                "weight": config["weight"],
                                "threshold": config["threshold"]
                            }
                            self.model_weights[config["name"]] = config["weight"]
                            successfully_loaded += 1
                            logger.info(f"✅ Loaded backup model {backup_id} for {config['name']}")
                            backup_loaded = True
                            break
                            
                        except Exception as backup_e:
                            logger.warning(f"Backup model {backup_id} also failed: {backup_e}")
                            continue
                
                if not backup_loaded:
                    logger.warning(f"All attempts failed for {config['name']}")
        
        if successfully_loaded == 0:
            # Last resort: Load a basic working model
            logger.warning("Loading minimal fallback model...")
            try:
                fallback_model = pipeline(
                    "image-classification",
                    model="google/vit-base-patch16-224",
                    device=-1
                )
                self.models["fallback"] = {
                    "pipeline": fallback_model,
                    "weight": 1.0,
                    "threshold": 0.5
                }
                self.model_weights["fallback"] = 1.0
                successfully_loaded = 1
                logger.info("✅ Loaded minimal fallback model")
            except Exception as final_e:
                raise Exception(f"No models could be loaded! Final error: {final_e}")
        
        # Rebalance weights
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            for name in self.model_weights:
                self.model_weights[name] /= total_weight
        
        logger.info(f"✅ Successfully loaded {successfully_loaded}/{len(models_config)} models")
    
    def setup_face_detection(self):
        """Setup optimized face detection"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                logger.warning("Face cascade failed to load")
                self.face_cascade = None
            else:
                logger.info("✅ Face detection initialized")
                
        except Exception as e:
            logger.warning(f"Face detection setup failed: {e}")
            self.face_cascade = None
    
    def quick_preprocess(self, image):
        """Minimal preprocessing for speed"""
        # Only resize if image is too large
        max_size = 512
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image

    def extract_face_region(self, image):
        """Fast face extraction"""
        if self.face_cascade is None:
            return image
        
        try:
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Faster detection with looser parameters
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(60, 60)
            )
            
            if len(faces) > 0:
                # Get the largest face
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                
                # Minimal padding
                padding = int(0.2 * min(w, h))
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(img_array.shape[1], x + w + padding)
                y2 = min(img_array.shape[0], y + h + padding)
                
                face_region = img_array[y1:y2, x1:x2]
                if face_region.size > 0:
                    return Image.fromarray(face_region)
            
        except Exception as e:
            logger.warning(f"Face extraction failed: {e}")
        
        return image
    
    def get_frame_hash(self, image):
        """Simple hash for frame caching"""
        img_array = np.array(image.resize((64, 64)))
        return hash(img_array.tobytes())
    
    def predict_single_model(self, model_name, model_info, image):
        """Predict with a single model"""
        try:
            results = model_info["pipeline"](image)
            
            fake_score = 0.5
            confidence = 0.5
            
            if isinstance(results, list) and len(results) > 0:
                # Look for deepfake-related labels
                for result in results:
                    label = result["label"].lower()
                    score = result["score"]
                    
                    # More comprehensive keyword matching
                    fake_keywords = ["fake", "deepfake", "generated", "ai", "synthetic", 
                                   "artificial", "manipulated", "forged", "gan", "stylegan"]
                    real_keywords = ["real", "authentic", "human", "genuine", "natural", 
                                   "original", "legit", "true"]
                    
                    if any(keyword in label for keyword in fake_keywords):
                        fake_score = score
                        confidence = score
                        break
                    elif any(keyword in label for keyword in real_keywords):
                        fake_score = 1 - score
                        confidence = score
                        break
            
            return {
                "score": fake_score,
                "confidence": confidence,
                "weight": model_info["weight"]
            }
            
        except Exception as e:
            logger.warning(f"Model {model_name} prediction failed: {e}")
            return None
    
    def ensemble_predict_fast(self, image):
        """Fast ensemble prediction with parallel processing"""
        predictions = {}
        futures = {}
        
        # Submit all model predictions to thread pool
        for name, model_info in self.models.items():
            future = self.executor.submit(self.predict_single_model, name, model_info, image)
            futures[name] = future
        
        # Collect results with timeout
        for name, future in futures.items():
            try:
                result = future.result(timeout=3.0)  # 3 second timeout
                if result:
                    predictions[name] = result
            except Exception as e:
                logger.warning(f"Model {name} timed out or failed: {e}")
        
        if not predictions:
            return 0.5, 0.3  # Default values
        
        # Weighted ensemble
        weighted_score = 0
        total_weight = 0
        confidence_scores = []
        
        for pred in predictions.values():
            weight = pred["weight"] * pred["confidence"]  # Weight by confidence
            weighted_score += pred["score"] * weight
            total_weight += weight
            confidence_scores.append(pred["confidence"])
        
        if total_weight > 0:
            weighted_score /= total_weight
        
        ensemble_confidence = np.mean(confidence_scores) if confidence_scores else 0.3
        
        return weighted_score, ensemble_confidence
    
    def minimal_smoothing(self, current_score, confidence):
        """Minimal smoothing to reduce lag"""
        self.score_history.append({
            'score': current_score,
            'confidence': confidence
        })
        
        if len(self.score_history) < 2:
            return current_score, confidence
        
        # Simple weighted average with more weight on recent frames
        weights = np.array([0.1, 0.2, 0.3, 0.4, 1.0])[-len(self.score_history):]
        weights = weights / np.sum(weights)
        
        scores = [entry['score'] for entry in self.score_history]
        confidences = [entry['confidence'] for entry in self.score_history]
        
        smoothed_score = np.average(scores, weights=weights)
        smoothed_confidence = np.average(confidences, weights=weights)
        
        return smoothed_score, smoothed_confidence
    
    def analyze_frame(self, image):
        """Optimized frame analysis"""
        current_time = time.time()
        
        # Rate limiting to prevent overload
        if current_time - self.last_analysis_time < self.min_analysis_interval:
            # Return last result if analyzing too frequently
            if hasattr(self, 'last_result'):
                return self.last_result
        
        self.last_analysis_time = current_time
        
        try:
            # Check cache first
            frame_hash = self.get_frame_hash(image)
            if frame_hash in self.cache:
                cached_result = self.cache[frame_hash]
                if current_time - cached_result['timestamp'] < 2.0:  # Use cache for 2 seconds
                    return cached_result['result']
            
            # Quick preprocessing
            processed_image = self.quick_preprocess(image)
            face_image = self.extract_face_region(processed_image)
            
            # Fast ensemble prediction
            raw_score, raw_confidence = self.ensemble_predict_fast(face_image)
            
            # Minimal smoothing
            smoothed_score, smoothed_confidence = self.minimal_smoothing(raw_score, raw_confidence)
            
            # Simple threshold
            threshold = 0.6  # Fixed threshold for consistency
            is_deepfake = smoothed_score > threshold
            
            # Adjust confidence based on score distance from threshold
            distance_from_threshold = abs(smoothed_score - threshold)
            adjusted_confidence = min(0.95, smoothed_confidence + (distance_from_threshold * 0.2))
            
            result = {
                "is_deepfake": is_deepfake,
                "raw_score": float(raw_score),
                "smoothed_score": float(smoothed_score),
                "confidence": float(adjusted_confidence),
                "threshold_used": float(threshold),
                "processing_time": time.time() - current_time,
                "face_detected": face_image.size != processed_image.size,
                "models_used": len(self.models)
            }
            
            # Cache result
            self.cache[frame_hash] = {
                'result': result,
                'timestamp': current_time
            }
            
            # Clean old cache entries
            if len(self.cache) > 10:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
                del self.cache[oldest_key]
            
            self.last_result = result
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                "is_deepfake": False,
                "error": str(e),
                "confidence": 0.0,
                "raw_score": 0.5,
                "smoothed_score": 0.5,
                "threshold_used": 0.6,
                "processing_time": time.time() - current_time
            }

# Initialize detector
logger.info("🚀 Initializing Optimized Deepfake Detector...")
detector = OptimizedDeepfakeDetector()
logger.info("✅ Optimized Detector ready!")

@app.route('/analyze', methods=['POST'])
def analyze_frame():
    start_time = time.time()
    
    try:
        data = request.get_json()
        frame_b64 = data.get('frame')
        
        if not frame_b64:
            return jsonify({"error": "No frame provided"}), 400
        
        # Decode image
        try:
            if ',' in frame_b64:
                header, encoded = frame_b64.split(',', 1)
            else:
                encoded = frame_b64
            
            image_data = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            if image.size[0] < 32 or image.size[1] < 32:
                return jsonify({"error": "Image too small for analysis"}), 400
            
        except Exception as e:
            return jsonify({"error": f"Invalid image data: {str(e)}"}), 400
        
        # Analyze frame
        result = detector.analyze_frame(image)
        
        # Format response
        response = {
            "prediction": "Deepfake" if result.get("is_deepfake", False) else "Real",
            "score": result.get("smoothed_score", 0.5),
            "confidence": result.get("confidence", 0.0),
            "details": {
                "raw_score": result.get("raw_score", 0.5),
                "threshold_used": result.get("threshold_used", 0.6),
                "face_detected": result.get("face_detected", False),
                "models_used": result.get("models_used", 0),
                "processing_time_ms": round((time.time() - start_time) * 1000, 2)
            },
            "message": f"Analysis complete in {result.get('processing_time', 0):.3f}s"
        }
        
        if "error" in result:
            response["error"] = result["error"]
            response["confidence"] = 0.0
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Request processing failed: {e}")
        return jsonify({
            "error": str(e),
            "prediction": "Real",
            "score": 0.5,
            "confidence": 0.0,
            "message": "Analysis failed - defaulting to Real",
            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "models_loaded": len(detector.models),
        "face_detection": detector.face_cascade is not None,
        "cache_size": len(detector.cache),
        "optimization_mode": "fast_and_accurate",
        "ready": True
    }), 200

@app.route('/stats', methods=['GET'])  
def get_stats():
    return jsonify({
        "models": list(detector.models.keys()),
        "weights": detector.model_weights,
        "history_size": len(detector.score_history),
        "cache_size": len(detector.cache),
        "face_detection_enabled": detector.face_cascade is not None,
        "analysis_interval_ms": detector.min_analysis_interval * 1000
    }), 200

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Clear the analysis cache"""
    detector.cache.clear()
    detector.score_history.clear()
    return jsonify({"message": "Cache cleared", "status": "success"}), 200

if __name__ == '__main__':
    logger.info("🚀 Starting Optimized Server...")
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
