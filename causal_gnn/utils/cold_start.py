"""Zero-shot learning approach to solve the cold start problem."""

import numpy as np
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from ..models.fusion import LearnableMultiModalFusion

# For zero-shot learning
try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: transformers not installed. Zero-shot text features will be disabled.")
    print("Install with: pip install transformers")
    TRANSFORMERS_AVAILABLE = False

# For image processing
try:
    import cv2
    from PIL import Image
    import torchvision.models as models
    import torchvision.transforms as transforms
    VISION_AVAILABLE = True
except ImportError:
    print("Warning: OpenCV or Pillow not installed. Image features will be disabled.")
    print("Install with: pip install opencv-python Pillow")
    VISION_AVAILABLE = False


class ColdStartSolver:
    """Zero-shot learning approach to solve the cold start problem."""
    
    def __init__(self, config):
        self.config = config
        self.attribute_embeddings = {}
        self.knowledge_graph = None
        self.pretrained_models = {}
        self.fusion_model = None
        
    def load_pretrained_models(self):
        """Load pretrained models for zero-shot learning."""
        if TRANSFORMERS_AVAILABLE:
            print("Loading DistilBERT for text features...")
            self.pretrained_models['text'] = AutoModel.from_pretrained('distilbert-base-uncased')
            self.pretrained_models['text_tokenizer'] = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        
        if VISION_AVAILABLE:
            print("Loading ResNet18 for image features...")
            self.pretrained_models['vision'] = models.resnet18(pretrained=True)
            self.pretrained_models['vision_transform'] = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        # Initialize learnable fusion model
        modalities = ['text', 'image', 'numeric', 'categorical']
        self.fusion_model = LearnableMultiModalFusion(modalities, self.config.embedding_dim)
        
    def extract_text_features(self, text):
        """Extract features from text using a pretrained language model."""
        if 'text' not in self.pretrained_models or not text:
            return np.zeros(self.config.embedding_dim)
        
        tokenizer = self.pretrained_models['text_tokenizer']
        model = self.pretrained_models['text']
        
        # Tokenize the text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Get the model output
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use the mean of the last hidden state as the text embedding
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        # Ensure embedding has the correct size
        if embedding.shape[0] != self.config.embedding_dim:
            # Pad or truncate to the correct size
            if embedding.shape[0] > self.config.embedding_dim:
                embedding = embedding[:self.config.embedding_dim]
            else:
                embedding = np.pad(embedding, (0, self.config.embedding_dim - embedding.shape[0]))
        
        return embedding
    
    def extract_image_features(self, image_path):
        """Extract features from an image using a pretrained vision model."""
        if 'vision' not in self.pretrained_models or not image_path:
            return np.zeros(self.config.embedding_dim)
        
        try:
            # Load and preprocess the image
            image = Image.open(image_path).convert('RGB')
            image = self.pretrained_models['vision_transform'](image).unsqueeze(0)
            
            # Get the model output
            model = self.pretrained_models['vision']
            with torch.no_grad():
                features = model(image)
            
            # Use the features before the classification layer
            embedding = features.squeeze().numpy()
            
            # Ensure embedding has the correct size
            if embedding.shape[0] != self.config.embedding_dim:
                if embedding.shape[0] > self.config.embedding_dim:
                    embedding = embedding[:self.config.embedding_dim]
                else:
                    embedding = np.pad(embedding, (0, self.config.embedding_dim - embedding.shape[0]))
            
            return embedding
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return np.zeros(self.config.embedding_dim)
    
    def generate_user_profile(self, user_data, schema):
        """Generate a user profile from available data using zero-shot learning."""
        profile = {
            'text_features': np.zeros(self.config.embedding_dim),
            'image_features': np.zeros(self.config.embedding_dim),
            'numeric_features': np.zeros(10),
            'categorical_features': np.zeros(10)
        }
        
        # Extract text features
        for col in schema.get('text_columns', []):
            if col in user_data and user_data[col]:
                text_features = self.extract_text_features(str(user_data[col]))
                profile['text_features'] += text_features
        
        # Extract image features
        for col in schema.get('image_columns', []):
            if col in user_data and user_data[col]:
                image_features = self.extract_image_features(user_data[col])
                profile['image_features'] += image_features
        
        # Extract numeric features
        numeric_cols = [col for col in schema.get('numeric_columns', []) if col in user_data]
        for i, col in enumerate(numeric_cols[:10]):  # Limit to 10 numeric features
            if not pd.isna(user_data[col]):
                profile['numeric_features'][i] = user_data[col]
        
        # Extract categorical features
        categorical_cols = [col for col in schema.get('user_columns', []) + schema.get('context_columns', []) if col in user_data and isinstance(user_data[col], str)]
        for i, col in enumerate(categorical_cols[:10]):  # Limit to 10 categorical features
            if not pd.isna(user_data[col]):
                # Simple hash-based encoding for categorical features
                profile['categorical_features'][i] = hash(str(user_data[col])) % 1000 / 1000
        
        return profile
    
    def generate_item_profile(self, item_data, schema):
        """Generate an item profile from available data using zero-shot learning."""
        profile = {
            'text_features': np.zeros(self.config.embedding_dim),
            'image_features': np.zeros(self.config.embedding_dim),
            'numeric_features': np.zeros(10),
            'categorical_features': np.zeros(10)
        }
        
        # Extract text features
        for col in schema.get('text_columns', []):
            if col in item_data and item_data[col]:
                text_features = self.extract_text_features(str(item_data[col]))
                profile['text_features'] += text_features
        
        # Extract image features
        for col in schema.get('image_columns', []):
            if col in item_data and item_data[col]:
                image_features = self.extract_image_features(item_data[col])
                profile['image_features'] += image_features
        
        # Extract numeric features
        numeric_cols = [col for col in schema.get('numeric_columns', []) if col in item_data]
        for i, col in enumerate(numeric_cols[:10]):  # Limit to 10 numeric features
            if not pd.isna(item_data[col]):
                profile['numeric_features'][i] = item_data[col]
        
        # Extract categorical features
        categorical_cols = [col for col in schema.get('item_columns', []) + schema.get('context_columns', []) if col in item_data and isinstance(item_data[col], str)]
        for i, col in enumerate(categorical_cols[:10]):  # Limit to 10 categorical features
            if not pd.isna(item_data[col]):
                # Simple hash-based encoding for categorical features
                profile['categorical_features'][i] = hash(str(item_data[col])) % 1000 / 1000
        
        return profile
    
    def predict_cold_start_interactions(self, user_profile, item_profiles):
        """Predict interactions for cold start users/items using learnable fusion."""
        if self.fusion_model is None:
            print("Fusion model not initialized. Using simple averaging.")
            # Fallback to simple averaging
            similarities = []
            for item_profile in item_profiles:
                text_sim = cosine_similarity(
                    user_profile['text_features'].reshape(1, -1),
                    item_profile['text_features'].reshape(1, -1)
                )[0, 0]
                
                image_sim = cosine_similarity(
                    user_profile['image_features'].reshape(1, -1),
                    item_profile['image_features'].reshape(1, -1)
                )[0, 0]
                
                numeric_sim = cosine_similarity(
                    user_profile['numeric_features'].reshape(1, -1),
                    item_profile['numeric_features'].reshape(1, -1)
                )[0, 0]
                
                categorical_sim = cosine_similarity(
                    user_profile['categorical_features'].reshape(1, -1),
                    item_profile['categorical_features'].reshape(1, -1)
                )[0, 0]
                
                combined_similarity = (0.4 * text_sim + 0.3 * image_sim + 0.2 * numeric_sim + 0.1 * categorical_sim)
                similarities.append(combined_similarity)
            
            return np.array(similarities)
        
        # Use learnable fusion
        similarities = []
        for item_profile in item_profiles:
            # Calculate individual similarities
            text_sim = cosine_similarity(
                user_profile['text_features'].reshape(1, -1),
                item_profile['text_features'].reshape(1, -1)
            )[0, 0]
            
            image_sim = cosine_similarity(
                user_profile['image_features'].reshape(1, -1),
                item_profile['image_features'].reshape(1, -1)
            )[0, 0]
            
            numeric_sim = cosine_similarity(
                user_profile['numeric_features'].reshape(1, -1),
                item_profile['numeric_features'].reshape(1, -1)
            )[0, 0]
            
            categorical_sim = cosine_similarity(
                user_profile['categorical_features'].reshape(1, -1),
                item_profile['categorical_features'].reshape(1, -1)
            )[0, 0]
            
            # Create similarity dictionary
            sim_dict = {
                'text': torch.tensor(text_sim),
                'image': torch.tensor(image_sim),
                'numeric': torch.tensor(numeric_sim),
                'categorical': torch.tensor(categorical_sim)
            }
            
            # Fuse similarities using learnable model
            with torch.no_grad():
                fused_sim = self.fusion_model(sim_dict)
                similarities.append(fused_sim.item())
        
        return np.array(similarities)

