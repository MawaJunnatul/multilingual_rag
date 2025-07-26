from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import pickle
import os
import torch

class MultilingualEmbeddings:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        
        # Set device explicitly to CPU to avoid meta tensor issues
        self.device = "cpu"
        
        print(f"🔧 Loading embedding model: {model_name}")
        
        try:
            # Load model with explicit device specification
            self.model = SentenceTransformer(model_name)
            # Force to CPU immediately to avoid device transfer issues
            self.model = self.model.cpu()
            print(f"✅ Successfully loaded {model_name} on CPU")
            
        except Exception as e:
            print(f"⚠️ Error loading {model_name}: {e}")
            print("🔄 Trying fallback model: all-MiniLM-L6-v2")
            
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.model = self.model.cpu()
                self.model_name = 'all-MiniLM-L6-v2'
                print(f"✅ Successfully loaded fallback model on CPU")
                
            except Exception as e2:
                print(f"❌ Could not load any model: {e2}")
                raise Exception(f"Failed to load embedding model. Error: {e2}")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        try:
            # Ensure we're using CPU
            embeddings = self.model.encode(
                texts, 
                convert_to_numpy=True,
                show_progress_bar=False,
                device=self.device
            )
            return embeddings
            
        except Exception as e:
            print(f"⚠️ Error in embed_texts: {e}")
            # Try without device specification as fallback
            try:
                embeddings = self.model.encode(texts, convert_to_numpy=True)
                return embeddings
            except Exception as e2:
                raise Exception(f"Could not generate embeddings: {e2}")
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query"""
        try:
            embedding = self.model.encode(
                [query], 
                convert_to_numpy=True,
                show_progress_bar=False,
                device=self.device
            )
            return embedding[0]
            
        except Exception as e:
            print(f"⚠️ Error in embed_query: {e}")
            # Try without device specification as fallback
            try:
                embedding = self.model.encode([query], convert_to_numpy=True)
                return embedding[0]
            except Exception as e2:
                raise Exception(f"Could not generate query embedding: {e2}")
    
    def save_embeddings(self, embeddings: np.ndarray, path: str):
        """Save embeddings to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(embeddings, f)
    
    def load_embeddings(self, path: str) -> np.ndarray:
        """Load embeddings from disk"""
        with open(path, 'rb') as f:
            return pickle.load(f)
