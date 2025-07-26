import os
import pickle
import hashlib
import json
import shutil
from typing import Dict, List, Optional
from pathlib import Path
from config.settings import settings

class CacheManager:
    def __init__(self):
        self.cache_dir = Path(settings.CACHE_PATH)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.knowledge_base_cache = self.cache_dir / "knowledge_base.pkl"
        self.embeddings_cache = self.cache_dir / "embeddings.pkl"
        self.metadata_cache = self.cache_dir / "metadata.json"
    
    def _get_file_hash(self, file_path: str) -> str:
        """Get MD5 hash of file to check for changes"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def is_cache_valid(self, pdf_path: str) -> bool:
        """Check if cache is valid for the given PDF"""
        if not all([
            self.knowledge_base_cache.exists(),
            self.embeddings_cache.exists(),
            self.metadata_cache.exists()
        ]):
            return False
        
        try:
            with open(self.metadata_cache, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Check if file hash matches
            current_hash = self._get_file_hash(pdf_path)
            return metadata.get('file_hash') == current_hash
        except:
            return False
    
    def save_knowledge_base(self, pdf_path: str, documents: List, embeddings, document_texts: List[str]):
        """Save knowledge base to cache"""
        try:
            # Save documents
            with open(self.knowledge_base_cache, 'wb') as f:
                pickle.dump(documents, f)
            
            # Save embeddings
            with open(self.embeddings_cache, 'wb') as f:
                pickle.dump(embeddings, f)
            
            # Save metadata
            metadata = {
                'file_path': pdf_path,
                'file_hash': self._get_file_hash(pdf_path),
                'document_count': len(documents),
                'embedding_model': settings.EMBEDDING_MODEL,
                'chunk_size': settings.CHUNK_SIZE,
                'chunk_overlap': settings.CHUNK_OVERLAP,
                'document_texts': document_texts
            }
            
            with open(self.metadata_cache, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Knowledge base cached with {len(documents)} documents")
            return True
        except Exception as e:
            print(f"❌ Failed to cache knowledge base: {e}")
            return False
    
    def load_knowledge_base(self) -> Optional[Dict]:
        """Load knowledge base from cache"""
        try:
            # Load documents
            with open(self.knowledge_base_cache, 'rb') as f:
                documents = pickle.load(f)
            
            # Load embeddings
            with open(self.embeddings_cache, 'rb') as f:
                embeddings = pickle.load(f)
            
            # Load metadata
            with open(self.metadata_cache, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            print(f"✅ Loaded cached knowledge base with {len(documents)} documents")
            
            return {
                'documents': documents,
                'embeddings': embeddings,
                'metadata': metadata,
                'document_texts': metadata.get('document_texts', [])
            }
        except Exception as e:
            print(f"❌ Failed to load cached knowledge base: {e}")
            return None
    
    def clear_cache(self):
        """Clear all cached data including vector store"""
        try:
            cache_files = [
                self.knowledge_base_cache,
                self.embeddings_cache,
                self.metadata_cache
            ]
            
            # Clear cache files
            for cache_file in cache_files:
                if cache_file.exists():
                    cache_file.unlink()
                    print(f"🗑️ Removed: {cache_file}")
            
            # Clear vector store cache
            vector_store_path = Path(settings.VECTOR_STORE_PATH)
            if vector_store_path.exists():
                shutil.rmtree(vector_store_path)
                print(f"🗑️ Removed vector store: {vector_store_path}")
            
            print("✅ All cache cleared successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to clear cache: {e}")
            return False
    
    def force_clear_vector_store(self):
        """Force clear vector store (for database errors)"""
        try:
            vector_store_path = Path(settings.VECTOR_STORE_PATH)
            
            if vector_store_path.exists():
                # Try to remove normally first
                try:
                    shutil.rmtree(vector_store_path)
                    print(f"✅ Successfully removed vector store directory")
                except PermissionError:
                    # If permission error, try to change permissions and retry
                    print("⚠️ Permission error, attempting to fix...")
                    self._fix_permissions_and_remove(vector_store_path)
                except Exception as e:
                    print(f"⚠️ Error removing vector store: {e}")
                    # Try alternative removal methods
                    self._force_remove_directory(vector_store_path)
            
            # Recreate empty directory
            vector_store_path.mkdir(parents=True, exist_ok=True)
            print("✅ Vector store directory recreated")
            return True
            
        except Exception as e:
            print(f"❌ Failed to force clear vector store: {e}")
            return False
    
    def _fix_permissions_and_remove(self, path: Path):
        """Fix permissions and remove directory"""
        try:
            # Change permissions recursively
            for root, dirs, files in os.walk(path):
                for d in dirs:
                    os.chmod(os.path.join(root, d), 0o755)
                for f in files:
                    os.chmod(os.path.join(root, f), 0o755)
            
            # Try to remove again
            shutil.rmtree(path)
            print("✅ Fixed permissions and removed directory")
            
        except Exception as e:
            print(f"❌ Permission fix failed: {e}")
            raise e
    
    def _force_remove_directory(self, path: Path):
        """Force remove directory using various methods"""
        import platform
        
        try:
            if platform.system() == "Windows":
                # Windows specific removal
                os.system(f'rmdir /s /q "{path}"')
            else:
                # Unix/Linux specific removal
                os.system(f'rm -rf "{path}"')
            
            print(f"✅ Force removed directory using system command")
            
        except Exception as e:
            print(f"❌ Force removal failed: {e}")
            print("🔧 Manual action required: Please delete the vector store directory manually")
            print(f"   Directory: {path}")
    
    def get_cache_info(self) -> Dict:
        """Get information about cached data"""
        if self.metadata_cache.exists():
            try:
                with open(self.metadata_cache, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                return {
                    'cached': True,
                    'file_path': metadata.get('file_path', 'Unknown'),
                    'document_count': metadata.get('document_count', 0),
                    'embedding_model': metadata.get('embedding_model', 'Unknown'),
                    'cache_size': self._get_cache_size(),
                    'vector_store_exists': Path(settings.VECTOR_STORE_PATH).exists()
                }
            except:
                pass
        
        return {
            'cached': False,
            'vector_store_exists': Path(settings.VECTOR_STORE_PATH).exists()
        }
    
    def _get_cache_size(self) -> str:
        """Get total cache size in human readable format"""
        total_size = 0
        
        # Add cache files size
        for cache_file in self.cache_dir.glob('*'):
            if cache_file.is_file():
                total_size += cache_file.stat().st_size
        
        # Add vector store size
        vector_store_path = Path(settings.VECTOR_STORE_PATH)
        if vector_store_path.exists():
            for file_path in vector_store_path.rglob('*'):
                if file_path.is_file():
                    try:
                        total_size += file_path.stat().st_size
                    except:
                        pass  # Skip files we can't access
        
        # Convert to human readable format
        for unit in ['B', 'KB', 'MB', 'GB']:
            if total_size < 1024.0:
                return f"{total_size:.1f} {unit}"
            total_size /= 1024.0
        return f"{total_size:.1f} TB"