from .document_processor import TesseractDocumentProcessor
from .embeddings import MultilingualEmbeddings
from .vector_store import VectorStore
from .generator import MultiProviderResponseGenerator
from .query_processor import EnhancedBengaliQueryProcessor
from .enhanced_retriever import EnhancedBengaliRetriever
from .language_detector import BengaliLanguageDetector
from .cache_manager import CacheManager
from config.settings import settings
import os
import glob
from typing import List, Dict, Optional
from pathlib import Path

class RAGPipeline:
    def __init__(self):
        self.document_processor = TesseractDocumentProcessor(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        
        self.embeddings = MultilingualEmbeddings(settings.EMBEDDING_MODEL)
        
        self.vector_store = VectorStore(
            persist_directory=settings.VECTOR_STORE_PATH,
            collection_name=settings.COLLECTION_NAME
        )
        
        # Use multi-provider generator
        self.generator = MultiProviderResponseGenerator()
        
        self.query_processor = EnhancedBengaliQueryProcessor()
        self.language_detector = BengaliLanguageDetector()
        self.cache_manager = CacheManager()
        
        # Enhanced retriever
        self.retriever = None
        
        self.conversation_history = []
        self.document_texts = []
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'knowledge_base_size': 0,
            'llm_provider': self.generator.provider,
            'cache_used': False
        }
    
    def auto_build_knowledge_base(self) -> bool:
        """Automatically build knowledge base from data/raw directory"""
        raw_data_path = Path(settings.RAW_DATA_PATH)
        
        if not raw_data_path.exists():
            print(f"❌ Raw data directory not found: {raw_data_path}")
            return False
        
        # Find PDF files in raw data directory
        pdf_files = list(raw_data_path.glob("*.pdf"))
        
        if not pdf_files:
            print(f"❌ No PDF files found in {raw_data_path}")
            return False
        
        # Use the first PDF file found
        pdf_path = str(pdf_files[0])
        print(f"📁 Found PDF: {pdf_path}")
        
        return self.build_knowledge_base(pdf_path)
    
    def build_knowledge_base(self, pdf_path: str = None) -> int:
        """Build knowledge base with caching support"""
        
        # If no PDF path provided, try to auto-detect
        if pdf_path is None:
            return self.auto_build_knowledge_base()
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        print(f"🚀 Building knowledge base from: {pdf_path}")
        print(f"🔧 Using LLM Provider: {self.generator.provider.upper()}")
        
        # Check if cache is valid
        if self.cache_manager.is_cache_valid(pdf_path):
            print("📦 Loading from cache...")
            cached_data = self.cache_manager.load_knowledge_base()
            
            if cached_data:
                # Restore from cache
                documents = cached_data['documents']
                embeddings = cached_data['embeddings']
                self.document_texts = cached_data['document_texts']
                
                # Store in vector database
                self.vector_store.add_documents(documents, embeddings)
                
                # Initialize enhanced retriever
                self.retriever = EnhancedBengaliRetriever(
                    self.vector_store,
                    self.embeddings,
                    self.query_processor
                )
                self.retriever.fit_tfidf(self.document_texts)
                
                self.stats['knowledge_base_size'] = len(documents)
                self.stats['cache_used'] = True
                
                print(f"✅ Knowledge base loaded from cache! Total chunks: {len(documents)}")
                return len(documents)
        
        # Process documents with OCR
        print("🔍 Processing PDF with OCR...")
        documents = self.document_processor.process_pdf(pdf_path)
        
        if not documents:
            raise Exception("No documents were processed from the PDF!")
        
        # Store document texts for hybrid search
        self.document_texts = [doc.page_content for doc in documents]
        
        # Generate embeddings
        texts = [doc.page_content for doc in documents]
        print(f"🧠 Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embeddings.embed_texts(texts)
        
        # Store in vector database
        self.vector_store.add_documents(documents, embeddings)
        
        # Initialize enhanced retriever
        self.retriever = EnhancedBengaliRetriever(
            self.vector_store,
            self.embeddings,
            self.query_processor
        )
        self.retriever.fit_tfidf(self.document_texts)
        
        # Cache the processed data
        print("💾 Caching processed data...")
        self.cache_manager.save_knowledge_base(pdf_path, documents, embeddings, self.document_texts)
        
        self.stats['knowledge_base_size'] = len(documents)
        self.stats['cache_used'] = False
        
        print(f"✅ Knowledge base built successfully! Total chunks: {len(documents)}")
        return len(documents)
    
    def query(self, question: str, include_history: bool = True) -> Dict:
        """Enhanced query processing"""
        
        if not self.retriever:
            raise Exception("Knowledge base not built. Please build knowledge base first.")
        
        self.stats['total_queries'] += 1
        
        print(f"\n🔍 Processing Query: {question}")
        print(f"🤖 Using {self.generator.provider.upper()} for generation")
        
        # Enhanced query preprocessing
        query_info = self.query_processor.preprocess_query(question)
        lang_info = query_info['language_info']
        
        print(f"📝 Language: {lang_info['language']} (confidence: {lang_info['confidence']:.2f})")
        
        # Enhanced retrieval
        all_retrieved_docs = []
        
        # Strategy 1: Character-focused search
        char_query = self.query_processor.create_character_focused_query(question)
        if char_query != question:
            char_docs = self.retriever.hybrid_search(char_query, k=5)
            all_retrieved_docs.extend(char_docs)
        
        # Strategy 2: Standard search
        standard_docs = self.retriever.hybrid_search(question, k=8)
        all_retrieved_docs.extend(standard_docs)
        
        # Remove duplicates and sort by relevance
        seen_content = set()
        unique_docs = []
        for doc in all_retrieved_docs:
            content_hash = hash(doc.get('document', '')[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        # Sort by combined score
        retrieved_docs = sorted(
            unique_docs,
            key=lambda x: x.get('combined_score', x.get('score', 0)),
            reverse=True
        )
        
        # Apply similarity threshold
        min_threshold = 0.2 if lang_info['language'] == 'bengali' else 0.4
        relevant_docs = [
            doc for doc in retrieved_docs[:10]
            if doc.get('combined_score', doc.get('score', 0)) >= min_threshold
        ]
        
        print(f"📊 Retrieved {len(retrieved_docs)} total, {len(relevant_docs)} above threshold")
        
        # Generate response
        history = self.conversation_history if include_history else None
        response = self.generator.generate_response_with_context(
            question,
            relevant_docs,
            history,
            lang_info
        )
        
        print(f"💬 Generated response: {response}")
        
        # Update conversation history
        self.conversation_history.append({
            "role": "USER",
            "message": question,
            "language": lang_info['language']
        })
        self.conversation_history.append({
            "role": "CHATBOT",
            "message": response,
            "language": lang_info['language']
        })
        
        # Keep recent history only
        if len(self.conversation_history) > 12:
            self.conversation_history = self.conversation_history[-12:]
        
        # Calculate confidence
        confidence = 0.0
        if relevant_docs:
            confidence = max([doc.get('combined_score', doc.get('score', 0)) for doc in relevant_docs])
        
        # Check success
        success_indicators = [
            confidence > 0.4,
            len(response.split()) <= 8,
            any(entity.lower() in response.lower() for entity in query_info.get('entities', [])),
            not response.startswith('দুঃখিত')
        ]
        
        if sum(success_indicators) >= 2:
            self.stats['successful_queries'] += 1
        
        return {
            "question": question,
            "answer": response,
            "retrieved_docs": retrieved_docs[:8],
            "relevant_docs_count": len(relevant_docs),
            "confidence": confidence,
            "language_info": lang_info,
            "query_info": query_info,
            "llm_provider": self.generator.provider
        }
    
    def get_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        success_rate = 0
        if self.stats['total_queries'] > 0:
            success_rate = (self.stats['successful_queries'] / self.stats['total_queries']) * 100
        
        cache_info = self.cache_manager.get_cache_info()
        
        return {
            "knowledge_base": {
                "total_documents": self.stats['knowledge_base_size'],
                "collection_name": self.vector_store.collection_name,
                "cache_used": self.stats['cache_used'],
                "cache_info": cache_info
            },
            "conversation": {
                "history_length": len(self.conversation_history),
                "total_queries": self.stats['total_queries'],
                "successful_queries": self.stats['successful_queries'],
                "success_rate": f"{success_rate:.1f}%"
            },
            "models": {
                "embedding_model": settings.EMBEDDING_MODEL,
                "llm_provider": self.stats['llm_provider'],
                "llm_model": self.generator.model
            },
            "settings": {
                "chunk_size": settings.CHUNK_SIZE,
                "chunk_overlap": settings.CHUNK_OVERLAP,
                "similarity_threshold": settings.SIMILARITY_THRESHOLD
            }
        }
    
    def clear_cache(self) -> bool:
        """Clear all cached data"""
        return self.cache_manager.clear_cache()
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("🗑️ Conversation history cleared")
    
    def test_common_questions(self) -> Dict:
        """Test the system with common expected questions"""
        test_cases = [
            {
                "question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
                "expected_keywords": ["শুম্ভুনাথ", "শশুনাথ"],
                "expected_answer": "শুম্ভুনাথ"
            },
            {
                "question": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে ?",
                "expected_keywords": ["মামা", "মামাকে"],
                "expected_answer": "মামাকে"
            },
            {
                "question": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
                "expected_keywords": ["১৫", "বছর"],
                "expected_answer": "১৫ বছর"
            }
        ]
        
        results = []
        
        for test_case in test_cases:
            question = test_case["question"]
            expected = test_case["expected_answer"]
            
            print(f"\n🧪 Testing: {question}")
            
            try:
                result = self.query(question, include_history=False)
                answer = result["answer"]
                confidence = result["confidence"]
                
                # Check if answer contains expected keywords
                contains_keywords = any(
                    keyword.lower() in answer.lower()
                    for keyword in test_case["expected_keywords"]
                )
                
                # Simple similarity check
                is_correct = (
                    contains_keywords or
                    expected.lower() in answer.lower() or
                    answer.lower() in expected.lower()
                )
                
                test_result = {
                    "question": question,
                    "expected": expected,
                    "actual": answer,
                    "confidence": confidence,
                    "is_correct": is_correct,
                    "contains_keywords": contains_keywords,
                    "relevant_docs_count": result["relevant_docs_count"],
                    "llm_provider": result["llm_provider"]
                }
                
                results.append(test_result)
                
                status = "✅ PASS" if is_correct else "❌ FAIL"
                print(f"   {status} | Expected: {expected} | Got: {answer} | Confidence: {confidence:.2f}")
                
            except Exception as e:
                results.append({
                    "question": question,
                    "expected": expected,
                    "actual": f"ERROR: {str(e)}",
                    "confidence": 0.0,
                    "is_correct": False,
                    "error": str(e)
                })
                print(f"   ❌ ERROR: {e}")
        
        # Calculate overall performance
        correct_count = sum(1 for r in results if r.get("is_correct", False))
        total_count = len(results)
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        
        summary = {
            "total_tests": total_count,
            "correct_answers": correct_count,
            "accuracy": f"{accuracy:.1f}%",
            "test_results": results,
            "llm_provider": self.generator.provider
        }
        
        print(f"\n📊 TEST SUMMARY:")
        print(f"   LLM Provider: {self.generator.provider.upper()}")
        print(f"   Total tests: {total_count}")
        print(f"   Correct answers: {correct_count}")
        print(f"   Accuracy: {accuracy:.1f}%")
        
        return summary