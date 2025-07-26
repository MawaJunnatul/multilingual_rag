# Enhanced retrieval with hybrid search
import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class EnhancedBengaliRetriever:
    def __init__(self, vector_store, embeddings, query_processor):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.query_processor = query_processor
        
        # TF-IDF for keyword-based backup search
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=1000,
            stop_words=None  # Don't remove Bengali words
        )
        self.tfidf_fitted = False
        self.document_texts = []
    
    def fit_tfidf(self, documents: List[str]):
        """Fit TF-IDF on document collection"""
        self.document_texts = documents
        self.tfidf_vectorizer.fit(documents)
        self.document_tfidf = self.tfidf_vectorizer.transform(documents)
        self.tfidf_fitted = True
    
    def hybrid_search(self, query: str, k: int = 10) -> List[Dict]:
        """Perform hybrid search combining semantic and keyword matching"""
        
        # Preprocess query
        query_info = self.query_processor.preprocess_query(query)
        
        # 1. Semantic search using embeddings
        semantic_results = self._semantic_search(query_info, k)
        
        # 2. Keyword-based search using TF-IDF (if available)
        keyword_results = []
        if self.tfidf_fitted:
            keyword_results = self._keyword_search(query_info, k)
        
        # 3. Combine and re-rank results
        combined_results = self._combine_results(semantic_results, keyword_results, query_info)
        
        return combined_results[:k]
    
    def _semantic_search(self, query_info: Dict, k: int) -> List[Dict]:
        """Semantic search using embeddings"""
        query_embedding = self.embeddings.embed_query(query_info['processed'])
        results = self.vector_store.similarity_search(query_embedding, k=k)
        
        # Add search method info
        for result in results:
            result['search_method'] = 'semantic'
            result['original_score'] = result['score']
        
        return results
    
    def _keyword_search(self, query_info: Dict, k: int) -> List[Dict]:
        """Keyword-based search using TF-IDF"""
        if not self.tfidf_fitted:
            return []
        
        # Try all query variations
        all_scores = []
        
        for variation in query_info['expanded_forms']:
            query_tfidf = self.tfidf_vectorizer.transform([variation])
            scores = cosine_similarity(query_tfidf, self.document_tfidf)[0]
            all_scores.append(scores)
        
        # Take maximum score across variations
        max_scores = np.maximum.reduce(all_scores) if all_scores else np.zeros(len(self.document_texts))
        
        # Get top k results
        top_indices = np.argsort(max_scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if max_scores[idx] > 0.1:  # Minimum threshold
                results.append({
                    'document': self.document_texts[idx],
                    'score': float(max_scores[idx]),
                    'metadata': {'doc_id': idx},
                    'search_method': 'keyword'
                })
        
        return results
    
    def _combine_results(self, semantic_results: List[Dict], 
                        keyword_results: List[Dict], 
                        query_info: Dict) -> List[Dict]:
        """Combine and re-rank results from different search methods"""
        
        # Create a combined result set
        all_results = {}
        
        # Add semantic results with weight
        semantic_weight = 0.7 if query_info['language_info']['language'] == 'bengali' else 0.8
        
        for result in semantic_results:
            doc_key = result['document'][:100]  # Use first 100 chars as key
            if doc_key not in all_results:
                all_results[doc_key] = result.copy()
                all_results[doc_key]['combined_score'] = result['score'] * semantic_weight
            else:
                # Boost score if found by multiple methods
                all_results[doc_key]['combined_score'] = max(
                    all_results[doc_key]['combined_score'],
                    result['score'] * semantic_weight
                )
        
        # Add keyword results with weight
        keyword_weight = 0.3 if query_info['language_info']['language'] == 'bengali' else 0.2
        
        for result in keyword_results:
            doc_key = result['document'][:100]
            if doc_key not in all_results:
                all_results[doc_key] = result.copy()
                all_results[doc_key]['combined_score'] = result['score'] * keyword_weight
            else:
                # Combine scores
                all_results[doc_key]['combined_score'] += result['score'] * keyword_weight
                all_results[doc_key]['search_method'] = 'hybrid'
        
        # Sort by combined score
        final_results = list(all_results.values())
        final_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return final_results