import re
from typing import Dict, List

class BengaliLanguageDetector:
    def __init__(self):
        # Bengali Unicode ranges
        self.bengali_range = (0x0980, 0x09FF)
        
        # Common Bengali words for detection
        self.bengali_words = {
            'common': ['এই', 'সেই', 'তাই', 'কাই', 'হয়', 'হল', 'করা', 'করে', 'হয়ে', 'থেকে'],
            'question': ['কী', 'কে', 'কার', 'কাকে', 'কোথায়', 'কেন', 'কিভাবে'],
            'literary': ['গল্প', 'কবিতা', 'সাহিত্য', 'লেখক', 'কবি', 'চরিত্র']
        }
    
    def detect_language(self, text: str) -> Dict:
        """Detect if text is Bengali, English, or mixed"""
        if not text.strip():
            return {'language': 'unknown', 'confidence': 0.0, 'bengali_ratio': 0.0}
        
        # Count Bengali characters
        bengali_chars = 0
        total_chars = 0
        
        for char in text:
            if char.isalpha():
                total_chars += 1
                if self.bengali_range[0] <= ord(char) <= self.bengali_range[1]:
                    bengali_chars += 1
        
        if total_chars == 0:
            return {'language': 'unknown', 'confidence': 0.0, 'bengali_ratio': 0.0}
        
        bengali_ratio = bengali_chars / total_chars
        
        # Check for Bengali words
        bengali_word_count = 0
        words = text.split()
        
        for word_category in self.bengali_words.values():
            for bengali_word in word_category:
                bengali_word_count += text.count(bengali_word)
        
        # Determine language
        if bengali_ratio > 0.7 or bengali_word_count > 2:
            language = 'bengali'
            confidence = min(bengali_ratio + (bengali_word_count * 0.1), 1.0)
        elif bengali_ratio > 0.3:
            language = 'mixed'
            confidence = bengali_ratio
        else:
            language = 'english'
            confidence = 1.0 - bengali_ratio
        
        return {
            'language': language,
            'confidence': confidence,
            'bengali_ratio': bengali_ratio,
            'bengali_words_found': bengali_word_count
        }
    
    def is_bengali_query(self, query: str) -> bool:
        """Simple check if query is primarily Bengali"""
        result = self.detect_language(query)
        return result['language'] in ['bengali', 'mixed'] and result['confidence'] > 0.5
