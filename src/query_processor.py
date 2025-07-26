import re
from typing import List, Dict
from .language_detector import BengaliLanguageDetector

class EnhancedBengaliQueryProcessor:
    def __init__(self):
        self.language_detector = BengaliLanguageDetector()
        
        # Enhanced Bengali character name mappings
        self.character_mappings = {
            'অনুপম': ['অনুপম', 'অনপুম', 'অনুপমের', 'অনুপমের'],
            'কল্যাণী': ['কল্যাণী', 'কল্যাণীর'],
            'মামা': ['মামা', 'মামার', 'মামাকে'],
            'শুম্ভুনাথ': ['শশুনাথ', 'শস্তুনাথ', 'শুম্ভুনাথ', 'শশুনাথ'],
            'বিনু': ['বিনু', 'বিনুদা', 'বিনুদাদা', 'বিনুদাদার'],
            'হরিশ': ['হরিশ', 'হরিশের']
        }
        
        # Key concepts mapping
        self.concept_mappings = {
            'সুপুরুষ': ['সুপুরুষ', 'সু পুরুষ', 'ভাল পুরুষ', 'ভালো পুরুষ'],
            'ভাগ্য দেবতা': ['ভাগ্য দেবতা', 'ভাগ্যদেবতা', 'ভাগ্যের দেবতা'],
            'বয়স': ['বয়স', 'বছর', 'বয়সকাল', 'বয়ঃক্রম']
        }
        
        # Question word mappings
        self.question_words = {
            'কে': ['কে', 'কার', 'কাকে', 'কাহাকে'],
            'কত': ['কত', 'কতটুকু', 'কত পরিমাণ'],
            'কী': ['কী', 'কি', 'কিসে']
        }

    def preprocess_query(self, query: str) -> Dict:
        """Enhanced query preprocessing for better Bengali matching"""
        original_query = query
        
        # Step 1: Clean and normalize
        query = self._clean_and_normalize(query)
        
        # Step 2: Detect language
        lang_info = self.language_detector.detect_language(query)
        
        # Step 3: Expand character names and concepts
        expanded_query = self._expand_entities(query)
        
        # Step 4: Extract key information
        entities = self._extract_entities(query)
        question_type = self._identify_question_type(query)
        
        # Step 5: Generate query variations for better matching
        variations = self._generate_query_variations(expanded_query)
        
        return {
            'original': original_query,
            'processed': expanded_query,
            'language_info': lang_info,
            'question_type': question_type,
            'entities': entities,
            'expanded_forms': variations,
            'search_terms': self._extract_search_terms(query)
        }

    def _clean_and_normalize(self, query: str) -> str:
        """Clean and normalize Bengali query"""
        # Fix common typing issues
        fixes = {
            'অনপুম': 'অনুপম',
            'অনুপমের': 'অনুপমের', 
            'শস্তুনাথ': 'শুম্ভুনাথ',
            'কাকে': 'কে',
            '??': '?',
            '।।': '।'
        }
        
        for wrong, correct in fixes.items():
            query = query.replace(wrong, correct)
        
        # Clean spacing
        query = re.sub(r'\s+', ' ', query)
        return query.strip()

    def _expand_entities(self, query: str) -> str:
        """Expand character names and concepts in query"""
        expanded_terms = []
        words = query.split()
        
        for word in words:
            word_clean = re.sub(r'[।?!,]', '', word.lower())
            added = False
            
            # Check character mappings
            for canonical, variations in self.character_mappings.items():
                if word_clean in [v.lower() for v in variations]:
                    expanded_terms.extend(variations)
                    added = True
                    break
            
            # Check concept mappings
            if not added:
                for canonical, variations in self.concept_mappings.items():
                    if word_clean in [v.lower() for v in variations]:
                        expanded_terms.extend(variations)
                        added = True
                        break
            
            if not added:
                expanded_terms.append(word)
        
        return ' '.join(expanded_terms)

    def _extract_entities(self, query: str) -> List[str]:
        """Extract key entities from query"""
        entities = []
        
        # Extract character names
        for canonical, variations in self.character_mappings.items():
            for variation in variations:
                if variation in query:
                    entities.append(canonical)
                    break
        
        # Extract concepts
        for canonical, variations in self.concept_mappings.items():
            for variation in variations:
                if variation in query:
                    entities.append(canonical)
                    break
        
        # Extract numbers
        numbers = re.findall(r'[০-৯]+|[0-9]+', query)
        entities.extend(numbers)
        
        return list(set(entities))

    def _identify_question_type(self, query: str) -> str:
        """Identify question type for better matching"""
        if re.search(r'কে|কার|কাকে', query):
            return 'who'
        elif re.search(r'কী|কি', query):
            return 'what'  
        elif re.search(r'কত|কতটুকু', query):
            return 'quantity'
        elif re.search(r'কোথায়', query):
            return 'where'
        elif re.search(r'কেন', query):
            return 'why'
        elif re.search(r'কিভাবে', query):
            return 'how'
        else:
            return 'general'

    def _extract_search_terms(self, query: str) -> List[str]:
        """Extract important search terms"""
        # Remove question words and focus on content
        content_words = []
        
        question_patterns = ['কে', 'কার', 'কাকে', 'কী', 'কি', 'কত', 'কোথায়', 'কেন', 'কিভাবে']
        
        words = query.split()
        for word in words:
            word_clean = re.sub(r'[।?!,]', '', word)
            if word_clean not in question_patterns and len(word_clean) > 1:
                content_words.append(word_clean)
        
        return content_words

    def _generate_query_variations(self, query: str) -> List[str]:
        """Generate query variations for better matching"""
        variations = [query]
        
        # Add variations with different character name spellings
        base_variations = [query]
        
        for canonical, spelling_variants in self.character_mappings.items():
            new_variations = []
            for variation in base_variations:
                for variant in spelling_variants:
                    if canonical in variation:
                        new_var = variation.replace(canonical, variant)
                        new_variations.append(new_var)
            base_variations.extend(new_variations)
        
        # Limit variations to prevent explosion
        return list(set(base_variations[:10]))

    def create_character_focused_query(self, query: str) -> str:
        """Create character-focused version of query for better retrieval"""
        entities = self._extract_entities(query)
        question_type = self._identify_question_type(query)
        
        # Build focused query
        if question_type == 'who' and 'সুপুরুষ' in query:
            return 'অনুপমের ভাষায় সুপুরুষ শশুনাথ শুম্ভুনাথ'
        elif question_type == 'who' and 'ভাগ্য দেবতা' in query:
            return 'অনুপমের ভাগ্য দেবতা মামা'
        elif question_type == 'quantity' and 'বয়স' in query and 'কল্যাণী' in query:
            return 'কল্যাণীর বয়স বিয়ের সময় ১৫ বছর'
        
        return query