import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append('src')

from .rag_pipeline import RAGPipeline
from config.settings import settings

class RAGEvaluator:
    def __init__(self, rag_pipeline):
        self.rag_pipeline = rag_pipeline
        
        # Extended test cases for comprehensive evaluation
        self.test_cases = [
            {
                "question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
                "expected_keywords": ["শুম্ভুনাথ", "শশুনাথ"],
                "expected_answer": "শুম্ভুনাথ",
                "category": "character_identification"
            },
            {
                "question": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে ?",
                "expected_keywords": ["মামা", "মামাকে"],
                "expected_answer": "মামাকে",
                "category": "character_identification"
            },
            {
                "question": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
                "expected_keywords": ["১৫", "বছর", "পনের"],
                "expected_answer": "১৫ বছর",
                "category": "factual_information"
            },
            {
                "question": "অনুপমের বয়স কত বছর?",
                "expected_keywords": ["সাতাই", "১৭", "seventeen"],
                "expected_answer": "সাতাই",
                "category": "factual_information"
            },
            {
                "question": "অনুপমের বাবা কী কাজ করতেন?",
                "expected_keywords": ["ওকালতি", "উকিল"],
                "expected_answer": "ওকালতি",
                "category": "factual_information"
            },
            # English variations
            {
                "question": "Who is referred to as 'সুপুরুষ' in Anupam's language?",
                "expected_keywords": ["শুম্ভুনাথ", "shumbhunath"],
                "expected_answer": "শুম্ভুনাথ",
                "category": "multilingual"
            },
            {
                "question": "What was Kallyani's real age at marriage?",
                "expected_keywords": ["১৫", "15", "fifteen"],
                "expected_answer": "১৫ বছর",
                "category": "multilingual"
            }
        ]
    
    def run_evaluation(self, detailed=False):
        """Run comprehensive evaluation"""
        print("🧪 Starting RAG System Evaluation")
        print(f"📊 Provider: {self.rag_pipeline.generator.provider.upper()}")
        print(f"🔬 Total test cases: {len(self.test_cases)}")
        print("=" * 60)
        
        results = []
        
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n[{i}/{len(self.test_cases)}] Testing: {test_case['question'][:50]}...")
            
            try:
                # Get response from RAG system
                result = self.rag_pipeline.query(test_case['question'], include_history=False)
                
                # Evaluate response
                evaluation = self._evaluate_response(test_case, result)
                results.append(evaluation)
                
                # Print immediate result
                status = "✅ PASS" if evaluation['is_correct'] else "❌ FAIL"
                print(f"   {status} | Expected: {test_case['expected_answer']}")
                print(f"   Got: {evaluation['actual_answer']}")
                print(f"   Confidence: {evaluation['confidence']:.2f}")
                print(f"   Relevance Score: {evaluation['avg_relevance_score']:.2f}")
                
                if detailed:
                    print(f"   Retrieved Docs: {evaluation['retrieved_docs_count']}")
                    print(f"   Category: {test_case['category']}")
                    print(f"   LLM Provider: {evaluation['llm_provider']}")
                
            except Exception as e:
                print(f"   ❌ ERROR: {str(e)}")
                results.append({
                    'question': test_case['question'],
                    'expected_answer': test_case['expected_answer'],
                    'actual_answer': f"ERROR: {str(e)}",
                    'is_correct': False,
                    'error': str(e),
                    'category': test_case['category']
                })
        
        # Generate summary
        summary = self._generate_summary(results)
        
        # Print summary
        self._print_summary(summary)
        
        # Save detailed report
        report_file = self._save_report(summary, results)
        
        return summary, results, report_file
    
    def _evaluate_response(self, test_case, result):
        """Evaluate a single response"""
        actual_answer = result['answer']
        expected_answer = test_case['expected_answer']
        expected_keywords = test_case['expected_keywords']
        
        # Check if answer contains expected keywords
        contains_keywords = any(
            keyword.lower() in actual_answer.lower()
            for keyword in expected_keywords
        )
        
        # Check direct match
        direct_match = (
            expected_answer.lower() in actual_answer.lower() or
            actual_answer.lower() in expected_answer.lower()
        )
        
        # Overall correctness
        is_correct = contains_keywords or direct_match
        
        # Calculate average relevance score
        avg_relevance_score = 0.0
        if result.get('retrieved_docs'):
            scores = [doc.get('combined_score', doc.get('score', 0)) 
                     for doc in result['retrieved_docs']]
            avg_relevance_score = sum(scores) / len(scores) if scores else 0.0
        
        return {
            'question': test_case['question'],
            'expected_answer': expected_answer,
            'actual_answer': actual_answer,
            'is_correct': is_correct,
            'contains_keywords': contains_keywords,
            'direct_match': direct_match,
            'confidence': result.get('confidence', 0.0),
            'retrieved_docs_count': result.get('relevant_docs_count', 0),
            'avg_relevance_score': avg_relevance_score,
            'category': test_case['category'],
            'language_detected': result.get('language_info', {}).get('language', 'unknown'),
            'llm_provider': result.get('llm_provider', 'unknown')
        }
    
    def _generate_summary(self, results):
        """Generate evaluation summary"""
        total_tests = len(results)
        correct_answers = sum(1 for r in results if r.get('is_correct', False))
        
        # Calculate accuracy
        accuracy = (correct_answers / total_tests * 100) if total_tests > 0 else 0
        
        # Calculate average confidence
        confidences = [r.get('confidence', 0) for r in results if 'confidence' in r]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Calculate average relevance
        relevance_scores = [r.get('avg_relevance_score', 0) for r in results if 'avg_relevance_score' in r]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        
        # Category-wise performance
        category_performance = {}
        for result in results:
            category = result.get('category', 'unknown')
            if category not in category_performance:
                category_performance[category] = {'total': 0, 'correct': 0}
            
            category_performance[category]['total'] += 1
            if result.get('is_correct', False):
                category_performance[category]['correct'] += 1
        
        # Calculate category accuracies
        for category in category_performance:
            total = category_performance[category]['total']
            correct = category_performance[category]['correct']
            category_performance[category]['accuracy'] = (correct / total * 100) if total > 0 else 0
        
        return {
            'total_tests': total_tests,
            'correct_answers': correct_answers,
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'avg_relevance': avg_relevance,
            'category_performance': category_performance,
            'llm_provider': results[0].get('llm_provider', 'unknown') if results else 'unknown',
            'timestamp': datetime.now().isoformat()
        }
    
    def _print_summary(self, summary):
        """Print evaluation summary"""
        print("\n" + "=" * 60)
        print("📊 EVALUATION SUMMARY")
        print("=" * 60)
        
        print(f"🤖 LLM Provider: {summary['llm_provider'].upper()}")
        print(f"📈 Overall Accuracy: {summary['accuracy']:.1f}% ({summary['correct_answers']}/{summary['total_tests']})")
        print(f"🎯 Average Confidence: {summary['avg_confidence']:.2f}")
        print(f"📄 Average Relevance: {summary['avg_relevance']:.2f}")
        
        print("\n📋 Category Performance:")
        for category, perf in summary['category_performance'].items():
            print(f"   {category}: {perf['accuracy']:.1f}% ({perf['correct']}/{perf['total']})")
        
        # Performance interpretation
        print(f"\n🎭 Performance Analysis:")
        if summary['accuracy'] >= 80:
            print("   🟢 Excellent performance!")
        elif summary['accuracy'] >= 60:
            print("   🟡 Good performance with room for improvement")
        else:
            print("   🔴 Performance needs significant improvement")
        
        if summary['avg_confidence'] >= 0.7:
            print("   🟢 High confidence in answers")
        elif summary['avg_confidence'] >= 0.5:
            print("   🟡 Moderate confidence in answers")
        else:
            print("   🔴 Low confidence - check retrieval quality")
    
    def _save_report(self, summary, results):
        """Save detailed evaluation report"""
        # Create reports directory
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        # Generate report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        provider = summary['llm_provider'].lower()
        report_file = reports_dir / f"rag_evaluation_{provider}_{timestamp}.json"
        
        # Prepare report data
        report_data = {
            'summary': summary,
            'detailed_results': results,
            'system_info': {
                'embedding_model': settings.EMBEDDING_MODEL,
                'chunk_size': settings.CHUNK_SIZE,
                'chunk_overlap': settings.CHUNK_OVERLAP,
                'similarity_threshold': settings.SIMILARITY_THRESHOLD
            }
        }
        
        # Save report
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Detailed report saved: {report_file}")
        
        # Also save markdown report
        md_report_file = reports_dir / f"rag_evaluation_{provider}_{timestamp}.md"
        self._save_markdown_report(md_report_file, summary, results)
        
        return str(report_file)
    
    def _save_markdown_report(self, file_path, summary, results):
        """Save markdown format report"""
        md_content = f"""# RAG System Evaluation Report

## Summary
- **LLM Provider:** {summary['llm_provider'].upper()}
- **Timestamp:** {summary['timestamp']}
- **Overall Accuracy:** {summary['accuracy']:.1f}% ({summary['correct_answers']}/{summary['total_tests']})
- **Average Confidence:** {summary['avg_confidence']:.2f}
- **Average Relevance:** {summary['avg_relevance']:.2f}

## Category Performance
"""
        
        for category, perf in summary['category_performance'].items():
            md_content += f"- **{category}:** {perf['accuracy']:.1f}% ({perf['correct']}/{perf['total']})\n"
        
        md_content += "\n## Detailed Results\n\n"
        
        for i, result in enumerate(results, 1):
            status = "✅ PASS" if result.get('is_correct', False) else "❌ FAIL"
            md_content += f"""### Test Case {i} {status}

**Question:** {result['question']}

**Expected:** {result['expected_answer']}

**Actual:** {result['actual_answer']}

**Metrics:**
- Confidence: {result.get('confidence', 0):.2f}
- Relevance: {result.get('avg_relevance_score', 0):.2f}
- Retrieved Docs: {result.get('retrieved_docs_count', 0)}
- Category: {result.get('category', 'unknown')}
- Language: {result.get('language_detected', 'unknown')}

---

"""
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"📄 Markdown report saved: {file_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate RAG System Performance')
    parser.add_argument('--provider', choices=['cohere', 'ollama'], 
                       help='LLM provider to use (overrides env setting)')
    parser.add_argument('--model', help='Model name for Ollama')
    parser.add_argument('--detailed', action='store_true', 
                       help='Show detailed output')
    parser.add_argument('--auto-build', action='store_true',
                       help='Automatically build knowledge base if needed')
    
    args = parser.parse_args()
    
    # Override settings if specified
    if args.provider:
        os.environ['LLM_PROVIDER'] = args.provider
        settings.LLM_PROVIDER = args.provider
    
    if args.model:
        os.environ['OLLAMA_MODEL'] = args.model
        settings.OLLAMA_MODEL = args.model
    
    print("🚀 Initializing RAG System for Evaluation")
    print(f"🔧 LLM Provider: {settings.LLM_PROVIDER.upper()}")
    
    try:
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline()
        
        # Check if knowledge base exists or auto-build
        if args.auto_build:
            print("🔨 Auto-building knowledge base...")
            doc_count = rag_pipeline.auto_build_knowledge_base()
            if not doc_count:
                print("❌ Failed to build knowledge base automatically")
                return 1
            print(f"✅ Knowledge base built with {doc_count} documents")
        else:
            # Try to load existing knowledge base
            try:
                # This will raise an exception if KB is not built
                rag_pipeline.query("test", include_history=False)
            except Exception:
                print("❌ Knowledge base not found. Use --auto-build flag or build it first.")
                print("   Run: python ui/streamlit_app.py to build knowledge base")
                return 1
        
        # Initialize evaluator
        evaluator = RAGEvaluator(rag_pipeline)
        
        # Run evaluation
        summary, results, report_file = evaluator.run_evaluation(detailed=args.detailed)
        
        print(f"\n✅ Evaluation completed!")
        print(f"📊 Final Score: {summary['accuracy']:.1f}%")
        print(f"📄 Report saved to: {report_file}")
        
        # Return appropriate exit code
        return 0 if summary['accuracy'] >= 60 else 1
        
    except KeyboardInterrupt:
        print("\n⏹️ Evaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())