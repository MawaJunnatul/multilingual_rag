import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_pipeline import RAGPipeline
from config.settings import settings
import json

# Page configuration
st.set_page_config(
    page_title="Multilingual RAG System",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'knowledge_base_built' not in st.session_state:
    st.session_state.knowledge_base_built = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def main():
    st.title("📚 Multilingual RAG System")
    st.markdown("**বহুভাষিক তথ্য পুনরুদ্ধার ও উত্তর প্রদান সিস্টেম**")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 System Configuration")
        
        # Check environment variables
        cohere_key = os.getenv("COHERE_API_KEY")
        
        st.info(f"**LLM Provider:** {settings.LLM_PROVIDER.upper()}")
        
        if settings.LLM_PROVIDER == "cohere":
            if cohere_key:
                st.success("✅ Cohere API Key found")
            else:
                st.error("❌ COHERE_API_KEY not found in environment")
                st.code("export COHERE_API_KEY=your_key_here")
        else:
            st.info(f"**Ollama Model:** {settings.OLLAMA_MODEL}")
            st.info("🖥️ Using local Ollama installation")
        
        st.markdown("---")
        
        # Cache Management
        st.header("💾 Cache Management")
        
        if st.session_state.rag_pipeline:
            cache_info = st.session_state.rag_pipeline.cache_manager.get_cache_info()
            
            if cache_info['cached']:
                st.success("✅ Cache Available")
                st.text(f"Documents: {cache_info['document_count']}")
                st.text(f"Size: {cache_info['cache_size']}")
                
                if st.button("🗑️ Clear Cache"):
                    if st.session_state.rag_pipeline.clear_cache():
                        st.success("Cache cleared successfully!")
                        st.session_state.knowledge_base_built = False
                        st.rerun()
                    else:
                        st.error("Failed to clear cache")
            else:
                st.info("📦 No cache available")
        
        st.markdown("---")
        
        # Knowledge Base Management
        st.header("📖 Knowledge Base")
        
        # Check for PDF files in data/raw
        raw_data_path = "data/raw"
        pdf_files = []
        if os.path.exists(raw_data_path):
            import glob
            pdf_files = glob.glob(os.path.join(raw_data_path, "*.pdf"))
        
        if pdf_files:
            st.success(f"✅ Found {len(pdf_files)} PDF file(s)")
            for pdf_file in pdf_files:
                st.text(f"📄 {os.path.basename(pdf_file)}")
            
            if not st.session_state.knowledge_base_built:
                if st.button("🚀 Build Knowledge Base"):
                    if not st.session_state.rag_pipeline:
                        st.session_state.rag_pipeline = RAGPipeline()
                    
                    with st.spinner("Building knowledge base..."):
                        try:
                            doc_count = st.session_state.rag_pipeline.auto_build_knowledge_base()
                            if doc_count:
                                st.session_state.knowledge_base_built = True
                                st.success(f"✅ Knowledge base built with {doc_count} chunks!")
                                st.rerun()
                            else:
                                st.error("Failed to build knowledge base")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        else:
            st.warning(f"⚠️ No PDF files found in {raw_data_path}")
            st.info("Please add PDF files to the data/raw directory")
        
        # Upload functionality as backup
        st.markdown("---")
        st.header("📤 Upload PDF")
        uploaded_file = st.file_uploader(
            "Upload Bengali PDF",
            type=['pdf'],
            help="Upload HSC Bangla textbook PDF"
        )
        
        if uploaded_file:
            if st.button("📁 Save & Build from Upload"):
                # Save uploaded file
                os.makedirs("data/raw", exist_ok=True)
                pdf_path = f"data/raw/{uploaded_file.name}"
                
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Initialize and build
                if not st.session_state.rag_pipeline:
                    st.session_state.rag_pipeline = RAGPipeline()
                
                with st.spinner("Processing uploaded PDF..."):
                    try:
                        doc_count = st.session_state.rag_pipeline.build_knowledge_base(pdf_path)
                        st.session_state.knowledge_base_built = True
                        st.success(f"✅ Knowledge base built with {doc_count} chunks!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # System Statistics
        if st.session_state.knowledge_base_built and st.session_state.rag_pipeline:
            st.markdown("---")
            st.header("📊 System Stats")
            stats = st.session_state.rag_pipeline.get_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", stats['knowledge_base']['total_documents'])
                st.metric("Queries", stats['conversation']['total_queries'])
            with col2:
                st.metric("Success Rate", stats['conversation']['success_rate'])
                st.metric("LLM Provider", stats['models']['llm_provider'].upper())
            
            if stats['knowledge_base']['cache_used']:
                st.success("🚀 Loaded from cache")
            else:
                st.info("🔨 Built from scratch")
    
    # Main content area
    if not st.session_state.knowledge_base_built:
        st.info("📁 Please build the knowledge base to start asking questions.")
        
        # Sample questions
        st.markdown("### 🎯 Sample Test Questions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Bengali Questions:**")
            bengali_questions = [
                "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
                "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে ?",
                "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
                "অনুপমের বয়স কত বছর?",
                "অনুপমের বাবা কী কাজ করতেন?"
            ]
            
            for i, question in enumerate(bengali_questions, 1):
                st.text(f"{i}. {question}")
        
        with col2:
            st.markdown("**Expected Answers:**")
            expected_answers = [
                "শুম্ভুনাথ",
                "মামাকে", 
                "১৫ বছর",
                "সাতাই",
                "ওকালতি"
            ]
            
            for i, answer in enumerate(expected_answers, 1):
                st.text(f"{i}. {answer}")
        
        return
    
    # Chat interface
    st.header("💬 Chat Interface")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("human"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                
                # Show metadata if available
                if "metadata" in message:
                    with st.expander("📊 Details"):
                        metadata = message["metadata"]
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Confidence", f"{metadata['confidence']:.2f}")
                        with col2:
                            st.metric("Docs", metadata['relevant_docs_count'])
                        with col3:
                            st.metric("Language", metadata.get('language_info', {}).get('language', 'unknown'))
                        with col4:
                            st.metric("Provider", metadata.get('llm_provider', 'unknown').upper())
    
    # Chat input
    if prompt := st.chat_input("Ask a question in Bengali or English..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("human"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                result = st.session_state.rag_pipeline.query(prompt)
                
                st.write(result["answer"])
                
                # Add assistant message with metadata
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "metadata": {
                        "confidence": result["confidence"],
                        "relevant_docs_count": result["relevant_docs_count"],
                        "language_info": result["language_info"],
                        "llm_provider": result["llm_provider"]
                    }
                })
                
                # Show details
                with st.expander("📊 Details"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Confidence", f"{result['confidence']:.2f}")
                    with col2:
                        st.metric("Docs", result["relevant_docs_count"])
                    with col3:
                        st.metric("Language", result["language_info"]["language"])
                    with col4:
                        st.metric("Provider", result["llm_provider"].upper())
                    
                    # Show top retrieved docs
                    if result['retrieved_docs']:
                        st.write("**Top Retrieved Content:**")
                        for i, doc in enumerate(result['retrieved_docs'][:3]):
                            score = doc.get('combined_score', doc.get('score', 0))
                            method = doc.get('search_method', 'semantic')
                            content = doc['document'][:200] + "..."
                            st.text_area(
                                f"Chunk {i+1} (Score: {score:.2f}, Method: {method})",
                                content,
                                height=100
                            )
    
    # Control buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            if st.session_state.rag_pipeline:
                st.session_state.rag_pipeline.clear_history()
            st.rerun()
    
    with col2:
        if st.button("🧪 Run Tests"):
            if st.session_state.rag_pipeline:
                with st.spinner("Running test cases..."):
                    test_results = st.session_state.rag_pipeline.test_common_questions()
                    
                    st.write("**Test Results:**")
                    st.metric("Accuracy", test_results['accuracy'])
                    st.metric("Provider", test_results['llm_provider'].upper())
                    
                    for result in test_results['test_results']:
                        status = "✅" if result['is_correct'] else "❌"
                        st.write(f"{status} **Q:** {result['question']}")
                        st.write(f"   **Expected:** {result['expected']}")
                        st.write(f"   **Got:** {result['actual']}")
                        st.write(f"   **Confidence:** {result['confidence']:.2f}")
                        st.write("---")
    
    with col3:
        if st.button("📈 Show Stats"):
            if st.session_state.rag_pipeline:
                stats = st.session_state.rag_pipeline.get_stats()
                st.json(stats)

if __name__ == "__main__":
    main()