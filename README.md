# Multilingual RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system capable of understanding and responding to both English and Bengali queries, specifically designed for the HSC Bangla textbook.

## 🚀 Features

- **Multilingual Support**: Handles both Bengali and English queries seamlessly
- **Multi-Provider LLM Support**: Choose between Cohere and Ollama
- **Advanced Document Processing**: Intelligent PDF text extraction with OCR
- **Semantic Search**: Uses multilingual embeddings for accurate retrieval
- **Smart Caching**: Automatic caching and loading of processed documents
- **Conversation Memory**: Maintains both short-term and long-term memory
- **Web Interface**: Beautiful Streamlit UI for easy interaction
- **REST API**: FastAPI-based API for programmatic access
- **Comprehensive Evaluation**: Built-in evaluation suite with detailed metrics

## 📁 Project Structure

```
multilingual-rag-system/
├── src/                          # Core RAG components
│   ├── rag_pipeline.py          # Main RAG pipeline with caching
│   ├── document_processor.py    # OCR-based document processing
│   ├── embeddings.py           # Multilingual embeddings
│   ├── generator.py             # Multi-provider response generation
│   ├── query_processor.py      # Enhanced Bengali query processing
│   ├── enhanced_retriever.py   # Hybrid search retrieval
│   ├── vector_store.py         # ChromaDB vector storage
│   ├── language_detector.py    # Bengali/English language detection
│   └── cache_manager.py        # Knowledge base caching
├── api/                         # FastAPI application
├── ui/                          # Streamlit web interface
├── data/                        # Data storage
│   ├── raw/                    # Place PDF files here
│   ├── cache/                  # Processed data cache
│   └── vectorstore/            # Vector database
├── config/                      # Configuration files
├── reports/                     # Evaluation reports
├── evaluate.py                  # Standalone evaluation script
├── requirements.txt             # Python dependencies
├── .env.example                # Environment variables template
└── README.md                   # This file
```

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/multilingual-rag-system.git
cd multilingual-rag-system
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR (for Bengali text extraction)
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-ben

# macOS
brew install tesseract tesseract-lang

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

### 5. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 6. Add PDF files
```bash
# Place your HSC Bangla textbook PDF in data/raw/
mkdir -p data/raw
# Copy your PDF file to data/raw/HSC26-Bangla1st-Paper.pdf
```

## 🚀 Usage

### Option 1: Streamlit Web Interface (Recommended)
```bash
streamlit run ui/streamlit_app.py
```
Then open your browser to `http://localhost:8501`

### Option 2: FastAPI Server
```bash
python run.py --api
```
API will be available at `http://localhost:8000`
API documentation at `http://localhost:8000/docs`


## 🔧 Configuration

### LLM Provider Setup

#### Using Cohere (Recommended)
1. Get API key from [Cohere](https://cohere.ai/)
2. Set in `.env`:
```bash
LLM_PROVIDER=cohere
COHERE_API_KEY=your_cohere_api_key_here
```

#### Using Ollama (Local LLM)
1. Install [Ollama](https://ollama.ai/)
2. Pull a model: `ollama pull llama3:latest`
3. Set in `.env`:
```bash
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3:latest
```

Note: Ollama runs locally on your PC - no URL configuration needed.

## 📊 Evaluation

### Run Evaluation Script
```bash
# Basic evaluation
python run.py --evaluation --auto-build
```

### Expected Test Results
The system is evaluated on these key questions:

| Question (Bengali) | Expected Answer |
|-------------------|----------------|
| অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে? | শুম্ভুনাথ |
| কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে ? | মামাকে |
| বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল? | ১৫ বছর |

### Sample Evaluation Output
```
🧪 Starting RAG System Evaluation
📊 Provider: COHERE
🔬 Total test cases: 7

[1/7] Testing: অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?...
   ✅ PASS | Expected: শুম্ভুনাথ
   Got: শুম্ভুনাথ
   Confidence: 0.85
   Relevance Score: 0.72

📊 EVALUATION SUMMARY
🤖 LLM Provider: COHERE
📈 Overall Accuracy: 85.7% (6/7)
🎯 Average Confidence: 0.78
📄 Average Relevance: 0.69
```

## 🎯 Sample Queries and Outputs

### Bengali Queries
```
Q: অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?
A: শুম্ভুনাথ

Q: কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে ?  
A: মামাকে

Q: বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?
A: ১৫ বছর
```

## 🔧 Architecture & Technical Details

### Text Extraction Method
- **Primary**: pdf2images with Tesseract OCR for Bengali text
- **Preprocessing**: Image enhancement and noise reduction for better OCR accuracy

### Chunking Strategy
- **Method**: Recursive character splitting with Bengali sentence awareness
- **Size**: 1000 characters with 150 character overlap
- **Separators**: Respects Bengali sentence boundaries (।) and context

### Embedding Model
- **Model**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Benefits**: Excellent multilingual support, balanced performance/accuracy
- **Dimensions**: 384-dimensional embeddings

### Similarity Method
- **Primary**: Cosine similarity with ChromaDB vector storage
- **Enhanced**: Hybrid search combining semantic and keyword matching
- **Thresholds**: Dynamic thresholds based on language detection

### Caching System
- **Smart Caching**: Automatically caches processed documents and embeddings
- **Cache Validation**: MD5 hash-based validation for file changes
- **Quick Reload**: Instant loading from cache on subsequent runs

## 🚀 Performance Optimization

### For Better Results
1. **OCR Quality**: Ensure high-quality PDF scans for better text extraction
2. **Bengali Text**: Use proper Bengali fonts and encoding in source documents
3. **Query Formatting**: Use clear, specific questions for better retrieval
4. **Cache Management**: Clear cache when updating documents or settings

### Memory Usage
- **Embeddings**: ~50MB for typical textbook
- **Vector Store**: ~20MB for ChromaDB
- **Cache**: ~100MB total for processed data
