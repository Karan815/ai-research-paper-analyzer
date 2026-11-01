# AI Research Paper Analyzer

A powerful tool for extracting, summarizing, and analyzing research papers using AI.

## Features

- 📄 **PDF Text Extraction** - Extract text, structure, and metadata from research papers
- 🤖 **AI Summarization** - Generate concise summaries using local models (BART, PEGASUS, T5) or LLM APIs (OpenAI GPT, Anthropic Claude)
- 📊 **Section-wise Analysis** - Get summaries for each paper section (Introduction, Methods, Results, etc.)
- 🔑 **Key Term Extraction** - Identify important terms and concepts using TF-IDF
- 📈 **Readability Metrics** - Flesch-Kincaid, Gunning Fog, and other readability scores
- 💾 **Multiple Export Formats** - JSON, Markdown, and PDF reports
- 🎨 **Interactive Web UI** - Built with Streamlit for easy use

## Installation

### 1. Clone and Setup

```bash
cd "ai research paper analyzer"
```

### 2. Create Virtual Environment (if not exists)

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows:**
```bash
.\venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. (Optional) Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

### 6. (Optional) Configure API Keys

Copy `.env.example` to `.env` and add your API keys for better summarization:

```bash
copy .env.example .env
# Edit .env with your keys
```

## Usage

### Run the Web Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Command Line Usage

```python
from src.core.analyzer import analyze_paper

# Analyze a paper
analysis = analyze_paper("path/to/paper.pdf")

# Access results
print(analysis.overall_summary.summary)
print(analysis.key_terms)

# Export
from src.utils.export import export_analysis
export_analysis(analysis, format="json")  # or "markdown", "pdf"
```

## Project Structure

```
ai research paper analyzer/
├── app.py                 # Main entry point
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── config/
│   └── settings.py       # Configuration management
├── src/
│   ├── core/
│   │   ├── pdf_extractor.py   # PDF text extraction
│   │   ├── summarizer.py      # Summarization backends
│   │   └── analyzer.py        # Main analysis pipeline
│   ├── ui/
│   │   └── app.py             # Streamlit UI
│   └── utils/
│       ├── export.py          # Export utilities
│       └── helpers.py         # UI helpers
└── data/
    ├── uploads/         # Uploaded PDFs
    └── exports/         # Generated reports
```

## Summarization Models

### Local Models (No API Key Required)
- `facebook/bart-large-cnn` - General purpose, good quality
- `facebook/bart-large-xsum` - Extreme summarization
- `google/pegasus-xsum` - Good for news/articles
- `t5-base` - Versatile, decent quality
- `sshleifer/distilbart-cnn-12-6` - Faster, smaller

### LLM APIs (Better Quality, Requires API Keys)
- **OpenAI**: GPT-4o-mini, GPT-4o, GPT-3.5-turbo
- **Anthropic**: Claude 3 Haiku, Sonnet, Opus

## Configuration

Edit `config/settings.py` to customize:
- Model selection
- Summary length limits
- Chunk sizes for long papers
- Export directory
- UI settings

## Requirements

- Python 3.9+
- 4GB+ RAM (for local models)
- GPU optional (CUDA for faster local inference)

## Troubleshooting

### Model Download Issues
First run downloads models (~1-2GB). Ensure internet connection.

### CUDA Out of Memory
Use smaller model (`distilbart-cnn-12-6`) or set device to CPU.

### PDF Extraction Issues
Some PDFs with complex layouts may not extract perfectly. Try different papers.

## License

MIT License