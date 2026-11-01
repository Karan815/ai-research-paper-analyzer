"""Streamlit UI components"""
import streamlit as st
from typing import Optional
import os

from src.utils.helpers import (
    save_uploaded_file, get_file_size_mb, format_file_size,
    init_session_state, display_paper_info, display_summary,
    display_section_summaries, display_key_terms, display_readability,
    display_statistics
)
from src.core.analyzer import PaperAnalyzer
from src.core.summarizer import LocalTransformersSummarizer
from src.utils.export import export_analysis
from config.settings import config


@st.cache_resource
def get_summarizer(model_name: str, device: str, max_new_tokens: int = 200, num_beams: int = 1, chunk_overlap: int = 50):
    """Cache the summarizer model to avoid reloading on every rerun"""
    return LocalTransformersSummarizer(
        model_name=model_name, 
        device=device,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        chunk_overlap=chunk_overlap
    )


@st.cache_resource
def get_analyzer(model_name: str, device: str, fast_mode: bool, use_cache: bool, 
                 max_new_tokens: int = 200, num_beams: int = 1, chunk_overlap: int = 50):
    """Cache the analyzer with summarizer to avoid re-creating on every rerun"""
    summarizer = LocalTransformersSummarizer(
        model_name=model_name,
        device=device,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        chunk_overlap=chunk_overlap
    )
    return PaperAnalyzer(summarizer=summarizer, fast_mode=fast_mode, use_cache=use_cache)


def render_sidebar():
    """Render sidebar with settings"""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        st.subheader("Summarization Model")
        use_local = st.checkbox("Use Local Model", value=not config.USE_LLM_API)
        
        if use_local:
            model_options = [
                "sshleifer/distilbart-cnn-12-6",  # Fast (default)
                "facebook/bart-large-cnn",
                "facebook/bart-large-xsum",
                "google/pegasus-xsum",
                "t5-base",
            ]
            selected_model = st.selectbox("Model", model_options, index=0)
            device = st.radio("Device", ["cpu", "cuda"], index=0, help="CUDA requires GPU")
        else:
            st.info("Using LLM API (OpenAI/Anthropic)")
            selected_model = "API"
            device = "cpu"
        
        st.divider()
        
        # Performance options
        st.subheader("⚡ Performance")
        fast_mode = st.checkbox("Fast Mode", value=True, help="Use optimizations for speed (recommended)")
        include_sections = st.checkbox("Include Section Summaries", value=False, help="Generate summaries for each section (slower)")
        
        # Advanced performance options
        with st.expander("Advanced Options"):
            max_new_tokens = st.slider("Max New Tokens", 50, 500, 200, 50, help="Maximum tokens to generate (lower = faster)")
            num_beams = st.selectbox("Num Beams", [1, 2, 4], index=0, help="1=greedy (fastest), 4=beam search (better quality)")
            chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50, 25, help="Token overlap between chunks")
        
        st.divider()
        
        # Summary length
        st.subheader("Summary Length")
        max_len = st.slider("Max Length", 100, 1000, config.MAX_SUMMARY_LENGTH, 50)
        min_len = st.slider("Min Length", 50, 300, config.MIN_SUMMARY_LENGTH, 25)
        
        st.divider()
        
        # Export options
        st.subheader("Export Format")
        export_format = st.radio("Format", ["json", "markdown", "pdf"], horizontal=True)
        
        st.divider()
        
        # Info
        st.subheader("About")
        st.caption("AI Research Paper Analyzer")
        st.caption("Upload a PDF to extract, summarize, and analyze research papers.")
        
        return {
            'use_local': use_local,
            'model': selected_model,
            'device': device,
            'max_length': max_len,
            'min_length': min_len,
            'export_format': export_format,
            'fast_mode': fast_mode,
            'include_sections': include_sections,
            'max_new_tokens': max_new_tokens,
            'num_beams': num_beams,
            'chunk_overlap': chunk_overlap
        }


def render_file_uploader(settings) -> Optional[str]:
    """Render file uploader and return file path"""
    st.header("📄 Upload Research Paper")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help=f"Max file size: {config.MAX_FILE_SIZE_MB}MB"
    )
    
    if uploaded_file:
        # Check file size
        try:
            file_size_mb = get_file_size_mb(uploaded_file)
        except Exception as e:
            st.error(f"Could not determine file size: {e}")
            return None
            
        if file_size_mb > config.MAX_FILE_SIZE_MB:
            st.error(f"File too large: {file_size_mb:.1f}MB (max: {config.MAX_FILE_SIZE_MB}MB)")
            return None
        
        if file_size_mb == 0:
            st.error("Uploaded file is empty")
            return None
        
        st.success(f"✅ Uploaded: {uploaded_file.name} ({format_file_size(uploaded_file.size)})")
        
        # Save file
        try:
            file_path = save_uploaded_file(uploaded_file)
        except Exception as e:
            st.error(f"Failed to save uploaded file: {e}")
            return None
        
        return file_path
    
    return None


def render_analysis_results(analysis, settings):
    """Render analysis results"""
    paper = analysis.paper
    
    # Paper info
    st.header("📋 Paper Information")
    display_paper_info(paper)
    
    st.divider()
    
    # Overall summary
    st.header("📝 Overall Summary")
    display_summary(analysis.overall_summary)
    
    st.divider()
    
    # Section summaries
    if analysis.section_summaries:
        display_section_summaries(analysis.section_summaries)
        st.divider()
    
    # Key terms
    display_key_terms(analysis.key_terms)
    st.divider()
    
    # Readability
    display_readability(analysis.readability_scores)
    st.divider()
    
    # Statistics
    display_statistics(analysis.statistics)
    
    st.divider()
    
    # Export
    render_export_section(analysis, settings)


def render_export_section(analysis, settings):
    """Render export section"""
    st.header("💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    formats = [
        ("JSON", "json", "📄"),
        ("Markdown", "markdown", "📝"),
        ("PDF", "pdf", "📕")
    ]
    
    for i, (label, fmt, icon) in enumerate(formats):
        with [col1, col2, col3][i]:
            if st.button(f"{icon} Export {label}", use_container_width=True):
                try:
                    with st.spinner(f"Exporting to {label}..."):
                        path = export_analysis(analysis, fmt)
                    st.success(f"Exported to: {path}")
                    # Provide download
                    with open(path, "rb") as f:
                        st.download_button(
                            f"Download {label}",
                            f.read(),
                            file_name=os.path.basename(path),
                            mime=get_mime_type(fmt)
                        )
                except Exception as e:
                    st.error(f"Export failed: {e}")


def get_mime_type(fmt: str) -> str:
    """Get MIME type for format"""
    types = {
        "json": "application/json",
        "markdown": "text/markdown",
        "pdf": "application/pdf"
    }
    return types.get(fmt, "application/octet-stream")


def render_processing_status():
    """Render processing status"""
    if st.session_state.get('processing'):
        st.info("🔄 Processing paper... Please wait.")
        st.progress(0.5)


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title=config.APP_TITLE,
        page_icon=config.APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    
    # Header
    st.title(f"{config.APP_ICON} {config.APP_TITLE}")
    st.caption("Extract, summarize, and analyze research papers with AI")
    
    # Sidebar settings
    settings = render_sidebar()
    
    # File upload
    file_path = render_file_uploader(settings)
    
    # Process button
    if file_path and not st.session_state.processing:
        if st.button("🔍 Analyze Paper", type="primary", use_container_width=True):
            st.session_state.processing = True
            st.session_state.current_file = file_path
            st.rerun()
    
    # Process paper
    if st.session_state.processing and st.session_state.current_file:
        # Show detailed progress
        progress_container = st.container()
        with progress_container:
            st.info("🔄 Processing paper... Please wait.")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Use cached analyzer which handles everything with built-in caching
                model_name = settings.get('model', config.SUMMARIZATION_MODEL)
                device = settings.get('device', config.DEVICE)
                fast_mode = settings.get('fast_mode', True)
                use_cache = True
                include_sections = settings.get('include_sections', False)
                max_length = settings.get('max_length', config.MAX_SUMMARY_LENGTH)
                min_length = settings.get('min_length', config.MIN_SUMMARY_LENGTH)
                
                status_text.text("🤖 Loading analyzer...")
                progress_bar.progress(10)
                
                # Get cached analyzer (includes summarizer and PDF extractor with caching)
                analyzer = get_analyzer(
                    model_name=model_name,
                    device=device,
                    fast_mode=fast_mode,
                    use_cache=use_cache,
                    max_new_tokens=max_length,
                    num_beams=1,
                    chunk_overlap=50
                )
                
                progress_bar.progress(30)
                status_text.text("📄 Analyzing paper (with caching)...")
                
                # Run full analysis - uses cached PDF extraction and cached analysis results
                analysis = analyzer.analyze(
                    st.session_state.current_file,
                    include_sections=include_sections,
                    max_length=max_length,
                    min_length=min_length
                )
                
                progress_bar.progress(90)
                
                # Check if extraction produced meaningful content
                if not analysis.paper.full_text or len(analysis.paper.full_text.strip()) < 50:
                    st.error("Could not extract readable text from the PDF. The file may be scanned, encrypted, or corrupted.")
                    st.session_state.processing = False
                    return
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
                st.session_state.analysis_result = analysis
                st.session_state.processing = False
                st.rerun()
                
            except FileNotFoundError:
                st.error("The uploaded file could not be found. Please try uploading again.")
                st.session_state.processing = False
            except PermissionError:
                st.error("Permission denied accessing the file. Please try again.")
                st.session_state.processing = False
            except Exception as e:
                error_msg = str(e).lower()
                if "password" in error_msg or "encrypted" in error_msg:
                    st.error("The PDF is password-protected or encrypted. Please provide an unencrypted version.")
                elif "no text" in error_msg or "empty" in error_msg:
                    st.error("No extractable text found in the PDF. The file may be a scanned image.")
                elif "corrupt" in error_msg or "invalid" in error_msg:
                    st.error("The PDF file appears to be corrupted or invalid.")
                elif "out of memory" in error_msg or "cuda" in error_msg:
                    st.error("Out of memory. Try enabling 'Fast Mode' or using a smaller model.")
                else:
                    st.error(f"Analysis failed: {e}")
                st.session_state.processing = False
    
    # Display results
    if st.session_state.analysis_result:
        render_analysis_results(st.session_state.analysis_result, settings)
    
    # Footer
    st.divider()
    st.caption("Built with Streamlit, Transformers, and PyMuPDF")


if __name__ == "__main__":
    main()