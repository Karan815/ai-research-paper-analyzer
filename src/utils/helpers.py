"""Helper utilities"""
import os
import hashlib
import streamlit as st
from typing import Optional, BinaryIO
from pathlib import Path


def save_uploaded_file(uploaded_file, save_dir: str = "data/uploads") -> str:
    """Save uploaded file to disk and return path"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Read file content once
    file_content = uploaded_file.read()
    if not file_content:
        raise ValueError("Uploaded file is empty")
    
    # Generate unique filename
    file_hash = hashlib.md5(file_content).hexdigest()[:8]
    name, ext = os.path.splitext(uploaded_file.name)
    safe_name = f"{name}_{file_hash}{ext}"
    file_path = os.path.join(save_dir, safe_name)
    
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    # Reset file pointer for any subsequent reads
    uploaded_file.seek(0)
    
    return file_path


def get_file_size_mb(file_path_or_uploaded) -> float:
    """Get file size in MB from a file path or Streamlit UploadedFile object"""
    # Handle Streamlit UploadedFile object
    if hasattr(file_path_or_uploaded, 'size'):
        return file_path_or_uploaded.size / (1024 * 1024)
    # Handle file path string
    return os.path.getsize(file_path_or_uploaded) / (1024 * 1024)


def format_file_size(size_bytes: int) -> str:
    """Format file size human readable"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(' ', 1)[0] + "..."


def init_session_state():
    """Initialize Streamlit session state"""
    defaults = {
        'analysis_result': None,
        'current_file': None,
        'processing': False,
        'export_format': 'json'
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def display_paper_info(paper):
    """Display paper metadata in Streamlit"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Pages", paper.page_count)
    with col2:
        st.metric("Sections", len(paper.sections))
    with col3:
        word_count = len(paper.full_text.split())
        st.metric("Words", f"{word_count:,}")
    
    if paper.authors:
        st.write("**Authors:**", ", ".join(paper.authors))
    
    with st.expander("Abstract", expanded=True):
        st.write(paper.abstract)
    
    if paper.sections:
        with st.expander("Paper Structure"):
            for section in paper.sections:
                indent = "  " * section.level
                st.write(f"{indent}• {section.title} ({len(section.content.split())} words)")


def display_summary(summary, title: str = "Summary"):
    """Display summary in Streamlit"""
    st.subheader(title)
    
    # Check if this is an error/fallback summary
    is_fallback = "fallback" in summary.model_used.lower() or summary.confidence < 0.6
    is_error = summary.confidence == 0.0
    
    if is_error:
        st.error(f"❌ {summary.summary}")
        if summary.key_points:
            st.write("**Details:**")
            for point in summary.key_points:
                st.write(f"• {point}")
    elif is_fallback:
        st.warning("⚠️ Using extractive summarization (ML model unavailable)")
        st.write(summary.summary)
        
        if summary.key_points:
            st.write("**Key Points:**")
            for i, point in enumerate(summary.key_points, 1):
                st.write(f"{i}. {point}")
    else:
        st.write(summary.summary)
        
        if summary.key_points:
            st.write("**Key Points:**")
            for i, point in enumerate(summary.key_points, 1):
                st.write(f"{i}. {point}")
    
    # Show model info
    model_display = summary.model_used
    if is_fallback:
        model_display += " (fallback)"
    st.caption(f"Model: {model_display}")
    
    # Only show confidence if meaningful
    if summary.confidence > 0:
        st.caption(f"Confidence: {summary.confidence:.0%}")


def display_section_summaries(section_summaries):
    """Display section summaries in Streamlit"""
    if not section_summaries:
        return
    
    st.subheader("Section Summaries")
    
    for section_title, summary in section_summaries.items():
        with st.expander(f"{section_title}", expanded=False):
            st.write(summary.summary)
            if summary.key_points:
                st.write("**Key Points:**")
                for point in summary.key_points:
                    st.write(f"• {point}")


def display_key_terms(key_terms):
    """Display key terms as tags"""
    if not key_terms:
        return
    
    st.subheader("Key Terms")
    cols = st.columns(4)
    for i, term in enumerate(key_terms):
        cols[i % 4].markdown(f"`{term}`")


def display_readability(readability):
    """Display readability metrics"""
    if not readability:
        return
    
    st.subheader("Readability Scores")
    
    # Create a nice display
    cols = st.columns(len(readability))
    for i, (metric, score) in enumerate(readability.items()):
        with cols[i]:
            # Color code based on score
            if 'ease' in metric.lower():
                delta_color = "normal" if score > 60 else "inverse"
            elif 'grade' in metric.lower() or 'fog' in metric.lower():
                delta_color = "normal" if score < 12 else "inverse"
            else:
                delta_color = "off"
            
            st.metric(
                metric.replace('_', ' ').title(),
                f"{score:.1f}",
                delta_color=delta_color
            )


def display_statistics(stats):
    """Display paper statistics"""
    st.subheader("Statistics")
    
    cols = st.columns(3)
    metrics = [
        ("Word Count", f"{stats.get('word_count', 0):,}"),
        ("Reading Time", f"{stats.get('estimated_reading_time_minutes', 0)} min"),
        ("Sections", stats.get('section_count', 0)),
    ]
    
    for i, (label, value) in enumerate(metrics):
        cols[i].metric(label, value)