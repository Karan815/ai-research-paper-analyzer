"""PDF text extraction module - optimized for speed"""
import fitz  # pymupdf (fast)
import pdfplumber  # fallback for structure
from typing import List, Dict, Optional
from dataclasses import dataclass
import re
import hashlib
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class PaperSection:
    """Represents a section of a research paper"""
    title: str
    content: str
    page_start: int
    page_end: int
    level: int = 0  # heading level


@dataclass
class ExtractedPaper:
    """Container for extracted paper data"""
    title: str
    authors: List[str]
    abstract: str
    sections: List[PaperSection]
    full_text: str
    metadata: Dict
    page_count: int


# Global cache for extracted papers (file_hash -> ExtractedPaper)
_pdf_cache: Dict[str, ExtractedPaper] = {}


def _get_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of file for caching"""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


class PDFExtractor:
    """Extract text and structure from research papers - optimized for speed"""
    
    def __init__(self, use_fast_mode: bool = True, use_cache: bool = True):
        """
        Args:
            use_fast_mode: If True, use PyMuPDF (fitz) which is 10-50x faster than pdfplumber
            use_cache: If True, cache extracted results by file hash
        """
        self.use_fast_mode = use_fast_mode
        self.use_cache = use_cache
        self.section_patterns = [
            r'^\d+\.?\s+[A-Z][A-Za-z\s]+$',  # Numbered sections
            r'^[A-Z][A-Z\s]{2,}$',  # ALL CAPS headings
            r'^(Abstract|Introduction|Related Work|Background|Methodology|Methods|Approach|Experiments|Results|Discussion|Conclusion|References|Bibliography|Acknowledgments?|Appendix)\s*$',
        ]
    
    def extract(self, file_path: str) -> ExtractedPaper:
        """Extract all content from a PDF paper"""
        # Check cache first
        if self.use_cache:
            file_hash = _get_file_hash(file_path)
            if file_hash in _pdf_cache:
                logger.info(f"Cache hit for PDF: {file_path}")
                return _pdf_cache[file_hash]
        
        if self.use_fast_mode:
            # Use PyMuPDF (fitz) as primary - much faster
            try:
                result = self._extract_with_pymupdf(file_path)
            except Exception as e:
                error_msg = str(e).lower()
                if "password" in error_msg or "encrypted" in error_msg:
                    raise ValueError("PDF is password-protected or encrypted") from e
                # Fallback to pdfplumber for structure
                try:
                    result = self._extract_with_pdfplumber(file_path)
                except Exception as e2:
                    raise ValueError(f"Failed to extract text from PDF: {e2}") from e2
        else:
            # Original behavior: pdfplumber first, then PyMuPDF
            try:
                result = self._extract_with_pdfplumber(file_path)
            except Exception as e:
                error_msg = str(e).lower()
                if "password" in error_msg or "encrypted" in error_msg:
                    raise ValueError("PDF is password-protected or encrypted") from e
                try:
                    result = self._extract_with_pymupdf(file_path)
                except Exception as e2:
                    raise ValueError(f"Failed to extract text from PDF: {e2}") from e2
        
        # Cache the result
        if self.use_cache:
            file_hash = _get_file_hash(file_path)
            _pdf_cache[file_hash] = result
            logger.info(f"Cached PDF extraction for: {file_path}")
        
        return result
    
    def _extract_with_pdfplumber(self, file_path: str) -> ExtractedPaper:
        import os
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")
        
        with pdfplumber.open(file_path) as pdf:
            if len(pdf.pages) == 0:
                raise ValueError("PDF file has no pages")
            
            full_text = ""
            pages_text = []
            
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages_text.append(text)
                full_text += f"\n--- PAGE {i+1} ---\n{text}"
            
            # Check if any text was extracted
            if not full_text.strip():
                raise ValueError("No extractable text found in PDF (may be scanned/image-only)")
            
            # Extract metadata
            metadata = pdf.metadata or {}
            
            # Clean text before parsing metadata (remove page markers)
            clean_text_for_metadata = re.sub(r'--- PAGE \d+ ---\n', '', full_text)
            
            # Parse structure
            title, authors, abstract = self._parse_metadata(clean_text_for_metadata, metadata)
            sections = self._extract_sections(pages_text)
            
            return ExtractedPaper(
                title=title,
                authors=authors,
                abstract=abstract,
                sections=sections,
                full_text=full_text,
                metadata=metadata,
                page_count=len(pdf.pages)
            )
    
    def _extract_with_pymupdf(self, file_path: str) -> ExtractedPaper:
        import os
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")
        
        doc = fitz.open(file_path)
        
        if doc.page_count == 0:
            doc.close()
            raise ValueError("PDF file has no pages")
        
        # Check if encrypted
        if doc.needs_pass:
            doc.close()
            raise ValueError("PDF is password-protected or encrypted")
        
        full_text = ""
        pages_text = []
        
        for i, page in enumerate(doc):
            text = page.get_text()
            if text:
                pages_text.append(text)
                full_text += f"\n--- PAGE {i+1} ---\n{text}"
            else:
                pages_text.append("")
                full_text += f"\n--- PAGE {i+1} ---\n"
        
        metadata = doc.metadata
        doc.close()
        
        # Check if any text was extracted
        if not full_text.strip():
            raise ValueError("No extractable text found in PDF (may be scanned/image-only)")
        
        # Clean text before parsing metadata (remove page markers)
        clean_text_for_metadata = re.sub(r'--- PAGE \d+ ---\n', '', full_text)
        
        title, authors, abstract = self._parse_metadata(clean_text_for_metadata, metadata)
        sections = self._extract_sections(pages_text)
        
        return ExtractedPaper(
            title=title,
            authors=authors,
            abstract=abstract,
            sections=sections,
            full_text=full_text,
            metadata=metadata,
            page_count=len(pages_text)
        )
    
    def _parse_metadata(self, full_text: str, metadata: Dict) -> tuple:
        """Extract title, authors, abstract from text/metadata"""
        title = metadata.get("title", "") if metadata else ""
        authors = []
        
        # Try to extract from first few pages
        first_pages = full_text[:3000]
        
        # Extract title (usually first large text)
        if not title:
            lines = first_pages.split('\n')
            for line in lines[:10]:
                line = line.strip()
                if len(line) > 10 and len(line) < 200:
                    if not any(kw in line.lower() for kw in ['arxiv', 'doi', 'http', '@', 'university', 'institute']):
                        title = line
                        break
        
        # Extract authors (look for email patterns or affiliation lines)
        author_lines = []
        for line in first_pages.split('\n')[:20]:
            if '@' in line or any(kw in line.lower() for kw in ['university', 'institute', 'lab', 'department']):
                author_lines.append(line.strip())
        authors = author_lines[:5] if author_lines else []
        
        # Extract abstract
        abstract = self._extract_abstract(full_text)
        
        return title or "Unknown Title", authors, abstract
    
    def _extract_abstract(self, text: str) -> str:
        """Extract abstract from paper text"""
        # Find abstract section
        patterns = [
            r'Abstract\s*\n(.*?)(?:\n\s*(?:Introduction|Keywords|1\.|I\.)|$)',
            r'ABSTRACT\s*\n(.*?)(?:\n\s*(?:INTRODUCTION|KEYWORDS|1\.|I\.)|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                abstract = match.group(1).strip()
                if len(abstract) > 50:
                    return abstract[:2000]
        
        # Fallback: first 500 chars after "Abstract"
        idx = text.lower().find('abstract')
        if idx >= 0:
            return text[idx:idx+1500].strip()
        
        return "Abstract not found"
    
    def _extract_sections(self, pages_text: List[str]) -> List[PaperSection]:
        """Extract paper sections with hierarchy"""
        sections = []
        current_section = None
        current_content = []
        
        full_text = "\n".join(pages_text)
        lines = full_text.split('\n')
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            
            # Check if line is a section header
            if self._is_section_header(stripped):
                # Save previous section
                if current_section:
                    sections.append(PaperSection(
                        title=current_section,
                        content="\n".join(current_content).strip(),
                        page_start=self._estimate_page(i, pages_text),
                        page_end=self._estimate_page(i, pages_text),
                        level=self._get_header_level(stripped)
                    ))
                current_section = stripped
                current_content = []
            elif current_section:
                current_content.append(line)
        
        # Add last section
        if current_section:
            sections.append(PaperSection(
                title=current_section,
                content="\n".join(current_content).strip(),
                page_start=self._estimate_page(len(lines), pages_text),
                page_end=self._estimate_page(len(lines), pages_text),
                level=self._get_header_level(current_section)
            ))
        
        return sections
    
    def _is_section_header(self, line: str) -> bool:
        """Check if a line is a section header"""
        if len(line) > 100 or len(line) < 3:
            return False
        
        for pattern in self.section_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        return False
    
    def _get_header_level(self, header: str) -> int:
        """Determine heading level"""
        if re.match(r'^\d+\.\d+\.', header):
            return 2
        elif re.match(r'^\d+\.', header):
            return 1
        return 0
    
    def _estimate_page(self, line_idx: int, pages_text: List[str]) -> int:
        """Estimate page number from line index"""
        total_lines = sum(len(p.split('\n')) for p in pages_text)
        if total_lines == 0:
            return 1
        return max(1, int((line_idx / total_lines) * len(pages_text)) + 1)


def extract_paper(file_path: str) -> ExtractedPaper:
    """Convenience function to extract paper"""
    extractor = PDFExtractor()
    return extractor.extract(file_path)