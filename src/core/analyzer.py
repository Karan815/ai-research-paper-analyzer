"""Paper analysis module - combines extraction, summarization, and insights"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import json
import hashlib
import logging
from datetime import datetime

from .pdf_extractor import PDFExtractor, ExtractedPaper, PaperSection, _get_file_hash
from .summarizer import SummarizerFactory, SummaryResult, SummarizerBase
from config.settings import config

logger = logging.getLogger(__name__)


def _make_cache_key(file_path: str, model_name: str, include_sections: bool, 
                    max_length: int, min_length: int, fast_mode: bool) -> str:
    """Create a deterministic cache key from file hash and analysis settings.
    
    Uses the file's SHA256 hash combined with analysis parameters to ensure
    identical PDFs with identical settings produce the same key.
    """
    file_hash = _get_file_hash(file_path)
    settings_str = f"{model_name}|{include_sections}|{max_length}|{min_length}|{fast_mode}"
    settings_hash = hashlib.sha256(settings_str.encode()).hexdigest()[:16]
    return f"{file_hash}|{settings_hash}"


@dataclass
class PaperAnalysis:
    """Complete analysis of a research paper"""
    paper: ExtractedPaper
    overall_summary: SummaryResult
    section_summaries: Dict[str, SummaryResult]
    key_terms: List[str]
    readability_scores: Dict[str, float]
    statistics: Dict[str, Any]
    analyzed_at: str
    
    def to_dict(self) -> Dict:
        return {
            "paper": {
                "title": self.paper.title,
                "authors": self.paper.authors,
                "abstract": self.paper.abstract,
                "page_count": self.paper.page_count,
                "sections": [
                    {"title": s.title, "content": s.content[:500], "level": s.level}
                    for s in self.paper.sections
                ]
            },
            "overall_summary": {
                "summary": self.overall_summary.summary,
                "key_points": self.overall_summary.key_points,
                "model": self.overall_summary.model_used
            },
            "section_summaries": {
                k: {"summary": v.summary, "key_points": v.key_points, "model": v.model_used}
                for k, v in self.section_summaries.items()
            },
            "key_terms": self.key_terms,
            "readability": self.readability_scores,
            "statistics": self.statistics,
            "analyzed_at": self.analyzed_at
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# Global cache for analysis results (cache_key -> PaperAnalysis)
_analysis_cache: Dict[str, "PaperAnalysis"] = {}


class PaperAnalyzer:
    """Main analyzer class that orchestrates the full pipeline"""
    
    def __init__(self, summarizer: Optional[SummarizerBase] = None, fast_mode: bool = True, use_cache: bool = True):
        """
        Args:
            summarizer: Custom summarizer instance
            fast_mode: If True, use optimizations for speed (default: True)
            use_cache: If True, cache analysis results (default: True)
        """
        self.extractor = PDFExtractor(use_fast_mode=fast_mode, use_cache=use_cache)
        self.summarizer = summarizer or SummarizerFactory.create(config)
        self.fast_mode = fast_mode
        self.use_cache = use_cache
    
    def analyze(self, file_path: str, include_sections: bool = False, 
                max_length: Optional[int] = None, min_length: Optional[int] = None) -> PaperAnalysis:
        """Run complete analysis pipeline on a paper
        
        Args:
            file_path: Path to PDF file
            include_sections: If True, generate section summaries (slower)
            max_length: Maximum summary length (overrides config)
            min_length: Minimum summary length (overrides config)
        """
        # Create cache key
        model_name = getattr(self.summarizer, 'model_name', 'unknown')
        max_len = max_length or config.MAX_SUMMARY_LENGTH
        min_len = min_length or config.MIN_SUMMARY_LENGTH
        
        if self.use_cache:
            cache_key = _make_cache_key(file_path, model_name, include_sections, 
                                        max_len, min_len, self.fast_mode)
            if cache_key in _analysis_cache:
                logger.info(f"Cache hit for analysis: {file_path}")
                return _analysis_cache[cache_key]
        
        # Extract paper content
        paper = self.extractor.extract(file_path)
        
        # Generate overall summary
        overall_summary = self.summarizer.summarize(
            paper.full_text,
            max_length=max_len,
            min_length=min_len
        )
        
        # Generate section summaries (optional, slower)
        section_summaries = {}
        if include_sections:
            sections_data = [
                {"title": s.title, "content": s.content}
                for s in paper.sections
                if len(s.content) > 200
            ]
            section_summaries = self.summarizer.summarize_sections(sections_data)
        
        # Extract key terms (fast mode by default)
        key_terms = self._extract_key_terms(paper.full_text, use_fast=self.fast_mode)
        
        # Calculate readability (fast mode by default)
        readability = self._calculate_readability(paper.full_text, use_fast=self.fast_mode)
        
        # Compute statistics
        stats = self._compute_statistics(paper)
        
        analysis = PaperAnalysis(
            paper=paper,
            overall_summary=overall_summary,
            section_summaries=section_summaries,
            key_terms=key_terms,
            readability_scores=readability,
            statistics=stats,
            analyzed_at=datetime.now().isoformat()
        )
        
        # Cache the result
        if self.use_cache:
            _analysis_cache[cache_key] = analysis
            logger.info(f"Cached analysis for: {file_path}")
        
        return analysis
    
    def _extract_key_terms(self, text: str, top_n: int = 20, use_fast: bool = True) -> List[str]:
        """Extract key terms - fast mode uses frequency-based, slow mode uses TF-IDF"""
        if use_fast:
            return self._simple_key_terms(text, top_n)
        
        # Slow but more accurate TF-IDF method
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            import nltk
            
            # Download required NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
            
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
            
            # Add domain-specific stop words
            stop_words.update(['paper', 'study', 'research', 'method', 'result', 'figure', 'table', 'section'])
            
            vectorizer = TfidfVectorizer(
                max_features=top_n,
                stop_words=list(stop_words),
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            # Split into sentences for better term extraction
            sentences = text.split('. ')
            if len(sentences) < 2:
                sentences = [text]
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.sum(axis=0).A1
            
            term_scores = list(zip(feature_names, scores))
            term_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [term for term, score in term_scores[:top_n] if score > 0]
        except Exception as e:
            # Fallback: simple frequency-based
            return self._simple_key_terms(text, top_n)
    
    def _simple_key_terms(self, text: str, top_n: int) -> List[str]:
        """Simple term extraction fallback"""
        import re
        from collections import Counter
        
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'man', 'put', 'say', 'she', 'too', 'use', 'paper', 'study', 'research', 'method', 'result', 'figure', 'table', 'section', 'using', 'used', 'based', 'approach', 'proposed', 'show', 'shown', 'also', 'well'}
        
        filtered = [w for w in words if w not in stop_words]
        freq = Counter(filtered)
        return [w for w, _ in freq.most_common(top_n)]
    
    def _calculate_readability(self, text: str, use_fast: bool = True) -> Dict[str, float]:
        """Calculate readability metrics - fast mode computes only Flesch-Kincaid"""
        if use_fast:
            # Fast: only compute the most useful metric
            try:
                import textstat
                return {
                    "flesch_reading_ease": textstat.flesch_reading_ease(text),
                    "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
                }
            except Exception:
                return {}
        
        # Slow: compute all metrics
        try:
            import textstat
            return {
                "flesch_reading_ease": textstat.flesch_reading_ease(text),
                "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
                "gunning_fog": textstat.gunning_fog(text),
                "automated_readability_index": textstat.automated_readability_index(text),
                "coleman_liau_index": textstat.coleman_liau_index(text),
            }
        except Exception:
            return {}
    
    def _compute_statistics(self, paper: ExtractedPaper) -> Dict[str, Any]:
        """Compute paper statistics"""
        word_count = len(paper.full_text.split())
        char_count = len(paper.full_text)
        section_count = len(paper.sections)
        
        # Estimate reading time (avg 250 words/min)
        reading_time_min = max(1, word_count // 250)
        
        return {
            "word_count": word_count,
            "character_count": char_count,
            "page_count": paper.page_count,
            "section_count": section_count,
            "estimated_reading_time_minutes": reading_time_min,
            "has_abstract": bool(paper.abstract and len(paper.abstract) > 50),
            "title": paper.title
        }


def analyze_paper(file_path: str, summarizer: Optional[SummarizerBase] = None) -> PaperAnalysis:
    """Convenience function to analyze a paper"""
    analyzer = PaperAnalyzer(summarizer)
    return analyzer.analyze(file_path)