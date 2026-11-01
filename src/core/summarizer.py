"""Text summarization module with multiple backends"""
import os
import logging
import hashlib
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class SummaryResult:
    """Result of summarization"""
    summary: str
    key_points: List[str]
    model_used: str
    confidence: float = 0.0


class SummarizerBase(ABC):
    """Abstract base class for summarizers"""
    
    @abstractmethod
    def summarize(self, text: str, max_length: int = 500, min_length: int = 100) -> SummaryResult:
        pass
    
    @abstractmethod
    def summarize_sections(self, sections: List[Dict], max_length: int = 300) -> Dict[str, SummaryResult]:
        pass


class LocalTransformersSummarizer(SummarizerBase):
    """Local summarization using Hugging Face transformers (Seq2Seq) - optimized for speed"""
    
    def __init__(
        self, 
        model_name: str = "sshleifer/distilbart-cnn-12-6", 
        device: str = "auto",
        max_new_tokens: int = 200,
        num_beams: int = 1,
        chunk_overlap: int = 50,
        use_compile: bool = False
    ):
        """
        Args:
            model_name: HuggingFace model name
            device: "auto", "cpu", or "cuda"
            max_new_tokens: Maximum new tokens to generate (faster than max_length)
            num_beams: Number of beams for beam search (1 = greedy, faster)
            chunk_overlap: Token overlap between chunks
            use_compile: Use torch.compile for faster inference (PyTorch 2.0+)
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.chunk_overlap = chunk_overlap
        self.use_compile = use_compile
        self._model = None
        self._tokenizer = None
        self._device = None
        self._load_model()
    
    def _load_model(self):
        """Load the Seq2Seq model and tokenizer for summarization with optimizations"""
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import torch
            
            # Determine device
            if self.device == "auto":
                if torch.cuda.is_available():
                    self._device = torch.device("cuda")
                    torch_dtype = torch.float16
                else:
                    self._device = torch.device("cpu")
                    torch_dtype = torch.float32
            elif self.device == "cuda" and torch.cuda.is_available():
                self._device = torch.device("cuda")
                torch_dtype = torch.float16
            else:
                self._device = torch.device("cpu")
                torch_dtype = torch.float32
            
            logger.info(f"Loading model {self.model_name} on {self._device}...")
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model with optimizations
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                dtype=torch_dtype,
                low_cpu_mem_usage=True,
                device_map="auto" if self._device.type == "cuda" else None
            )
            
            # Set forced_bos_token_id for proper generation (suppresses warning)
            if hasattr(self._model.config, 'forced_bos_token_id'):
                if self._model.config.forced_bos_token_id is None:
                    self._model.config.forced_bos_token_id = (
                        self._tokenizer.bos_token_id or self._tokenizer.cls_token_id or 0
                    )
            
            # Also set in generation_config if available
            if hasattr(self._model, 'generation_config') and self._model.generation_config is not None:
                if self._model.generation_config.forced_bos_token_id is None:
                    self._model.generation_config.forced_bos_token_id = self._model.config.forced_bos_token_id
            
            if self._device.type == "cpu":
                self._model = self._model.to(self._device)
            
            self._model.eval()
            
            # Disable gradient computation for inference
            for param in self._model.parameters():
                param.requires_grad = False
            
            # Apply torch.compile for faster inference (PyTorch 2.0+)
            if self.use_compile and hasattr(torch, 'compile') and self._device.type == "cuda":
                try:
                    self._model = torch.compile(self._model, mode="reduce-overhead")
                    logger.info("Applied torch.compile for faster inference")
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}")
            
            logger.info(f"Successfully loaded summarization model: {self.model_name}")
            self._model_loaded = True
            self._load_error = None
            
        except Exception as e:
            error_msg = f"Failed to load model '{self.model_name}': {e}"
            logger.error(error_msg)
            self._load_error = str(e)
            self._model = None
            self._tokenizer = None
            self._model_loaded = False
    
    def summarize(self, text: str, max_length: int = 500, min_length: int = 100) -> SummaryResult:
        if not self._model_loaded or self._model is None or self._tokenizer is None:
            logger.warning(f"Model not loaded, using extractive fallback: {getattr(self, '_load_error', 'Unknown error')}")
            return self._extractive_summarize(text, max_length, min_length)
        
        # Clean and prepare text
        text = self._clean_text(text)
        if not text or len(text.strip()) < 50:
            return SummaryResult(
                summary="Input text too short to summarize.",
                key_points=[],
                model_used=f"{self.model_name} (extractive fallback)",
                confidence=0.1
            )
        
        # Chunk text by tokens
        chunks = self._chunk_text_by_tokens(text)
        logger.info(f"Processing {len(chunks)} chunks for summarization")
        
        summaries = []
        for i, chunk in enumerate(chunks):
            try:
                summary = self._summarize_chunk(chunk, max_length, min_length)
                summaries.append(summary)
            except Exception as e:
                logger.error(f"Error summarizing chunk {i}: {e}")
                # Add fallback for this chunk
                summaries.append(self._extractive_summarize(chunk, max_length, min_length).summary)
        
        # Combine summaries
        if len(summaries) == 1:
            combined = summaries[0]
        else:
            # For multiple chunks, concatenate and truncate instead of re-summarizing (faster)
            combined = " ".join(summaries)
            if len(combined) > max_length * 2:
                combined = combined[:max_length * 2]
        
        key_points = self._extract_key_points(combined)
        
        return SummaryResult(
            summary=combined,
            key_points=key_points,
            model_used=self.model_name,
            confidence=0.85
        )
    
    def _summarize_chunk(self, text: str, max_length: int, min_length: int) -> str:
        """Summarize a single chunk using the Seq2Seq model - optimized for speed"""
        import torch
        
        # Prepare input - use tokenizer's model_max_length if available
        model_max_length = getattr(self._tokenizer, 'model_max_length', 1024)
        if model_max_length > 10000:  # Some tokenizers have very large max_length
            model_max_length = 1024
        
        inputs = self._tokenizer(
            text,
            max_length=model_max_length,
            truncation=True,
            return_tensors="pt",
            padding=True
        ).to(self._device)
        
        # Get forced_bos_token_id (suppresses warning)
        forced_bos_token_id = getattr(self._model.config, 'forced_bos_token_id', None)
        if forced_bos_token_id is None:
            forced_bos_token_id = self._tokenizer.bos_token_id or self._tokenizer.cls_token_id or 0
        
        # Calculate max_new_tokens based on input length and desired output
        input_length = inputs["input_ids"].shape[1]
        # Reserve tokens for output, use max_new_tokens instead of max_length
        effective_max_new_tokens = min(self.max_new_tokens, max_length)
        
        # Generate summary with optimized parameters - use inference_mode for speed
        with torch.inference_mode():
            summary_ids = self._model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=effective_max_new_tokens,
                min_length=min(min_length, effective_max_new_tokens // 2),
                num_beams=self.num_beams,
                length_penalty=1.0,  # Reduced from 2.0 for speed
                early_stopping=True,
                no_repeat_ngram_size=3,
                do_sample=False,
                use_cache=True,
                forced_bos_token_id=forced_bos_token_id
            )
        
        # Decode
        summary = self._tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary.strip()
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text from PDF"""
        import re
        # Remove page markers
        text = re.sub(r'--- PAGE \d+ ---', '', text)
        # Remove PDF artifacts: page numbers, headers, footers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # Standalone page numbers
        text = re.sub(r'[•◦▪▫·]\s*', '', text)  # Bullet points at line starts
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove repeated headers/footers (simple heuristic)
        lines = text.split('\n')
        seen = set()
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip very short lines that are likely artifacts
            if len(line) < 3:
                continue
            if line and line not in seen:
                seen.add(line)
                cleaned_lines.append(line)
        return ' '.join(cleaned_lines).strip()
    
    def _extractive_summarize(self, text: str, max_length: int = 500, min_length: int = 100) -> SummaryResult:
        """Fallback extractive summarization - no ML model required"""
        import re
        from collections import Counter
        
        # Clean text first
        text = self._clean_text(text)
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return SummaryResult(
                summary="Could not extract sentences from text.",
                key_points=[],
                model_used=f"{self.model_name} (extractive fallback)",
                confidence=0.3
            )
        
        # Score sentences by word frequency (simple extractive approach)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'man', 'put', 'say', 'she', 'too', 'use', 'paper', 'study', 'research', 'method', 'result', 'figure', 'table', 'section', 'using', 'used', 'based', 'approach', 'proposed', 'show', 'shown', 'also', 'well', 'this', 'that', 'with', 'from', 'have', 'been', 'were', 'their', 'there', 'which', 'their', 'about', 'would', 'could', 'should', 'into', 'more', 'some', 'than', 'then', 'them', 'these', 'they', 'will', 'your', 'when', 'what', 'where', 'while', 'after', 'before', 'during', 'under', 'over', 'such', 'each', 'both', 'few', 'most', 'other', 'another', 'many', 'much', 'very', 'much', 'into', 'upon', 'within', 'without', 'through', 'between', 'among', 'across', 'against', 'around', 'behind', 'beyond', 'since', 'until', 'unless', 'because', 'although', 'though', 'whether', 'either', 'neither', 'nor', 'once', 'here', 'there', 'where', 'why', 'how', 'what', 'who', 'whom', 'whose', 'which', 'that', 'this', 'these', 'those', 'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for', 'by', 'with', 'as', 'is', 'it', 'be', 'or', 'if', 'we', 'he', 'she', 'do', 'my', 'me', 'no', 'so', 'up', 'am'}
        filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
        word_freq = Counter(filtered_words)
        
        # Score sentences
        sentence_scores = []
        for sent in sentences:
            sent_words = re.findall(r'\b[a-zA-Z]{3,}\b', sent.lower())
            score = sum(word_freq.get(w, 0) for w in sent_words if w not in stop_words)
            sentence_scores.append((score, sent))
        
        # Sort by score and take top sentences
        sentence_scores.sort(reverse=True, key=lambda x: x[0])
        
        # Build summary within length limits
        summary_sentences = []
        current_length = 0
        for score, sent in sentence_scores:
            if current_length + len(sent) <= max_length:
                summary_sentences.append(sent)
                current_length += len(sent)
            else:
                break
        
        # Ensure minimum length
        if current_length < min_length and len(summary_sentences) < len(sentence_scores):
            for score, sent in sentence_scores:
                if sent not in summary_sentences:
                    summary_sentences.append(sent)
                    current_length += len(sent)
                    if current_length >= min_length:
                        break
        
        summary = " ".join(summary_sentences)
        key_points = summary_sentences[:5]
        
        return SummaryResult(
            summary=summary,
            key_points=key_points,
            model_used=f"{self.model_name} (extractive fallback)",
            confidence=0.5
        )
    
    def summarize_sections(self, sections: List[Dict], max_length: int = 300) -> Dict[str, SummaryResult]:
        """Optimized: only summarize first 3 sections to save time"""
        results = {}
        # Limit to first 3 major sections for speed
        for section in sections[:3]:
            content = section.get('content', '')
            if len(content) > 200:  # Only summarize substantial sections
                results[section['title']] = self.summarize(content, max_length, 50)
        return results
    
    def _chunk_text_by_tokens(self, text: str, max_tokens: int = 1024) -> List[str]:
        """Split text into chunks by tokens with overlap for better context"""
        if not self._tokenizer:
            return self._chunk_text(text, max_tokens * 4)  # Rough word estimate
        
        tokens = self._tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= max_tokens:
            return [text]
        
        chunks = []
        step = max_tokens - self.chunk_overlap
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self._tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
            if i + max_tokens >= len(tokens):
                break
        
        return chunks
    
    def _chunk_text(self, text: str, max_chunk: int) -> List[str]:
        """Split text into chunks by words (fallback)"""
        words = text.split()
        chunks = []
        current = []
        current_len = 0
        
        for word in words:
            current_len += len(word) + 1
            if current_len > max_chunk:
                chunks.append(" ".join(current))
                current = [word]
                current_len = len(word)
            else:
                current.append(word)
        
        if current:
            chunks.append(" ".join(current))
        
        return chunks if chunks else [text[:max_chunk]]
    
    def _extract_key_points(self, summary: str) -> List[str]:
        """Extract key points from summary"""
        sentences = summary.split('. ')
        return [s.strip() + '.' for s in sentences[:5] if len(s) > 20]


class OpenAISummarizer(SummarizerBase):
    """Summarization using OpenAI API"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self._client = None
        self._init_client()
    
    def _init_client(self):
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Failed to init OpenAI client: {e}")
    
    def summarize(self, text: str, max_length: int = 500, min_length: int = 100) -> SummaryResult:
        if not self._client:
            return SummaryResult(
                summary="OpenAI client not initialized",
                key_points=[],
                model_used=self.model,
                confidence=0.0
            )
        
        prompt = f"""Summarize this research paper section in {max_length} words.
        Provide a clear summary and 3-5 key points.
        
        Text: {text[:8000]}"""
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert research paper summarizer."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_length + 200,
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            key_points = self._parse_key_points(content)
            
            return SummaryResult(
                summary=content,
                key_points=key_points,
                model_used=self.model,
                confidence=0.95
            )
        except Exception as e:
            logger.error(f"OpenAI summarization error: {e}")
            return SummaryResult(
                summary=f"Error: {str(e)}",
                key_points=[],
                model_used=self.model,
                confidence=0.0
            )
    
    def summarize_sections(self, sections: List[Dict], max_length: int = 300) -> Dict[str, SummaryResult]:
        results = {}
        for section in sections:
            content = section.get('content', '')
            if len(content) > 100:
                results[section['title']] = self.summarize(content, max_length, 50)
        return results
    
    def _parse_key_points(self, text: str) -> List[str]:
        """Extract key points from LLM response"""
        points = []
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith(('-', '•', '*', '1.', '2.', '3.', '4.', '5.')):
                points.append(line.lstrip('-•*1234567890. ').strip())
        return points[:5] if points else [text[:200]]


class AnthropicSummarizer(SummarizerBase):
    """Summarization using Anthropic Claude API"""
    
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        self.api_key = api_key
        self.model = model
        self._client = None
        self._init_client()
    
    def _init_client(self):
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Failed to init Anthropic client: {e}")
    
    def summarize(self, text: str, max_length: int = 500, min_length: int = 100) -> SummaryResult:
        if not self._client:
            return SummaryResult(
                summary="Anthropic client not initialized",
                key_points=[],
                model_used=self.model,
                confidence=0.0
            )
        
        prompt = f"""Summarize this research paper in approximately {max_length} words.
        Include 3-5 key bullet points.
        
        Paper text: {text[:12000]}"""
        
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_length + 300,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            key_points = self._parse_key_points(content)
            
            return SummaryResult(
                summary=content,
                key_points=key_points,
                model_used=self.model,
                confidence=0.95
            )
        except Exception as e:
            logger.error(f"Anthropic summarization error: {e}")
            return SummaryResult(
                summary=f"Error: {str(e)}",
                key_points=[],
                model_used=self.model,
                confidence=0.0
            )
    
    def summarize_sections(self, sections: List[Dict], max_length: int = 300) -> Dict[str, SummaryResult]:
        results = {}
        for section in sections:
            content = section.get('content', '')
            if len(content) > 100:
                results[section['title']] = self.summarize(content, max_length, 50)
        return results
    
    def _parse_key_points(self, text: str) -> List[str]:
        points = []
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith(('-', '•', '*', '1.', '2.', '3.', '4.', '5.')):
                points.append(line.lstrip('-•*1234567890. ').strip())
        return points[:5] if points else [text[:200]]


class SummarizerFactory:
    """Factory to create appropriate summarizer based on config"""
    
    @staticmethod
    def create(config) -> SummarizerBase:
        """Create summarizer based on configuration"""
        # Priority: OpenAI > Anthropic > Local
        if config.OPENAI_API_KEY and config.USE_LLM_API:
            logger.info("Using OpenAI summarizer")
            return OpenAISummarizer(config.OPENAI_API_KEY)
        elif config.ANTHROPIC_API_KEY and config.USE_LLM_API:
            logger.info("Using Anthropic summarizer")
            return AnthropicSummarizer(config.ANTHROPIC_API_KEY)
        else:
            logger.info(f"Using local transformers summarizer: {config.SUMMARIZATION_MODEL}")
            return LocalTransformersSummarizer(
                model_name=config.SUMMARIZATION_MODEL,
                device=config.DEVICE
            )