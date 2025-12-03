"""
PDF Ingestion and Chunking for ATE Knowledge Base.
Handles Bulgarian legal documents with smart chunking.
"""

import fitz  # PyMuPDF
import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata."""
    text: str
    document: str  # "naredba_24", "uchebnik_ate", or "court_expertise"
    section: str
    article: Optional[str]
    page: int
    chunk_type: str  # "regulation", "methodology", "formula", "definition", "example"
    chunk_index: int


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from PDF with page information.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of dicts with 'text' and 'page' keys
    """
    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        if text.strip():
            pages.append({
                "text": text,
                "page": page_num + 1
            })

    doc.close()
    logger.info(f"Extracted {len(pages)} pages from {pdf_path}")
    return pages


def detect_document_type(filename: str, text_sample: str) -> str:
    """Detect document type: Naredba 24, ATE textbook, or court expertise."""
    filename_lower = filename.lower()
    text_lower = text_sample.lower()[:2000]

    # Check for Naredba 24 (legal regulation)
    if "naredba" in filename_lower or "наредба" in text_lower:
        return "naredba_24"

    # Check for ATE textbook
    elif "uchebnik" in filename_lower or "учебник" in text_lower:
        return "uchebnik_ate"

    # Check for court expertise (real case files)
    # Patterns: "s-v" (court case number), "PTP/ПТП" (traffic accident), date patterns like "04.04.2025"
    elif (
        "s-v" in filename_lower or
        "ptp" in filename_lower or
        "експертиза" in text_lower or
        "съдебно" in text_lower or
        re.search(r'\d{2}\.\d{2}\.\d{4}.*s-v', filename_lower) or
        "районен съд" in text_lower or
        "окръжен съд" in text_lower
    ):
        return "court_expertise"

    else:
        # Default based on content patterns
        if re.search(r'чл\.\s*\d+', text_sample[:5000], re.IGNORECASE):
            return "naredba_24"
        return "uchebnik_ate"


def detect_chunk_type(text: str) -> str:
    """Classify chunk content type."""
    text_lower = text.lower()

    # Check for formulas (math patterns)
    if re.search(r'[vVсС]\s*[=×·]\s*|√|²|³|\d+\s*[*/+-]\s*\d+', text):
        return "formula"

    # Check for article references (regulations)
    if re.search(r'чл\.\s*\d+|член\s+\d+|ал\.\s*\d+', text_lower):
        return "regulation"

    # Check for definitions
    if re.search(r'означава|определя|дефинира|се нарича', text_lower):
        return "definition"

    # Check for examples
    if re.search(r'пример|например|случай\s+\d+|задача', text_lower):
        return "example"

    # Default to methodology
    return "methodology"


def extract_article_number(text: str) -> Optional[str]:
    """Extract article number if present (e.g., 'Чл. 5')."""
    match = re.search(r'чл\.\s*(\d+)', text, re.IGNORECASE)
    if match:
        return f"Чл. {match.group(1)}"
    return None


def extract_section_title(text: str, prev_section: str = "") -> str:
    """Extract section or chapter title from text."""
    # Look for chapter patterns
    chapter_match = re.search(
        r'глава\s+([IVXLCDM]+|\d+)[:\s]*([^\n]+)?',
        text,
        re.IGNORECASE
    )
    if chapter_match:
        title = chapter_match.group(2) or ""
        return f"Глава {chapter_match.group(1)} {title}".strip()

    # Look for section patterns
    section_match = re.search(
        r'раздел\s+([IVXLCDM]+|\d+)[:\s]*([^\n]+)?',
        text,
        re.IGNORECASE
    )
    if section_match:
        title = section_match.group(2) or ""
        return f"Раздел {section_match.group(1)} {title}".strip()

    # Keep previous section if no new one found
    return prev_section or "Общи положения"


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50
) -> List[str]:
    """
    Split text into chunks with overlap.
    Tries to break at sentence boundaries.

    Args:
        text: Text to chunk
        chunk_size: Target chunk size in words
        overlap: Number of overlapping words

    Returns:
        List of text chunks
    """
    # Split into sentences (Bulgarian uses same sentence endings)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        words = sentence.split()
        sentence_size = len(words)

        if current_size + sentence_size > chunk_size and current_chunk:
            # Save current chunk
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)

            # Start new chunk with overlap
            overlap_words = ' '.join(current_chunk).split()[-overlap:]
            current_chunk = overlap_words + words
            current_size = len(current_chunk)
        else:
            current_chunk.extend(words)
            current_size += sentence_size

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def process_pdf(
    pdf_path: str,
    chunk_size: int = 512,
    overlap: int = 50
) -> List[DocumentChunk]:
    """
    Process a PDF into indexed chunks with metadata.

    Args:
        pdf_path: Path to PDF file
        chunk_size: Words per chunk
        overlap: Overlapping words between chunks

    Returns:
        List of DocumentChunk objects
    """
    path = Path(pdf_path)
    logger.info(f"Processing PDF: {path.name}")

    # Extract all pages
    pages = extract_text_from_pdf(pdf_path)

    if not pages:
        logger.warning(f"No text extracted from {pdf_path}")
        return []

    # Detect document type
    full_text = ' '.join(p['text'] for p in pages[:5])
    doc_type = detect_document_type(path.name, full_text)
    logger.info(f"Detected document type: {doc_type}")

    # Process pages into chunks
    all_chunks = []
    current_section = ""
    chunk_index = 0

    for page_info in pages:
        page_text = page_info['text']
        page_num = page_info['page']

        # Update section if new one found
        current_section = extract_section_title(page_text, current_section)

        # Chunk this page's text
        text_chunks = chunk_text(page_text, chunk_size, overlap)

        for text_chunk in text_chunks:
            if len(text_chunk.strip()) < 50:  # Skip very short chunks
                continue

            chunk = DocumentChunk(
                text=text_chunk,
                document=doc_type,
                section=current_section,
                article=extract_article_number(text_chunk),
                page=page_num,
                chunk_type=detect_chunk_type(text_chunk),
                chunk_index=chunk_index
            )
            all_chunks.append(chunk)
            chunk_index += 1

    logger.info(f"Created {len(all_chunks)} chunks from {path.name}")

    # Log chunk type distribution
    type_counts = {}
    for c in all_chunks:
        type_counts[c.chunk_type] = type_counts.get(c.chunk_type, 0) + 1
    logger.info(f"Chunk types: {type_counts}")

    return all_chunks
