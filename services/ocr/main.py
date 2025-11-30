"""
Tesseract OCR Service for Bulgarian Document Processing
Works on any CPU (no AVX required)
"""

import os
import tempfile
import logging
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="OCR Service",
    description="Tesseract OCR service for Bulgarian document text extraction",
    version="1.0.0"
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Verify tesseract is available
        version = pytesseract.get_tesseract_version()
        return {"status": "healthy", "service": "ocr", "tesseract_version": str(version)}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.post("/ocr")
async def extract_text(file: UploadFile = File(...)):
    """
    Extract text from uploaded document (PDF or image).

    Supports:
    - PDF files (multi-page)
    - Images (PNG, JPG, JPEG, TIFF)

    Returns:
    - text: Combined text from all pages
    - pages: Number of pages processed
    - page_texts: Text per page (for multi-page documents)
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Get file extension
    ext = Path(file.filename).suffix.lower()

    if ext not in ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif']:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: PDF, PNG, JPG, JPEG, TIFF"
        )

    try:
        content = await file.read()
        logger.info(f"Processing file: {file.filename} ({len(content)} bytes)")

        page_texts = []
        # Use Bulgarian + Russian + English for best Cyrillic support
        lang = 'bul+rus+eng'

        if ext == '.pdf':
            # Convert PDF pages to images
            images = convert_from_bytes(content, dpi=300)
            logger.info(f"PDF has {len(images)} pages")

            for i, image in enumerate(images):
                page_text = pytesseract.image_to_string(image, lang=lang)
                page_texts.append(page_text)
                logger.info(f"Page {i+1}: extracted {len(page_text)} characters")
        else:
            # Single image file
            image = Image.open(io.BytesIO(content))
            page_text = pytesseract.image_to_string(image, lang=lang)
            page_texts.append(page_text)

        # Combine all pages
        combined_text = "\n\n--- Page Break ---\n\n".join(page_texts)

        return {
            "text": combined_text,
            "pages": len(page_texts),
            "page_texts": page_texts,
            "filename": file.filename
        }

    except Exception as e:
        logger.error(f"OCR error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "Tesseract OCR Service",
        "version": "1.0.0",
        "supported_formats": ["PDF", "PNG", "JPG", "JPEG", "TIFF"],
        "language": "Bulgarian + Russian + English (Cyrillic)"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
