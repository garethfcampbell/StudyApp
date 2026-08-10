import PyPDF2
import io
import re
from pptx import Presentation
import logging

def extract_text_from_file(uploaded_file):
    """
    Extract text from an uploaded file (PDF or PowerPoint).

    Args:
        uploaded_file: Flask FileStorage object or an in-memory file object
                       with a 'filename' attribute (see app.py background upload)

    Returns:
        str: Extracted text from the file

    Raises:
        Exception: If file processing fails
    """
    logging.info("PDF DEBUG: Starting text extraction from file")

    try:
        filename = getattr(uploaded_file, 'filename', None)
        if not filename:
            logging.error("PDF DEBUG: Invalid file object or no filename found")
            raise Exception("Invalid file object or no filename found")

        # Get file extension
        if '.' not in filename:
            logging.error(f"PDF DEBUG: File has no extension. Filename: '{filename}'")
            raise Exception(f"File has no extension. Filename: '{filename}'")

        file_extension = filename.lower().split('.')[-1]
        logging.info(f"PDF DEBUG: File extension: {file_extension}")

        if file_extension == 'pdf':
            logging.info("PDF DEBUG: Processing PDF file")
            return extract_text_from_pdf(uploaded_file)
        elif file_extension == 'pptx':
            logging.info("PDF DEBUG: Processing PPTX file")
            return extract_text_from_pptx(uploaded_file)
        else:
            logging.error(f"PDF DEBUG: Unsupported file type: {file_extension}")
            raise Exception(f"Unsupported file type: {file_extension}")

    except Exception as e:
        logging.error(f"PDF DEBUG: Error in extract_text_from_file: {e}")
        logging.error(f"PDF DEBUG: Exception type: {type(e)}")
        import traceback
        logging.error(f"PDF DEBUG: Traceback: {traceback.format_exc()}")
        raise

def _read_file_bytes(uploaded_file):
    """Return a BytesIO of the file content.

    Uses getvalue() when available (io.BytesIO / in-memory objects - this is
    the production path, since app.py wraps uploads in BytesIO), which works
    regardless of the current read position. Falls back to read() for
    stream-like objects (e.g. Flask FileStorage).
    """
    if hasattr(uploaded_file, 'getvalue'):
        return io.BytesIO(uploaded_file.getvalue())
    return io.BytesIO(uploaded_file.read())

def extract_text_from_pdf(uploaded_file):
    """
    Extract text from an uploaded PDF file using PyPDF2.

    Args:
        uploaded_file: File object (BytesIO or Flask FileStorage)

    Returns:
        str: Extracted text from the PDF

    Raises:
        Exception: If PDF processing fails or the PDF has no extractable text
    """
    logging.info("PDF DEBUG: Starting PDF text extraction")

    try:
        pdf_bytes = _read_file_bytes(uploaded_file)

        logging.info("PDF DEBUG: Creating PDF reader object")
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(pdf_bytes)

        total_pages = len(pdf_reader.pages)
        logging.info(f"PDF DEBUG: PDF has {total_pages} pages")

        # Initialize empty text string
        extracted_text = ""
        pages_with_text = 0

        # Extract text from each page
        for page_num in range(total_pages):
            try:
                logging.debug(f"PDF DEBUG: Processing page {page_num + 1}")
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()

                if page_text and page_text.strip():
                    extracted_text += f"\n--- Page {page_num + 1} ---\n"
                    extracted_text += page_text
                    extracted_text += "\n"
                    pages_with_text += 1
                    logging.debug(f"PDF DEBUG: Page {page_num + 1} processed successfully")
                else:
                    logging.warning(f"PDF DEBUG: Page {page_num + 1} had no text")

            except Exception as e:
                logging.error(f"PDF DEBUG: Error processing page {page_num + 1}: {e}")
                continue  # Skip this page but continue with others

        logging.info(f"PDF DEBUG: Total extracted text length: {len(extracted_text)} ({pages_with_text}/{total_pages} pages had text)")

        # Aggregate check: if no page yielded any text, the PDF is almost
        # certainly scanned images. Raise a clear message rather than letting
        # the caller retry identical extractions.
        if total_pages > 0 and pages_with_text == 0:
            raise Exception(
                "This PDF appears to be scanned images with no extractable text. "
                "Please upload a text-based PDF (or the original PowerPoint file) instead."
            )

        # Clean up the text
        logging.info("PDF DEBUG: Cleaning extracted text")
        extracted_text = clean_text(extracted_text)

        logging.info("PDF DEBUG: PDF text extraction completed successfully")
        return extracted_text

    except Exception as e:
        logging.error(f"PDF DEBUG: Critical error in PDF extraction: {e}")
        logging.error(f"PDF DEBUG: Exception type: {type(e)}")
        import traceback
        logging.error(f"PDF DEBUG: Traceback: {traceback.format_exc()}")
        raise Exception(f"Failed to extract text from PDF: {str(e)}")

def extract_text_from_pptx(uploaded_file):
    """
    Extract text from an uploaded PowerPoint file using python-pptx.

    Args:
        uploaded_file: File object (BytesIO or Flask FileStorage)

    Returns:
        str: Extracted text from the PowerPoint

    Raises:
        Exception: If PowerPoint processing fails
    """
    try:
        pptx_bytes = _read_file_bytes(uploaded_file)

        # Create a Presentation object
        prs = Presentation(pptx_bytes)

        # Initialize empty text string
        extracted_text = ""

        # Extract text from each slide
        for slide_num, slide in enumerate(prs.slides):
            extracted_text += f"\n--- Slide {slide_num + 1} ---\n"

            # Extract text from all text-containing shapes
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    extracted_text += shape.text + "\n"

                # Handle tables
                if shape.has_table:
                    for row in shape.table.rows:
                        row_text = []
                        for cell in row.cells:
                            row_text.append(cell.text.strip())
                        extracted_text += " | ".join(row_text) + "\n"

        # Clean up the text
        extracted_text = clean_text(extracted_text)

        return extracted_text

    except Exception as e:
        logging.error(f"PowerPoint extraction error: {e}")
        raise Exception(f"Failed to extract text from PowerPoint: {str(e)}")

def clean_text(text):
    """
    Clean and normalize extracted text while preserving paragraph structure.

    Trailing/leading whitespace is stripped from each line, but blank lines
    are kept (collapsed to a single blank line) so paragraph breaks survive.

    Args:
        text (str): Raw extracted text

    Returns:
        str: Cleaned text
    """
    if not text:
        return ""

    # Strip whitespace from each line but keep blank lines so paragraph
    # boundaries are preserved
    cleaned_lines = [line.strip() for line in text.split('\n')]
    cleaned_text = '\n'.join(cleaned_lines)

    # Collapse runs of blank lines: 3+ consecutive newlines become exactly one
    # blank line between paragraphs
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)

    return cleaned_text.strip()
