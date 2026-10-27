import PyPDF2
import io
import re
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from docx import Document as DocxDocument
from docx.oxml.ns import qn
import logging

# Placed under the marker of a page/slide that yielded no text, so the
# model (and anyone reading the context) knows material is missing there.
NO_TEXT_PAGE_NOTE = ("[No extractable text on this page: it may be an image, a diagram, "
                     "or a formula rendered as a picture.]")

def extract_text_from_file(uploaded_file):
    """
    Extract text from an uploaded file (PDF, PowerPoint or Word).

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

        # Check the bytes before parsing, so a file that cannot be read gets a
        # specific reason instead of a parser traceback retried three times.
        _check_file_signature(uploaded_file, file_extension)

        if file_extension == 'pdf':
            logging.info("PDF DEBUG: Processing PDF file")
            return extract_text_from_pdf(uploaded_file)
        elif file_extension == 'pptx':
            logging.info("PDF DEBUG: Processing PPTX file")
            return extract_text_from_pptx(uploaded_file)
        elif file_extension == 'docx':
            logging.info("PDF DEBUG: Processing DOCX file")
            return extract_text_from_docx(uploaded_file)
        else:
            logging.error(f"PDF DEBUG: Unsupported file type: {file_extension}")
            raise Exception(f"Unsupported file type: {file_extension}")

    except Exception as e:
        logging.error(f"PDF DEBUG: Error in extract_text_from_file: {e}")
        logging.error(f"PDF DEBUG: Exception type: {type(e)}")
        import traceback
        logging.error(f"PDF DEBUG: Traceback: {traceback.format_exc()}")
        raise

class UnreadableDocument(ValueError):
    """The upload cannot be parsed and retrying will not help; the message is
    written for the student."""


_OLE_SIGNATURE = b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"


def _check_file_signature(uploaded_file, file_extension):
    """Raise UnreadableDocument for an empty file, an old binary Office file or
    an encrypted / sensitivity-labelled Office file saved under a .pptx or
    .docx name, and for a .pdf that does not start with a PDF header."""
    try:
        if hasattr(uploaded_file, 'getvalue'):
            head = uploaded_file.getvalue()[:8]
            size = len(uploaded_file.getvalue())
        else:
            pos = uploaded_file.tell()
            head = uploaded_file.read(8)
            uploaded_file.seek(0, 2)
            size = uploaded_file.tell()
            uploaded_file.seek(pos)
    except Exception as e:
        logging.debug(f"PDF DEBUG: could not inspect the file header: {e}")
        return
    if size == 0:
        raise UnreadableDocument(
            "The file is empty (0 bytes). If it is stored in a cloud drive, open it once so it "
            "is downloaded, or re-export it, and try again.")
    if file_extension in ('pptx', 'docx'):
        if head.startswith(_OLE_SIGNATURE):
            raise UnreadableDocument(
                f"This .{file_extension} file could not be opened. It is either an older binary "
                f"Office file (.{file_extension[:3]}) saved with a new extension, or it is protected "
                "by a password or a sensitivity label. Please remove the protection or save a copy "
                "as a standard .pptx/.docx or as a PDF, and upload that.")
        if not head.startswith(b"PK\x03\x04"):
            raise UnreadableDocument(
                f"This .{file_extension} file is not a readable Office document (it may be corrupt "
                "or a different file type renamed). Please save a fresh copy or export it to PDF.")
    elif file_extension == 'pdf' and not head.startswith(b"%PDF"):
        raise UnreadableDocument(
            "This file does not look like a PDF (it may be corrupt or a different file type renamed). "
            "Please export it again and upload the new copy.")


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
                    # Keep the page in the sequence with a note, so the model
                    # knows a slide (often a formula or diagram rendered as a
                    # picture) exists here and is missing, instead of treating
                    # the deck as complete.
                    extracted_text += f"\n--- Page {page_num + 1} ---\n{NO_TEXT_PAGE_NOTE}\n"
                    logging.warning(f"PDF DEBUG: Page {page_num + 1} had no text")

            except Exception as e:
                logging.error(f"PDF DEBUG: Error processing page {page_num + 1}: {e}")
                continue  # Skip this page but continue with others

        logging.info(f"PDF DEBUG: Total extracted text length: {len(extracted_text)} ({pages_with_text}/{total_pages} pages had text)")
        if 0 < pages_with_text < total_pages:
            logging.warning(f"PDF DEBUG: {total_pages - pages_with_text} of {total_pages} pages had no extractable text (image-only)")

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

def _docx_table_lines(table, depth=0):
    """One line per row, cells joined by " | ". Exam papers often nest a data
    table (e.g. project cash flows) inside a question cell, so nested tables
    are rendered after the row that contains them."""
    lines = []
    for row in table.rows:
        cells, nested = [], []
        seen = set()
        for cell in row.cells:
            if id(cell._tc) in seen:   # merged cells repeat the same element
                continue
            seen.add(id(cell._tc))
            cells.append(' '.join(p.text.strip() for p in cell.paragraphs if p.text.strip()))
            for inner in cell.tables:
                nested.extend(_docx_table_lines(inner, depth + 1))
        if any(cells):
            lines.append(' | '.join(cells))
        lines.extend(nested)
    return lines

def extract_text_from_docx(uploaded_file):
    """
    Extract text from an uploaded Word document (.docx) using python-docx.
    Paragraphs are kept in order; tables are rendered one row per line with
    cells separated by " | " so question papers with data tables stay readable.
    Page numbers are not available in .docx, so citations fall back to section
    headings / question numbers (the prompts already handle that for exam papers).
    """
    try:
        doc = DocxDocument(_read_file_bytes(uploaded_file))
        parts = []
        # Walk body elements in document order so tables appear where they sit
        from docx.table import Table
        from docx.text.paragraph import Paragraph
        body = doc.element.body

        # Word records where each page ended when the file was last saved
        # (w:lastRenderedPageBreak) and explicit page breaks (w:br type=page).
        # When present they give real page numbers, so the same "--- Page N ---"
        # markers as PDFs can be emitted: citations work and the calculation
        # features can select pages instead of truncating the document.
        def _page_breaks_in(el):
            n = 0
            for node in el.iter():
                if node.tag == qn('w:lastRenderedPageBreak'):
                    n += 1
                elif node.tag == qn('w:br') and node.get(qn('w:type')) == 'page':
                    n += 1
            return n

        has_breaks = _page_breaks_in(body) > 0
        page = 1
        if has_breaks:
            parts.append(f"--- Page {page} ---")
        for child in body.iterchildren():
            tag = child.tag.rsplit('}', 1)[-1]
            if has_breaks:
                breaks = _page_breaks_in(child)
                if breaks:
                    page += breaks
                    parts.append(f"\n--- Page {page} ---")
            if tag == 'p':
                text = Paragraph(child, doc).text
                if text.strip():
                    parts.append(text)
            elif tag == 'tbl':
                parts.extend(_docx_table_lines(Table(child, doc)))
                parts.append('')
        extracted_text = '\n'.join(parts)
        if has_breaks:
            logging.info(f"PDF DEBUG: DOCX has {page} pages from recorded page breaks")
        logging.info(f"PDF DEBUG: DOCX extraction complete, {len(extracted_text)} characters")
        if not extracted_text.strip():
            raise Exception("No text could be extracted from the Word document")
        return extracted_text
    except Exception as e:
        logging.error(f"PDF DEBUG: Error extracting DOCX text: {e}")
        raise Exception(f"Failed to extract text from Word document: {str(e)}")

def _pptx_shape_text(shape):
    """Text of one shape, including the shapes inside a group (grouped text
    boxes and equation/diagram labels were previously dropped) and tables."""
    text = ""
    try:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            for inner in shape.shapes:
                text += _pptx_shape_text(inner)
            return text
    except Exception:
        pass
    if hasattr(shape, "text") and shape.text:
        text += shape.text + "\n"
    try:
        if shape.has_table:
            for row in shape.table.rows:
                text += " | ".join(cell.text.strip() for cell in row.cells) + "\n"
    except Exception:
        pass
    return text


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

            slide_text = ""
            for shape in slide.shapes:
                slide_text += _pptx_shape_text(shape)
            if slide_text.strip():
                extracted_text += slide_text
            else:
                # Picture-only slide (a diagram, a formula pasted as an image):
                # say so rather than leaving a silently empty slide.
                extracted_text += NO_TEXT_PAGE_NOTE + "\n"

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
