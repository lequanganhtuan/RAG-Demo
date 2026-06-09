from pathlib import Path
import fitz  # PyMuPDF
from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.table import Table
from docx.text.paragraph import Paragraph

def extract_text_from_pdf(file_path: str) -> str:
    """
    Read the full text from a PDF using PyMuPDF.
    """
    text_parts = []

    with fitz.open(file_path) as pdf:
        for page in pdf:
            page_text = page.get_text()
            text_parts.append(page_text)

    return "\n".join(text_parts)


def extract_text_from_docx(file_path: str) -> str:
    """
    Read the entire text from DOCX in correct order without duplicating table text.
    """
    doc = Document(file_path)
    text_parts = []

    for element in doc.element.body:
        if isinstance(element, OxmlElement):
            if element.tag.endswith('p'):
                p = Paragraph(element, doc)
                text_parts.append(p.text)
            elif element.tag.endswith('tbl'):
                t = Table(element, doc)
                for row in t.rows:
                    for cell in row.cells:
                        text_parts.append(cell.text)

    return "\n".join(text_parts)



def load_document(file_path: str) -> str:
    # Validation 
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Can't find: {file_path}")
    
    # File Extension Check)
    file_ext = path.suffix.lower()

    if file_ext == ".pdf":
        print("PDF Reader")
        return extract_text_from_pdf(file_path)

    elif file_ext == ".docx":
        print("DOCX Reader")
        return extract_text_from_docx(file_path)

    elif file_ext == ".txt":
        print("TXT Reader")
        
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    else:
        raise ValueError(f"Unsupported file formats: {file_ext}")
    
    # Edge Case Error Trap: PDF scan or file without text
    if not raw_text or not raw_text.strip():
        raise ValueError(
            f"File '{file_path}' contains no text. It might be an image, a scanned PDF, or empty."
        )

    return raw_text