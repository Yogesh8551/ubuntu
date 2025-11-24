import pdfplumber
import io
from docx import Document

def extract_text(filename: str, data: bytes):
    if filename.endswith(".pdf"):
        pages = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for p in pdf.pages:
                pages.append(p.extract_text() or "")
        return "\n".join(pages)

    if filename.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        return "\n".join([p.text for p in doc.paragraphs])

    return data.decode("utf-8", errors="ignore")

