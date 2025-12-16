import fitz
from pathlib import Path

INPUT_DIR = Path("data/raw_pdfs")
OUTPUT_DIR = Path("data/extracted_text")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    return "\n".join(pages)

for pdf in INPUT_DIR.glob("*.pdf"):
    text = extract_text(pdf)
    output_file = OUTPUT_DIR / f"{pdf.stem}.txt"
    output_file.write_text(text, encoding="utf-8")
    print(f"Extracted: {pdf.name}")
