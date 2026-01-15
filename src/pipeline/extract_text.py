import yaml
from pathlib import Path
from unstructured.partition.pdf import partition_pdf

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

INPUT_DIR = Path(config["data"]["raw_pdf_dir"])
OUTPUT_DIR = Path(config["data"]["extracted_text_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    """
    Extracts structured text elements from a PDF using unstructured.
    This preserves layout and table information better than raw text extraction.
    """
    print(f"Partitioning PDF: {pdf_path.name}")
    elements = partition_pdf(
        filename=str(pdf_path),
        # Using "hi_res" strategy for better table and layout detection.
        strategy="hi_res",
        # You can add other parameters here, like `infer_table_structure=True`
        # if you have tables with complex structures.
    )
    # Combine the text from all extracted elements.
    return "\n\n".join([el.text for el in elements])

# --- Main Execution ---
print("Starting PDF text extraction with 'unstructured'...")
for pdf_file in INPUT_DIR.glob("*.pdf"):
    # Extract the structured text.
    extracted_content = extract_text_from_pdf(pdf_file)

    # Define the output file path.
    output_file = OUTPUT_DIR / f"{pdf_file.stem}.txt"

    # Save the content to a text file.
    output_file.write_text(extracted_content, encoding="utf-8")

    print(f"  -> Saved extracted text to: {output_file}")

print("\nPDF text extraction completed.")
