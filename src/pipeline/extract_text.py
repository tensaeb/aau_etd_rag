import sys
from pathlib import Path
from unstructured.partition.pdf import partition_pdf
from ..core.config_loader import ConfigLoader
from ..core.logger import setup_logger

logger = setup_logger("extract_text")

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extracts text from a single PDF file."""
    logger.info(f"Processing: {pdf_path.name}")
    try:
        elements = partition_pdf(
            filename=str(pdf_path),
            strategy="hi_res",
        )
        return "\n\n".join([el.text for el in elements])
    except Exception as e:
        logger.error(f"Failed to partition {pdf_path.name}: {e}")
        raise

def main():
    try:
        config_loader = ConfigLoader()
        input_dir = config_loader.get_path("data.raw_pdf_dir")
        output_dir = config_loader.get_path("data.extracted_text_dir")
        output_dir.mkdir(parents=True, exist_ok=True)

        pdf_files = list(input_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {input_dir}")
            return

        logger.info(f"Starting extraction for {len(pdf_files)} files...")
        
        for pdf_file in pdf_files:
            output_file = output_dir / f"{pdf_file.stem}.txt"
            
            # Skip if already exists (optional, but good for resuming)
            if output_file.exists():
                logger.info(f"  - Skipping {pdf_file.name} (already extracted)")
                continue
                
            try:
                text = extract_text_from_pdf(pdf_file)
                output_file.write_text(text, encoding="utf-8")
                logger.info(f"  -> Success: {output_file.name}")
            except Exception:
                logger.error(f"  !! Failed to process {pdf_file.name}, skipping.")
                continue

        logger.info("Extraction completed.")

    except Exception as e:
        logger.critical(f"Critical error in extraction pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
