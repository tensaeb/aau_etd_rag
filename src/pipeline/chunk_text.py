import sys
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..core.config_loader import ConfigLoader
from ..core.logger import setup_logger

logger = setup_logger("chunk_text")

def main():
    try:
        config_loader = ConfigLoader()
        input_dir = config_loader.get_path("data.extracted_text_dir")
        output_dir = config_loader.get_path("data.chunk_dir")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Note: chunk_size and overlap are in CHARACTERS in this implementation
        chunk_size = config_loader.get("chunking.chunk_size")
        overlap = config_loader.get("chunking.overlap")

        logger.info(f"Initializing text splitter (size={chunk_size}, overlap={overlap})")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
        )

        text_files = list(input_dir.glob("*.txt"))
        if not text_files:
            logger.warning(f"No text files found in {input_dir}")
            return

        for file in text_files:
            logger.info(f"Chunking: {file.name}")
            text = file.read_text(encoding="utf-8")
            chunks = splitter.split_text(text)
            
            output_file = output_dir / f"{file.stem}.txt"
            output_file.write_text("\n\n---\n\n".join(chunks), encoding="utf-8")
            logger.info(f"  -> Created {len(chunks)} chunks.")

        logger.info("Chunking completed.")

    except Exception as e:
        logger.critical(f"Critical error in chunking pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
