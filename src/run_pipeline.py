import sys
import subprocess
import time
from pathlib import Path
from .core.config_loader import ConfigLoader
from .core.logger import setup_logger

logger = setup_logger("pipeline_runner")

def run_step(script_path: str):
    """Runs a pipeline step as a module using the current python executable."""
    # Convert path to module notation: src/pipeline/extract_text.py -> src.pipeline.extract_text
    module_name = script_path.replace("/", ".").replace("\\", ".").replace(".py", "")
    
    logger.info(f"--- STARTING STEP: {module_name} ---")
    start_time = time.time()
    
    try:
        # Run as a module so imports work correctly
        result = subprocess.run(
            [sys.executable, "-m", module_name],
            check=True,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(result.stdout)
        
        duration = time.time() - start_time
        logger.info(f"--- COMPLETED STEP: {module_name} in {duration:.2f}s ---")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in step {module_name}:")
        print(e.stdout)
        print(e.stderr)
        sys.exit(1)

def main():
    """Orchestrates the full data pipeline from Step Zero to Indexing."""
    try:
        config_loader = ConfigLoader()
        
        pipeline_steps = [
            "src/pipeline/extract_text.py",
            "src/pipeline/chunk_text.py",
            "src/pipeline/build_bm25.py",
            "src/pipeline/embed_chunks.py",
            "src/pipeline/build_faiss.py"
        ]

        logger.info("="*70)
        logger.info("AAU ETD RAG PIPELINE RUNNER")
        logger.info("="*70)
        
        for step in pipeline_steps:
            run_step(step)

        logger.info("="*70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*70)

    except Exception as e:
        logger.critical(f"Pipeline runner failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
