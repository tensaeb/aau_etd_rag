
import yaml
import subprocess
from pathlib import Path

def run_script(script_name):
    """Run a script and check for errors."""
    result = subprocess.run(["python", script_name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_name}:")
        print(result.stdout)
        print(result.stderr)
        exit(1)
    print(result.stdout)

def main():
    """Run the full data pipeline."""
    # Load config to ensure it's valid
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Define the pipeline steps
    pipeline_steps = [
        "src/pipeline/extract_text.py",
        "src/pipeline/chunk_text.py",
        "src/pipeline/build_bm25.py",  # New step for keyword search
        "src/pipeline/embed_chunks.py",
        "src/pipeline/build_faiss.py"
    ]

    # Ensure all required directories exist
    Path(config["data"]["extracted_text_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["data"]["chunk_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["data"]["embedding_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["faiss"]["index_path"]).parent.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("STARTING DATA PIPELINE")
    print("="*70)

    for step in pipeline_steps:
        print(f"\n----- Running: {step} -----")
        run_script(step)

    print("="*70)
    print("DATA PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)

if __name__ == "__main__":
    main()
