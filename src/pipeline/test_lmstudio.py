from ..core.config_loader import ConfigLoader
from ..models.llm import LMStudioLLM

def main():
    config_loader = ConfigLoader()
    config = config_loader.all
    
    llm = LMStudioLLM(
        url=config["lm_studio"]["url"],
        model=config["lm_studio"]["model"],
        temperature=config["lm_studio"]["temperature"]
    )
    
    prompt = "Explain artificial intelligence in one sentence."
    print(f"Testing LM Studio connection (URL: {llm.url})...")
    
    answer = llm.generate(prompt)
    print(f"\nResponse: {answer}")

if __name__ == "__main__":
    main()
