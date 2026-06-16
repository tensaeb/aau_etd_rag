import requests
import time
from typing import Any, Dict, Optional
from .interfaces import LLMProvider
from ..core.logger import setup_logger

logger = setup_logger("llm_provider")

class LMStudioLLM(LLMProvider):
    """Implementation of LLMProvider for LM Studio API."""
    
    def __init__(self, url: str, model: str, temperature: float = 0.0):
        self.url = url
        self.model = model
        self.temperature = temperature

    def _check_for_errors(self, error_text: str) -> Optional[str]:
        error_lower = error_text.lower()
        if any(x in error_lower for x in ["devicelost", "vk::device", "fencestatus"]):
            return "gpu"
        if any(x in error_lower for x in ["cache", "sequence", "position", "inconsistent"]):
            return "cache"
        if "channel error" in error_lower:
            return "channel"
        return None

    def generate(self, prompt: str, max_retries: int = 3, **kwargs) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": f"You are a factual academic assistant.\n\n{prompt}"}],
            "temperature": self.temperature,
            **kwargs
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(self.url, json=payload, timeout=120)
                
                # Check for API errors in the JSON body first
                result = response.json()
                if "error" in result:
                    error_msg = str(result.get("error", ""))
                    error_type = self._check_for_errors(error_msg)
                    if error_type:
                        logger.error(f"LM Studio {error_type.upper()} error detected: {error_msg}")
                        return f"Error: {error_type.upper()} issue in LM Studio. Please check the server."

                # Now check HTTP status
                if response.status_code != 200:
                    logger.error(f"LM Studio API error ({response.status_code}): {response.text}")
                
                response.raise_for_status()

                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                
                logger.warning(f"Unexpected response format: {result}")
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue

            except requests.exceptions.RequestException as e:
                logger.error(f"Request error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    return f"Error calling LLM: {str(e)}"

        return "Error: Failed to generate response after multiple retries."
