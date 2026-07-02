import requests
from typing import Generator, Dict, Any, List
from utils.logger import get_logger

logger = get_logger("ai.providers")

class LLMProvider:
    """
    Abstract Base Class defining the interface for local offline LLM runtimes.
    """
    
    def generate_chat(self, model: str, messages: List[Dict[str, str]], temperature: float, stream: bool = True) -> Generator[str, None, None]:
        """
        Sends conversational messages to the local runtime.
        Yields tokens/chunks in streaming mode.
        """
        raise NotImplementedError("Providers must implement generate_chat()")

    def get_installed_models(self) -> List[str]:
        """
        Scans local runtime environment for downloaded model tags.
        """
        raise NotImplementedError("Providers must implement get_installed_models()")


class OllamaProvider(LLMProvider):
    """
    Local runtime provider implementing connection to Ollama API (localhost:11434).
    """

    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host
        logger.info(f"Ollama provider initialized with host: {self.host}")

    def get_installed_models(self) -> List[str]:
        """
        Queries Ollama API tags endpoint. Returns list of local model names.
        """
        try:
            url = f"{self.host}/api/tags"
            logger.debug(f"Fetching Ollama tags from: {url}")
            response = requests.get(url, timeout=2.0)
            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                logger.info(f"Ollama detected models: {models}")
                return models
        except Exception as e:
            logger.warning(f"Ollama provider is offline or unreachable: {e}")
        return []

    def generate_chat(self, model: str, messages: List[Dict[str, str]], temperature: float, stream: bool = True) -> Generator[str, None, None]:
        """
        Sends chat queries to Ollama API chat endpoint. Yields text tokens.
        """
        url = f"{self.host}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "options": {
                "temperature": temperature
            },
            "stream": stream
        }
        
        try:
            logger.info(f"Sending chat query to Ollama (model: {model}, stream: {stream})")
            response = requests.post(url, json=payload, stream=stream, timeout=10.0)
            
            if response.status_code != 200:
                logger.error(f"Ollama returned error status: {response.status_code}")
                yield f"Error: Local model runtime returned status code {response.status_code}."
                return
                
            if stream:
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json_data = line.decode('utf-8')
                            data = json_data = json_data = line.decode('utf-8')
                            import json
                            parsed = json.loads(data)
                            content = parsed.get("message", {}).get("content", "")
                            if content:
                                yield content
                        except Exception as e:
                            logger.error(f"Error parsing Ollama stream chunk: {e}")
            else:
                data = response.json()
                yield data.get("message", {}).get("content", "")
                
        except Exception as e:
            logger.exception(f"Ollama chat generation failed: {e}")
            yield f"Error: Failed to connect to local Ollama API. Details: {e}"
