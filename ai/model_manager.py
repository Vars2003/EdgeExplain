from typing import List, Dict, Any
from utils.logger import get_logger
from ai.providers import OllamaProvider, LLMProvider

logger = get_logger("ai.model_manager")

class ModelManager:
    """
    Manages active offline model runtimes and checks memory requirements.
    Supports provider abstractions (Ollama, LM Studio).
    """

    def __init__(self):
        self.providers: Dict[str, LLMProvider] = {
            "Ollama": OllamaProvider()
        }
        self.active_provider_name = "Ollama"
        logger.info("Model Manager initialized with default provider: Ollama")

    def get_provider(self) -> LLMProvider:
        """
        Retrieves active provider implementation.
        """
        return self.providers.get(self.active_provider_name, self.providers["Ollama"])

    def switch_provider(self, name: str) -> bool:
        """
        Switches active model provider interface.
        """
        if name in self.providers:
            self.active_provider_name = name
            logger.info(f"Switched active provider to: {name}")
            return True
        logger.error(f"Failed to switch provider: '{name}' not registered.")
        return False

    def detect_available_models(self) -> List[str]:
        """
        Retrieves installed model names from active provider.
        """
        provider = self.get_provider()
        return provider.get_installed_models()

    @staticmethod
    def estimate_model_memory(model_name: str) -> Dict[str, Any]:
        """
        Heuristically estimates model sizes and hardware requirements.
        """
        name_lower = model_name.lower()
        size_gb = 4.5
        ram_gb = 6.5
        vram_gb = 5.5
        quant = "Q4_K_M"
        
        if "phi" in name_lower:
            size_gb = 2.2
            ram_gb = 3.5
            vram_gb = 3.0
            quant = "Q4_K_M"
        elif "gemma" in name_lower:
            size_gb = 5.2
            ram_gb = 7.5
            vram_gb = 6.5
            quant = "Q4_K_M"
        elif "llama" in name_lower and "3" in name_lower:
            size_gb = 4.7
            ram_gb = 6.8
            vram_gb = 6.0
            quant = "Q4_K_M"
        elif "tiny" in name_lower:
            size_gb = 0.6
            ram_gb = 1.2
            vram_gb = 0.8
            quant = "Q4_0"
        elif "mistral" in name_lower:
            size_gb = 4.1
            ram_gb = 6.2
            vram_gb = 5.2
            quant = "Q4_K_M"
            
        return {
            "model_name": model_name,
            "estimated_file_size_gb": size_gb,
            "minimum_system_ram_gb": ram_gb,
            "recommended_gpu_vram_gb": vram_gb,
            "quantization": quant,
            "compatible_features": ["Streaming", "Citations", "Custom Personas"]
        }

# Global Singleton Model Manager
model_manager = ModelManager()
