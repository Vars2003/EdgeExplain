from typing import Dict, Any, List
import pandas as pd
from utils.logger import get_logger

logger = get_logger("intelligence.plugins")

class IntelligencePlugin:
    """
    Base class interface for intelligence layer plugins (e.g. SHAP, LIME, AutoML).
    """
    
    def run(self, df: pd.DataFrame, memory_obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the plugin computation.
        Returns a dictionary of results to append to the MemoryObject.
        """
        raise NotImplementedError("Plugins must implement the run() method.")

class IntelligencePluginManager:
    """
    Manages and triggers intelligence plugins.
    Allows downstream expansions to enrich dataset summaries.
    """

    def __init__(self):
        self._plugins: Dict[str, IntelligencePlugin] = {}
        logger.info("Intelligence plugin manager initialized.")

    def register(self, name: str, plugin: IntelligencePlugin) -> None:
        """
        Registers a plugin in the registry.
        """
        self._plugins[name] = plugin
        logger.info(f"Plugin registered successfully: '{name}'")

    def run_all(self, df: pd.DataFrame, memory_obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs all registered plugins and compiles their outputs.
        """
        plugin_outputs = {}
        if not self._plugins:
            return plugin_outputs
            
        logger.info(f"Triggering {len(self._plugins)} registered plugins...")
        for name, plugin in self._plugins.items():
            try:
                logger.info(f"Executing plugin: '{name}'")
                plugin_outputs[name] = plugin.run(df, memory_obj)
            except Exception as e:
                logger.exception(f"Plugin '{name}' failed during execution: {e}")
                plugin_outputs[name] = {"error": str(e)}
                
        return plugin_outputs

# Global Singleton Plugin Manager
plugin_manager = IntelligencePluginManager()
