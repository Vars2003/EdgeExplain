import time
import pandas as pd
from typing import Dict, Any, List
from utils.logger import get_logger
from intelligence.plugins import plugin_manager, IntelligencePlugin

logger = get_logger("plugins.base")

class BasePlugin(IntelligencePlugin):
    """
    Standardized base class for EdgeExplain plugins.
    Enforces standardized execution output schemas.
    """

    def __init__(self, name: str, dependencies: List[str]):
        self.name = name
        self.dependencies = dependencies

    def run(self, df: pd.DataFrame, memory_obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wrapper that logs execution metrics and formats the final envelope.
        """
        start_time = time.time()
        logger.info(f"Triggering execution of plugin: '{self.name}'")
        
        try:
            # Execute actual plugin logic
            res_dict = self.execute(df, memory_obj)
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            return {
                "status": "SUCCESS",
                "result": res_dict.get("result", {}),
                "confidence": res_dict.get("confidence", 100),
                "evidence": res_dict.get("evidence", []),
                "execution_time_ms": elapsed_ms,
                "dependencies": self.dependencies
            }
        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.exception(f"Execution failed for plugin '{self.name}': {e}")
            return {
                "status": "ERROR",
                "result": {"error": str(e)},
                "confidence": 0,
                "evidence": [f"Runtime exception: {str(e)}"],
                "execution_time_ms": elapsed_ms,
                "dependencies": self.dependencies
            }

    def execute(self, df: pd.DataFrame, memory_obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Must be implemented by children to run specific calculations.
        Returns a dict: { "result": ..., "confidence": ..., "evidence": ... }
        """
        raise NotImplementedError("Plugins must implement execute()")
