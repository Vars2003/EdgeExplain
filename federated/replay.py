from typing import Dict, Any, List

class FederatedReplayController:
    """
    Replays completed round snapshots from history without rerunning training fits.
    """
    @staticmethod
    def load_round_snapshot(round_number: int, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Retrieves matching version status log for a round index.
        """
        for entry in history:
            if entry.get("round") == round_number:
                return entry
        return None
