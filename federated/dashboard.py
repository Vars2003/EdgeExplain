from typing import List, Dict, Any

class FederatedDashboardHelper:
    """
    Format utilities for rendering federated statistics.
    """
    @staticmethod
    def format_size(bytes_count: int) -> str:
        """
        Formats byte sizes to readable strings.
        """
        if bytes_count < 1024:
            return f"{bytes_count} B"
        elif bytes_count < 1024 * 1024:
            return f"{bytes_count / 1024:.2f} KB"
        else:
            return f"{bytes_count / (1024 * 1024):.2f} MB"
