from typing import Dict, Any

class FederatedCommunicationMonitor:
    """
    Analyzes and tracks live network telemetry stats (bandwidth rates, latency ratios).
    """
    @staticmethod
    def compile_monitor_stats(round_number: int, K: int, comm_bytes: int, latency_ms: float, total_time_ms: float) -> Dict[str, Any]:
        """
        Extrapolates channel details from metric counts.
        """
        duration_sec = total_time_ms / 1000.0
        bandwidth_mbps = 0.0
        if duration_sec > 0:
            bits = comm_bytes * 8
            bandwidth_mbps = float(bits / (1024 * 1024 * duration_sec))
            
        return {
            "round": round_number,
            "bytes_transferred": comm_bytes,
            "latency_ms": latency_ms,
            "upload_count": K,
            "download_count": K,
            "bandwidth_mbps": bandwidth_mbps,
            "latency_ratio": float(latency_ms / max(1.0, total_time_ms))
        }
