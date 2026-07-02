import time
from typing import Generator, Any
from utils.logger import get_logger

logger = get_logger("ai.streaming")

class TokenStreamer:
    """
    Yields chunks progressively to simulate typing effects or formats LLM tokens 
    for Streamlit's st.write_stream layout.
    """

    @staticmethod
    def stream_tokens(token_generator: Generator[str, None, None]) -> Generator[str, None, None]:
        """
        Passes chunks through directly.
        """
        for token in token_generator:
            yield token

    @staticmethod
    def simulate_streaming(text: str, delay: float = 0.01) -> Generator[str, None, None]:
        """
        Splits a static text block by word/spaces to simulate a token stream.
        Used as a visual fallback when local LLM streaming is disabled or offline.
        """
        words = text.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            time.sleep(delay)
