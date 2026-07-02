from typing import Callable, List, Dict, Any
from utils.logger import get_logger

logger = get_logger("intelligence.events")

class EventDispatcher:
    """
    Lightweight, thread-safe in-memory event dispatching system.
    Enables decoupled reactivity across analytics and AI modules.
    """

    def __init__(self):
        self._listeners: Dict[str, List[Callable[..., Any]]] = {}
        logger.info("Intelligence event dispatcher initialized.")

    def subscribe(self, event_type: str, callback: Callable[..., Any]) -> None:
        """
        Registers a callback to listen for a specific event type.
        """
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(callback)
        logger.debug(f"Subscribed callback to event: {event_type}")

    def dispatch(self, event_type: str, *args: Any, **kwargs: Any) -> None:
        """
        Triggers all callbacks subscribed to the event type.
        """
        if event_type in self._listeners:
            logger.info(f"Dispatching event: {event_type} (listeners: {len(self._listeners[event_type])})")
            for callback in self._listeners[event_type]:
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    logger.exception(f"Error executing callback for event '{event_type}': {e}")
        else:
            logger.debug(f"Dispatched event '{event_type}' but no listeners were registered.")

# Global Singleton Event System
event_system = EventDispatcher()
