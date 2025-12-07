from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

from src.engine.request_queue import BatchedRequest

logger = logging.getLogger(__name__)

class BaseHandler(ABC):
    """
    Base class for model handlers.
    Each handler implements loading and inference logic for a specific model type.
    """
    
    def __init__(
        self, 
        model_path: str, 
        device: str, 
        max_length: int, 
        use_fp16: bool
    ):
        self.model_path = model_path
        self.device = device
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.model = None

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the model into memory.
        """
        pass

    @abstractmethod
    def predict(self, batch: BatchedRequest) -> List[List[Dict[str, Any]]]:
        """
        Run inference on a batch of requests.
        
        Args:
            batch: The batch of requests to process.
            
        Returns:
            A list of results for each request in the batch.
            Each result is a list of dictionaries containing score and other metadata.
        """
        pass
    
    def unload(self) -> None:
        """Unload the model to free resources."""
        if self.model is not None:
            del self.model
            self.model = None
