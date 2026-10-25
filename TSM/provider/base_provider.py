from abc import ABC, abstractmethod
from typing import Dict, Optional, List

class BaseLLMProvider(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def chat_completion(
        self,
        messages: List[Dict],
        model: str,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        stream: bool = False,
        max_tokens: int = 8192,
        **kwargs
    ) -> Dict:
        pass