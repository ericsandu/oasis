from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class RecsysInterface(ABC):
    """
    Abstract Base Class for Recommender Systems in OASIS.
    All recommender backends (TwinBERT, Gorse) must implement this interface.
    """

    @abstractmethod
    def update_rec_table(
        self,
        user_table: List[Dict[str, Any]],
        post_table: List[Dict[str, Any]],
        trace_table: List[Dict[str, Any]],
        rec_matrix: Any,
        max_rec_post_len: int,
    ) -> Any:
        """
        Takes the current state of users, posts, and traces,
        and returns the updated recommendation matrix.
        """
        pass
