# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from abc import ABC, abstractmethod
from typing import Any, Dict, List


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
