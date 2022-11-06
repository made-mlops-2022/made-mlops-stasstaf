from dataclasses import dataclass, field
from typing import List


@dataclass()
class TrainingParams:
    model_type: str = field()
    evaluate_metrics: List[str] = field()
    random_state: int = field(default=42)

