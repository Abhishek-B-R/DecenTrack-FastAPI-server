from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class Validator:
    id: str
    balance: int = 0
    reputation: float = 1.0

@dataclass
class Website:
    id: str
    url: str
    active: bool = True

@dataclass
class ChainState:
    validators: Dict[str, Validator] = field(default_factory=dict)
    websites: Dict[str, Website] = field(default_factory=dict)
    reports: List[dict] = field(default_factory=list)
    chain: List[dict] = field(default_factory=list)