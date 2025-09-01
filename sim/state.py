from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class Validator:
    address: str
    public_key: str = ""
    location: str = ""
    authenticated: bool = True
    balance: int = 0
    reputation: float = 1.0

@dataclass
class Website:
    id: str
    url: str
    contact_info: str
    owner: str
    active: bool = True
    balance_wei: int = 0

@dataclass
class ChainState:
    validators: Dict[str, Validator] = field(default_factory=dict)
    websites: Dict[str, Website] = field(default_factory=dict)
    reports: List[dict] = field(default_factory=list)
    payouts: Dict[str, List[dict]] = field(default_factory=dict)