from pydantic import BaseModel
from typing import List

class TickIn(BaseModel):
    websiteId: str
    status: int
    latency: int

class TicksBatch(BaseModel):
    data: List[TickIn]

class CreateWebsiteIn(BaseModel):
    url: str
    contactInfo: str

class RegisterValidatorIn(BaseModel):
    publicKey: str
    location: str

class AddBalanceIn(BaseModel):
    amount: str