from dataclasses import dataclass

@dataclass
class UptimeReportTx:
    website_id: str
    validator_id: str
    status: int
    latency_ms: int
    timestamp: int