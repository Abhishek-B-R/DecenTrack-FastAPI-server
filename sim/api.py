import threading
import time
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from .state import ChainState
from .node import Node
from .models import TickIn, TicksBatch, CreateWebsiteIn, RegisterValidatorIn, AddBalanceIn

app = FastAPI(title="DecenTrack Simulator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

state = ChainState()
node = Node(state)

@app.get("/health")
def health():
    return {"ok": True}

def block_loop():
    while True:
        time.sleep(2.0)
        node.produce_block()

threading.Thread(target=block_loop, daemon=True).start()

@app.post("/tx/addTick")
def add_tick(body: TickIn, validator: str = Query(default="0xvalidator")):
    ok = node.submit_tick(
        {
            "website_id": body.websiteId,
            "validator": validator,
            "status": int(body.status),
            "latency": int(body.latency),
            "timestamp": int(time.time()),
        }
    )
    return {"status": "Success" if ok else "Rejected", "accepted": ok}

@app.post("/tx/addMultipleTicks")
def add_multiple_ticks(batch: TicksBatch, validator: str = Query(default="0xvalidator")):
    accepted = 0
    for t in batch.data:
        ok = node.submit_tick(
            {
                "website_id": t.websiteId,
                "validator": validator,
                "status": int(t.status),
                "latency": int(t.latency),
                "timestamp": int(time.time()),
            }
        )
        accepted += int(ok)
    return {"status": "Success", "accepted": accepted, "total": len(batch.data)}

@app.post("/website/create")
def create_website(body: CreateWebsiteIn, owner: str = Query(default="0xowner")):
    wid = node.add_website(body.url, body.contactInfo, owner)
    return {"status": "Success", "websiteId": wid}

@app.delete("/website/{website_id}")
def delete_website(website_id: str):
    ok = node.delete_website(website_id)
    return {"status": "Success" if ok else "NotFound"}

@app.get("/websites")
def get_all_websites():
    return {"websites": [vars(w) for w in state.websites.values()]}

@app.get("/website/{website_id}")
def get_website(website_id: str):
    w = state.websites.get(website_id)
    if not w:
        return {"status": "Error", "error": "Not found"}
    return {"status": "Success", "data": vars(w)}

@app.post("/validator/register")
def register_validator(body: RegisterValidatorIn, address: str = Query(default="0xvalidator")):
    v = node.register_validator(address, body.publicKey, body.location)
    return {"status": "Success", "data": vars(v)}

@app.get("/validator/{address}")
def get_validator(address: str):
    v = state.validators.get(address)
    return {"status": "Success", "data": vars(v) if v else None}

@app.get("/validator/{address}/authenticated")
def is_validator_authenticated(address: str):
    v = state.validators.get(address)
    return bool(v and v.authenticated)

# sim/api.py (only showing the two ticks endpoints)
@app.get("/ticks/{website_id}")
def get_recent_ticks(website_id: str, n: int = 10):
    ticks = [r for r in state.reports if r["website_id"] == website_id][-n:]
    return {
        "status": "Success",
        "data": [
            {
                "validator": r["validator"],
                "createdAt": r["createdAt"],
                "status": r["status"],
                "latency": r["latency"],
                "location": r.get("location", "sim-location"),
                "ml_weight": r.get("ml_weight", 1.0),
            }
            for r in ticks
        ],
    }

@app.get("/ticks/{website_id}/all")
def get_all_ticks(website_id: str):
    ticks = [r for r in state.reports if r["website_id"] == website_id]
    return {
        "status": "Success",
        "data": [
            {
                "validator": r["validator"],
                "createdAt": r["createdAt"],
                "status": r["status"],
                "latency": r["latency"],
                "location": r.get("location", "sim-location"),
                "ml_weight": r.get("ml_weight", 1.0),
            }
            for r in ticks
        ],
    }

@app.get("/me/websites")
def get_my_websites(owner: str):
    items = [vars(w) for w in state.websites.values() if w.owner == owner]
    return {"status": "Success", "data": items}

@app.get("/me/pendingPayout")
def my_pending_payout(owner: str):
    v = state.validators.get(owner)
    bal = v.balance if v else 0
    return {"status": "Success", "data": str(bal)}

@app.post("/me/payouts")
def get_my_payouts(owner: str = Query(default="0xowner")):
    v = state.validators.get(owner)
    if v:
        amount = v.balance
        v.balance = 0
        rec = {"time": int(time.time()), "amount": amount}
        state.payouts.setdefault(owner, []).append(rec)
        return {"status": "Success", "txHash": f"sim-{rec['time']}"}
    return {"status": "Error", "error": "Not found"}

@app.post("/website/{website_id}/balance")
def add_website_balance(website_id: str, body: AddBalanceIn):
    w = state.websites.get(website_id)
    if not w:
        return {"status": "Error", "error": "Not found"}
    try:
        wei = int(float(body.amount) * 1e18)
    except Exception:
        wei = 0
    w.balance_wei += wei
    return {"status": "Success", "txHash": f"sim-{int(time.time())}"}

@app.get("/website/{website_id}/balance")
def get_website_balance(website_id: str):
    w = state.websites.get(website_id)
    if not w:
        return {"status": "Error", "error": "Not found"}
    return {"status": "Success", "data": str(w.balance_wei)}