# 🚀 DecenTrack Python Simulator + ML Integration


This is the Python-based backend for DecenTrack, migrated from the original Solidity + Sepolia setup.
It replaces Ethereum smart contracts with a FastAPI server + BlockSim-style simulator, and integrates a custom ML model to validate and weight validator uptime reports.


---

## 🧠 Features

#### 🌐 Python Blockchain Simulator


	- In-memory chain state (websites, validators, ticks, balances).
	- Block production loop with reward distribution.
- 
#### 🤖 ML-Driven Validation


	- A trained ML model (ml_engine/model.joblib) is used to score validator ticks.
	- Low-quality ticks are rejected automatically.
	- Rewards are weighted by ML score.
- 
#### 🔌 FastAPI Endpoints (Ethereum-like)


	- /website/create → create a website
	- /tx/addTick → submit uptime tick
	- /tx/addMultipleTicks → batch tick submission
	- /websites, /website/{id} → query websites
	- /ticks/{id} → query ticks (with ML weights)
	- /validator/register → register validator
	- /me/pendingPayout, /me/payouts → validator rewards
- 
#### 📊 Frontend Compatibility


	- API mirrors the original Solidity contract functions.
	- Works with the existing Next.js + Tailwind UI.

---

### 📂 Project Structure

	.
	├── ml_engine/
	│   ├── __init__.py
	│   ├── features.py        # Feature engineering for ML
	│   ├── model.py           # MLEngine wrapper
	│   └── model.joblib       # Trained ML model (generated)
	├── sim/
	│   ├── __init__.py
	│   ├── api.py             # FastAPI server
	│   ├── node.py            # Blockchain node logic
	│   ├── state.py           # Chain state (websites, validators, reports)
	│   └── models.py          # Pydantic request/response models
	├── blocksim/              # Optional BlockSim experiment code
	├── train_model.py         # Script to train and save ML model
	├── requirements.txt       # Python dependencies
	└── README.md


---

### ⚙️ Setup & Installation

1. Clone the repo

	git clone https://github.com/<your-username>/<your-repo>.git
	cd <your-repo>

2. Create virtual environment

	python -m venv .venv
	source .venv/bin/activate   # Linux/Mac
	.venv\Scripts\activate      # Windows

3. Install dependencies

	pip install -r requirements.txt

4. Train ML model

	python train_model.py

This generates ml_engine/model.joblib.

5. Run FastAPI server

	uvicorn sim.api:app --host 0.0.0.0 --port 8000 --reload

Server will be available at:
👉 http://localhost:8000
👉 API docs: http://localhost:8000/docs


---

### 🔗 Example API Usage

##### Create Website

	curl -X POST "http://localhost:8000/website/create?owner=0xabc" \
	  -H "Content-Type: application/json" \
	  -d '{"url":"https://github.com","contactInfo":"me@example.com"}'

##### Submit Tick

	curl -X POST "http://localhost:8000/tx/addTick?validator=0xval1" \
	  -H "Content-Type: application/json" \
	  -d '{"websiteId":"1","status":0,"latency":250}'

##### Get Ticks

	curl "http://localhost:8000/ticks/1?n=5"

##### Response includes ML weight:


	{
	  "status": "Success",
	  "data": [
	    {
	      "validator": "0xval1",
	      "createdAt": 1735710732,
	      "status": 0,
	      "latency": 250,
	      "location": "sim-location",
	      "ml_weight": 0.87
	    }
	  ]
	}


---

## 🧩 Integration with Frontend

- Replace ethers.js calls in your Next.js frontend with fetch calls to this FastAPI server.
- Example:
	- contract.addTick(...) → POST /tx/addTick
	- contract.getAllWebsites() → GET /websites
- The API responses are shaped to match the Solidity contract outputs, so minimal frontend changes are needed.

---

## 📊 BlockSim Experiments (Optional)


You can run the BlockSim simulation to test consensus + ML integration:


	python -m blocksim.run_sim

This prints block counts, reports, and validator balances.


---

## 📝 Notes

- ML model is pluggable: retrain with new dataset → replace model.joblib.
- Default ML threshold = 0.3 (ticks below this score are rejected).
- Rewards are distributed proportionally to ML weights.

---

## ✨ Author


Developed with ❤️ by Abhishek B R
CSE Student | Full-stack & Blockchain Developer | Learning AI, Golang, Rust