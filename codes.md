# start the application

```bash
    uvicorn sim.api:app --reload
```

# add sample data

```bash
    # health
    curl http://127.0.0.1:8000/health

    # create site
    curl -X POST "http://127.0.0.1:8000/website/create?owner=0xabc" \
    -H "Content-Type: application/json" \
    -d '{"url":"https://github.com","contactInfo":"me@example.com"}'

    # submit ticks with different latencies
    curl -X POST "http://127.0.0.1:8000/tx/addTick?validator=sim-val" \
    -H "Content-Type: application/json" \
    -d '{"websiteId":"1","status":0,"latency":120}'

    curl -X POST "http://127.0.0.1:8000/tx/addTick?validator=sim-val" \
    -H "Content-Type: application/json" \
    -d '{"websiteId":"1","status":0,"latency":800}'

    curl -X POST "http://127.0.0.1:8000/tx/addTick?validator=sim-val" \
    -H "Content-Type: application/json" \
    -d '{"websiteId":"1","status":0,"latency":3500}'

    # read ticks
    curl "http://127.0.0.1:8000/ticks/1?n=10"
```

# run the comparison between PoW and PoW + ML
```bash
    python -m sim.experiment_pow_vs_ml
```