# Deployment (Task 10)

Two documents cover deployment, in order:

1. **[AWS_SETUP.md](AWS_SETUP.md)** — provisioning the EC2 instance itself: instance type, security group rules, Elastic IP, and what to hand back once it exists. Read this first if the instance doesn't exist yet.
2. **This file** — configuring that instance once it's reachable over SSH.

## Steps

1. SSH into the instance.
2. Clone this repository.
3. Copy `.env.example` to `.env` and fill in `ELEVENLABS_API_KEY` and `ANTHROPIC_API_KEY` (see [docs/ENVIRONMENT_VARIABLES.md](../docs/ENVIRONMENT_VARIABLES.md)).
4. Run `./deploy/setup.sh` from the repo root. It installs dependencies, builds the corpus and both FAISS indices if `data/` is empty, writes a systemd unit, and starts the service.
5. Confirm it's live: `curl http://localhost:8000/health` should return `{"status": "ok"}`. From outside the instance, the same check against the Elastic IP confirms the security group is open on port 8000.
6. Re-run `scripts/benchmark_latency.py` against the live URL (not localhost) once Task 9 is complete, so the submitted P50/P70/P100 numbers reflect the real deployed path (PRD Open Decision 4).

## Redeploying code changes

```bash
git pull
sudo systemctl restart voice-rag
```

The corpus and indices under `data/` are not rebuilt on restart — delete them first if the ingestion or chunking logic changed and needs to run again.

## Troubleshooting

- `sudo systemctl status voice-rag` — service state and recent log lines.
- `sudo journalctl -u voice-rag -f` — follow logs live.
- `.env` missing or incomplete → `setup.sh` exits early with an explicit error rather than starting a service that will fail on its first request.
