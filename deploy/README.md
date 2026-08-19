# Deployment (Task 10)

Live URL: **https://ragingoa.duckdns.org** · Preflight: **https://ragingoa.duckdns.org/preflight**

Two documents cover deployment, in order:

1. **[AWS_SETUP.md](AWS_SETUP.md)** — the EC2 instance itself: instance type, security group,
   Elastic IP, and the applied-change log. The instance is already provisioned; read this
   for its current state or before changing anything about it.
2. **This file** — configuring and operating that instance.

## Current state

| | |
|---|---|
| Instance | `i-09e157bfae9bb82a6` · `ap-south-1b` · `m7i-flex.large` (2 vCPU, 7.6 GB) |
| Address | Elastic IP `13.234.228.244` — stable across stop/start |
| Disk | 20 GiB gp3, ~9 GB free · 2 GB swap (`vm.swappiness=10`) |
| TLS | Caddy + Let's Encrypt, auto-renewing, valid to 16 Nov 2026 |
| App dir | `/opt/hhgoa-rag` |
| Service | `voice-rag.service` → `uv run uvicorn app.main:app --host 127.0.0.1 --port 8000` |

Ports 80/443 are public; SSH on 22 is restricted. The app binds **loopback only** — Caddy
terminates TLS and proxies to it. **Public port 8000 has been revoked** (verified unreachable).

## First-time provisioning

1. SSH into the instance.
2. Clone this repository into `/opt/hhgoa-rag`.
3. Copy `.env.example` to `.env` and fill in `ELEVENLABS_API_KEY` and `ANTHROPIC_API_KEY`
   (see [docs/ENVIRONMENT_VARIABLES.md](../docs/ENVIRONMENT_VARIABLES.md)).
4. Run `./deploy/setup.sh` from the repo root. It installs `uv` and Caddy, runs `uv sync`,
   builds the corpus and both FAISS indices if `data/` is empty, writes the systemd unit,
   and configures TLS.
5. Confirm: `curl https://ragingoa.duckdns.org/health`.
6. Open `/preflight` in a browser and confirm the microphone prompt appears.

`setup.sh` is safe to re-run.

## SSH

```bash
ssh -i ~/.ssh/hhgoa-rag-key.pem ubuntu@13.234.228.244
```

If this times out it is almost certainly **not** the instance — the security group pins one
residential IP, which rotates. Re-authorise:

```bash
MYIP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress --group-id sg-01967e366d79ce0c8 \
  --region ap-south-1 --protocol tcp --port 22 --cidr ${MYIP}/32
```

## Redeploying code changes

```bash
cd /opt/hhgoa-rag
git pull
uv sync                            # only when dependencies change
sudo systemctl restart voice-rag
journalctl -u voice-rag -f         # watch it come up
curl -sS https://ragingoa.duckdns.org/health
```

Startup takes ~20 s — `sentence-transformers` import and model load alone is ~17.5 s,
before FAISS indices. The unit allows `TimeoutStartSec=300`; don't mistake that for a hang.

## Secrets

`/opt/hhgoa-rag/.env` is **not** in git and must exist before the service starts — systemd
reads it via `EnvironmentFile` and refuses to start without it.

The **production** ElevenLabs key is IP-restricted to `13.234.228.244`: it works from the
instance and returns `403` anywhere else. **Local development needs the separate
unrestricted dev key.** Do not add a laptop IP to the production key — residential IPs rotate.

## Building the corpus and indices

Run these **on the instance**, not on a laptop. Measured: the instance pulls from
HuggingFace at **74 MB/s** versus **4.8 MB/s** on a residential connection, and building in
place avoids uploading GB-scale index artifacts afterwards. `data/` is gitignored.

`setup.sh` runs them automatically when `data/` is empty.

## Preflight page

`/preflight` is served by Caddy directly from `/opt/hhgoa-rag-preflight`, independently of
the app, so it works even when the backend is down. It checks HTTPS, secure context,
`getUserMedia`, `MediaRecorder` and `WebSocket`, and has a button that actually requests
microphone permission. **Open it on whatever machine you demo from, before demo day.**

## Go-live checklist

- [x] `curl -sS https://ragingoa.duckdns.org/health` returns 200
- [x] Preflight all-green and mic prompt granted on the demo machine
- [x] TS-001 / TS-002 / TS-003 pass against the live URL (answer, off-topic refusal, unsafe refusal)
- [x] Public port 8000 revoked; `curl --max-time 6 http://13.234.228.244:8000/health` times out
- [x] Benchmark run against the live HTTPS URL — see [docs/LATENCY_REPORT.md](../docs/LATENCY_REPORT.md)
- [ ] Record the demo video
- [ ] Submit: repo link, live link, both videos

## Troubleshooting

| Symptom | Cause |
|---|---|
| `503 no upstreams available` | App isn't running. `systemctl status voice-rag` |
| `503` with JSON body | App is up but indices aren't loaded — `data/` is empty |
| Service won't start | Usually missing `/opt/hhgoa-rag/.env`. `journalctl -u voice-rag -n 50` |
| No mic prompt in browser | Not a secure context. Check `/preflight` first |
| ElevenLabs `403` | Using the production (IP-restricted) key off-instance — use the dev key |
| Ingestion killed, `exit 137` | OOM. Project `passages.English_passages`, not `passages` |
| `No space left on device` | CUDA torch. `pyproject.toml` pins the CPU-only index — re-run `uv sync` |
| Certificate errors | `journalctl -u caddy`. Check `dig +short ragingoa.duckdns.org` |

Logs are in journald, not files: `journalctl -u voice-rag -f`, `journalctl -u caddy -f`.
