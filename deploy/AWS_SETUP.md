# AWS Setup Handover (Task 10)

This is a handover for whoever provisions AWS, so that person can work in parallel with the application code in Tasks 1-9. It covers what to provision and why; `deploy/setup.sh` (built in Task 10) is the script that actually configures the instance once it exists.

## What you need before starting

- An AWS account with permission to create EC2 instances, security groups, and key pairs (or access to an existing VPC to deploy into).
- The AWS CLI installed and configured locally (`aws configure`) with an access key that has EC2 permissions — this is **only used from your own machine to provision the instance**. It is never placed on the server or in the app's `.env` (see [`docs/ENVIRONMENT_VARIABLES.md`](../docs/ENVIRONMENT_VARIABLES.md)) — nothing in this pipeline calls an AWS API at runtime.
- An SSH key pair for connecting to the instance (create one in the EC2 console, or `aws ec2 create-key-pair`).

## What to provision

### 1. One EC2 instance — always-on, not serverless

Per the plan's hosting decision, this must be a single long-running instance, not Lambda/Fargate — a warm, in-memory FAISS index and embedding model are load-bearing for the latency target, and cold starts work directly against it.

- **Instance type:** `m7i-flex.large` (2 vCPU / 8 GB RAM, Sapphire Rapids). Chosen because it is free-tier-eligible on this account *and* beats the originally-planned `t3.medium` on both memory and clock — see the Free Plan constraint in the change log before changing it.
- **AMI:** the latest Ubuntu LTS AMI (22.04 or 24.04) for your chosen region — pick it from the EC2 console or `aws ec2 describe-images` at launch time rather than hardcoding an AMI ID here, since AMI IDs are region-specific and change over time.
- **Region:** pick the region closest to where judges/users will actually connect from, since network latency to the instance is part of the pipeline's measured latency (PRD Open Decision 4).
- **Storage:** the 8 GiB default is **not** sufficient. Grow the root volume to 20 GiB. Note this is one-way — EBS volumes can be grown online but never shrunk.

### 2. Networking — security group rules

| Direction | Port | Source | Purpose |
|---|---|---|---|
| Inbound | 22 (SSH) | Your IP only (not `0.0.0.0/0`) | Deployment and debugging access |
| Inbound | 80 | `0.0.0.0/0` | Let's Encrypt HTTP-01 challenge + redirect to HTTPS |
| Inbound | 443 | `0.0.0.0/0` | The live demo link, TLS-terminated by Caddy |
| Outbound | all | `0.0.0.0/0` | Default — the app needs outbound HTTPS to reach the ElevenLabs and Anthropic APIs |

Port 8000 should **not** be open to the world — uvicorn binds `127.0.0.1` and Caddy proxies to it.

> **SSH lock-out warning.** The rule pins a single Indian residential IP (`152.58.139.215/32`), which rotates. If SSH starts timing out, the instance is almost certainly fine — re-run `aws ec2 authorize-security-group-ingress` with your current address.

### 3. A stable address for the live link

Allocate an **Elastic IP** and associate it with the instance (or use the instance's public DNS name), so the submitted "live working link" doesn't change if the instance is stopped and restarted. A bare public IP that changes on reboot will silently break an already-submitted link — and the task's "no resubmissions" rule means that mistake can't be corrected afterward.

### 4. IAM

No IAM role or instance profile is required for the application itself — it makes no AWS API calls at runtime (FAISS is local; no S3 or other AWS service is used by this plan). Only your own local AWS CLI credentials, used to provision the instance, need EC2 permissions.

### 5. Systemd service (built in Task 10's `deploy/setup.sh`)

The instance should run the app as a systemd service (not a manually-started process in a terminal), so it survives SSH disconnects and instance reboots:

- Start command: `uvicorn app.main:app --host 127.0.0.1 --port 8000` (matches the plan's Runtime Environment section) — loopback only, Caddy fronts it
- `Restart=on-failure` and `WantedBy=multi-user.target` so it comes back up automatically

`deploy/setup.sh` (Task 10) is where this gets scripted — this document is the provisioning context that script assumes.

## TLS is mandatory (this section was previously marked optional — it is not)

`getUserMedia`/`MediaRecorder` are **secure-context-only** browser APIs. Served over plain HTTP on an IP or the EC2 public DNS, `navigator.mediaDevices` is `undefined`: the demo has no microphone and fails silently rather than with a debuggable error. A plain HTTP link does **not** satisfy "live working link" for a voice project.

Let's Encrypt refuses to issue certificates for `*.compute.amazonaws.com`, so the instance's own public DNS name cannot be used. Register a free DuckDNS subdomain pointing at the Elastic IP and let Caddy auto-provision the certificate — see Task 10 of the implementation plan.

## Optional (not required for MVP submission)

- **Autoscaling / load balancing:** explicitly out of scope per the plan — one instance is deliberately sufficient for a demo audience.

## Cost

The instance type in use (`m7i-flex.large`) and the 20 GiB gp3 volume are both within free-tier-eligible bounds on this account's plan. Check the [EC2 pricing page](https://aws.amazon.com/ec2/pricing/on-demand/) for current `ap-south-1` rates if anything is added beyond this. For a ~4-day build-and-demo window, cost is not the binding constraint.

## Handback

Once the instance, Elastic IP, and security group exist, hand back: the instance's public IP/DNS, the region, and confirmation that port 22 (from the deployer's IP) and ports 80/443 (public) are open. Task 10 takes it from there — deploying the code, running `deploy/setup.sh`, and pointing the systemd service at it.

---

## Infrastructure change log

Applied 2026-08-19. Steps 1–5 are **done and verified**; steps 6–7 remain and depend on the application being deployed.

| # | Change | Status |
|---|---|---|
| 1 | Instance type → `m7i-flex.large` (7.6 GB RAM) | **Done** |
| 2 | Root volume 8 → 20 GiB gp3 | **Done** |
| 3 | 2 GB swapfile, `vm.swappiness=10` | **Done** |
| 4 | DuckDNS `ragingoa.duckdns.org` → `13.234.228.244` | **Done** |
| 5 | Open ports 80 and 443 | **Done** |
| 6 | Install Caddy, issue Let's Encrypt certificate | **Done** — valid to 16 Nov 2026 |
| 7 | Revoke public port 8000 | **Done** — verified unreachable; only 22/80/443 remain |

Verified final state: `i-09e157bfae9bb82a6`, `ap-south-1b`, `m7i-flex.large`, Xeon Platinum 8488C (Sapphire Rapids), 2 vCPU, 7778 MiB RAM, 19 G filesystem with 15 G free, 2 G swap active and persisted in `/etc/fstab`, Elastic IP `13.234.228.244` still associated.

### The Free Plan constraint (read this before changing instance type again)

This account is on the **AWS Free Plan**, which allows only free-tier-eligible instance types — both to launch *and* to resize into. The rejection message is actively misleading:

```
FreeTierRestrictionError: This operation is not available for free plan accounts.
```

That reads as `ModifyInstanceAttribute` being blocked wholesale. It is not — the *target type* was the problem. `t3.medium` is not free-tier-eligible; resizing to a type that is works fine. A `RunInstances --dry-run` for `t3.medium` also returns "would have succeeded", because DryRun does not evaluate the free-tier check — do not trust it as a pre-flight for this.

Always check the permitted list first:

```bash
aws ec2 describe-instance-types --region ap-south-1 \
  --filters Name=free-tier-eligible,Values=true \
  --query 'InstanceTypes[].{Type:InstanceType,vCPU:VCpuInfo.DefaultVCpus,MiB:MemoryInfo.SizeInMiB}' --output table
```

In `ap-south-1` that currently yields `t3.micro` (1 GB), `t3.small` (2 GB), `t4g.micro`/`t4g.small` (ARM), **`c7i-flex.large` (4 GB)**, and **`m7i-flex.large` (8 GB)**. The two flex types beat the originally-planned `t3.medium` on both memory and clock speed, so the Free Plan cost nothing here.

### AWS CLI session expiry

This account authenticates via `aws login` (`login_session` in `~/.aws/config`), and the session goes stale after a few minutes idle, failing with `CreateOAuth2Token ... authorization grant is invalid`. A single `aws sts get-caller-identity` re-warms it. When scripting a batch of calls, wrap them:

```bash
awsr() {
  local out rc; out=$(aws "$@" 2>&1); rc=$?
  if [ $rc -ne 0 ] && printf '%s' "$out" | grep -q "CreateOAuth2Token"; then
    aws sts get-caller-identity >/dev/null 2>&1
    out=$(aws "$@" 2>&1); rc=$?
  fi
  printf '%s\n' "$out"; return $rc
}
```

### Step 6 — Caddy and TLS (done)

`deploy/setup.sh` installs and configures Caddy from `deploy/Caddyfile`. The certificate
for `ragingoa.duckdns.org` was issued by Let's Encrypt and renews automatically. Caddy
also serves `/preflight` from `/opt/hhgoa-rag-preflight`, independently of the app, so
the browser-precondition check still works when the backend is down.

The app binds `127.0.0.1:8000` and is reached only through Caddy.

### Step 7 — public port 8000 revoked (done)

**Only after step 6 is confirmed working.** Port 8000 is currently the only route to the app; revoking it before Caddy serves 443 leaves no working path in.

```bash
aws ec2 revoke-security-group-ingress --group-id sg-01967e366d79ce0c8 --region ap-south-1 \
  --protocol tcp --port 8000 --cidr 0.0.0.0/0
```

### Verification

```bash
curl -sS https://ragingoa.duckdns.org/health          # expect 200, valid cert
curl -sS --max-time 5 http://13.234.228.244:8000/health || echo "correctly unreachable"
ssh -i ~/.ssh/hhgoa-rag-key.pem ubuntu@13.234.228.244 'free -m; df -h /; swapon --show'
```

Then open the HTTPS URL in a browser and confirm the microphone permission prompt actually appears — that is the single check this whole section exists for.

### If SSH stops working mid-build

The rule pins one residential IP, which rotates. This is not an instance failure:

```bash
MYIP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress --group-id sg-01967e366d79ce0c8 --region ap-south-1 \
  --protocol tcp --port 22 --cidr ${MYIP}/32
```
