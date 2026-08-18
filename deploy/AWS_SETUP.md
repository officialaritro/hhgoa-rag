# AWS Setup Handover (Task 10)

This is a handover for whoever provisions AWS, so that person can work in parallel with the application code in Tasks 1-9. It covers what to provision and why; `deploy/setup.sh` (built in Task 10) is the script that actually configures the instance once it exists.

## What you need before starting

- An AWS account with permission to create EC2 instances, security groups, and key pairs (or access to an existing VPC to deploy into).
- The AWS CLI installed and configured locally (`aws configure`) with an access key that has EC2 permissions — this is **only used from your own machine to provision the instance**. It is never placed on the server or in the app's `.env` (see [`docs/ENVIRONMENT_VARIABLES.md`](../docs/ENVIRONMENT_VARIABLES.md)) — nothing in this pipeline calls an AWS API at runtime.
- An SSH key pair for connecting to the instance (create one in the EC2 console, or `aws ec2 create-key-pair`).

## What to provision

### 1. One EC2 instance — always-on, not serverless

Per the plan's hosting decision, this must be a single long-running instance, not Lambda/Fargate — a warm, in-memory FAISS index and embedding model are load-bearing for the latency target, and cold starts work directly against it.

- **Instance type:** `t3.small` (2 vCPU / 2 GiB RAM), chosen for budget reasons over the `t3.medium` a naive float32 FAISS index would need. `app.indexing.build_index` persists both chunking strategies' vectors as an int8 scalar-quantized (`IndexScalarQuantizer`, `QT_8bit`) index rather than a raw `IndexFlatIP` — ~4x smaller in memory for a measured 99.6% top-5 retrieval agreement with the uncompressed index — and `app.retrieval` loads each strategy's index+metadata once per process instead of per request. Still watch actual RAM/CPU usage during Task 9's benchmark run; size up if constrained.
- **AMI:** the latest Ubuntu LTS AMI (22.04 or 24.04) for your chosen region — pick it from the EC2 console or `aws ec2 describe-images` at launch time rather than hardcoding an AMI ID here, since AMI IDs are region-specific and change over time.
- **Region:** pick the region closest to where judges/users will actually connect from, since network latency to the instance is part of the pipeline's measured latency (PRD Open Decision 4).
- **Storage:** default EBS volume size is sufficient — the ingested corpus, chunks, and two FAISS indices are small relative to typical instance storage.

### 2. Networking — security group rules

| Direction | Port | Source | Purpose |
|---|---|---|---|
| Inbound | 22 (SSH) | Your IP only (not `0.0.0.0/0`) | Deployment and debugging access |
| Inbound | 8000 | `0.0.0.0/0` | The live demo link (matches the app's `PORT`, Runtime Environment section of the plan) |
| Outbound | all | `0.0.0.0/0` | Default — the app needs outbound HTTPS to reach the ElevenLabs and Anthropic APIs |

If you later add a reverse proxy with TLS (see Optional below), open 443 instead of/alongside 8000 and keep 8000 internal.

### 3. A stable address for the live link

Allocate an **Elastic IP** and associate it with the instance (or use the instance's public DNS name), so the submitted "live working link" doesn't change if the instance is stopped and restarted. A bare public IP that changes on reboot will silently break an already-submitted link — and the task's "no resubmissions" rule means that mistake can't be corrected afterward.

### 4. IAM

No IAM role or instance profile is required for the application itself — it makes no AWS API calls at runtime (FAISS is local; no S3 or other AWS service is used by this plan). Only your own local AWS CLI credentials, used to provision the instance, need EC2 permissions.

### 5. Systemd service (built in Task 10's `deploy/setup.sh`)

The instance should run the app as a systemd service (not a manually-started process in a terminal), so it survives SSH disconnects and instance reboots:

- Start command: `uvicorn app.main:app --host 0.0.0.0 --port 8000` (matches the plan's Runtime Environment section)
- `Restart=on-failure` and `WantedBy=multi-user.target` so it comes back up automatically

`deploy/setup.sh` (Task 10) is where this gets scripted — this document is the provisioning context that script assumes.

## Optional (not required for MVP submission)

- **Domain name + TLS:** if you want `https://yourdomain.com` instead of a raw IP/DNS, add a Route 53 record (or point an existing domain) plus an nginx reverse proxy and a Let's Encrypt certificate via `certbot`. The PRD does not require this — a plain HTTP link on the Elastic IP/public DNS satisfies "live working link."
- **Autoscaling / load balancing:** explicitly out of scope per the plan — one instance is deliberately sufficient for a demo audience.

## Cost

AWS pricing varies by region and changes over time — check the [EC2 pricing page](https://aws.amazon.com/ec2/pricing/on-demand/) for current `t3.medium`/`t3.large` on-demand rates in your chosen region rather than relying on a fixed number here. For a ~4-day build-and-demo window, cost is unlikely to be the binding constraint.

## Handback

Once the instance, Elastic IP, and security group exist, hand back: the instance's public IP/DNS, the region, and confirmation that port 22 (from the deployer's IP) and port 8000 (public) are open. Task 10 takes it from there — deploying the code, running `deploy/setup.sh`, and pointing the systemd service at it.
