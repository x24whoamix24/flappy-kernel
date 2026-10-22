# Deployment

The arcade is one container on one H100. This page explains what runs inside it, how
a player's browser connects to the GPU kernel and back, and how to deploy it with
Helm, Docker or on a plain machine.

## What runs, and how a browser connects back

```
 browser ── page + one WebSocket (/ws) ──►  port-forward, LoadBalancer or Ingress
                                                          │
                                         Service :80 ──► pod :8080
┌──────────────────────── pod: one container, one H100 ───┼─────────────────────────┐
│ deploy/serve.sh starts two processes                    ▼                         │
│                                                                                   │
│   Mirage process (Python)                     web server (server/app.py)          │
│   launches the megakernel once     ◄─queue/── writes each question to a file      │
│   tokenizes and submits questions  ─done/──►  reads finished answers              │
│          │ launch                                  ▲ frames, tokens, counters     │
│          ▼                                         │ flaps, slot controls ▼       │
│   ┌──────────────── GPU: one kernel ─────────────┐ ┌────────────────────────────┐ │
│   │ LLM workers · trainer blocks · arcade block  │◄┤ /dev/shm/flappy            │ │
│   │                                              ├►│ pinned shared memory       │ │
│   └──────────────────────────────────────────────┘ └────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────────────┘
```

`deploy/serve.sh` starts both processes in one container because they share memory
through `/dev/shm`. If either one exits, it stops the other and exits too, so Docker or
Kubernetes restarts the whole arcade.

A player's actions take these paths:

1. **Opening the page.** The browser loads the page from the web server and opens one
   WebSocket to `/ws`. Everything after that goes over this one connection.
2. **A flap.** The web server writes it into the player's slot in `/dev/shm/flappy`.
   The kernel reads that memory directly (it is registered with CUDA), so the game sees
   the flap at its next tick, 30 times a second.
3. **A frame.** The arcade block draws each game's frame into the same shared memory and
   then bumps a frame number. The web server notices the new number, encodes the frame
   as JPEG once and sends it to everyone watching that game.
4. **A question.** The web server writes it to `queue/` as a file. The Mirage process
   tokenizes it and hands it to the kernel that is already running, through Mirage's
   pinned request ring; there is no new launch. As the LLM decodes, the arcade block
   copies each token into shared memory, and the web server streams the text to the
   player. The finished answer goes to `done/`.
5. **Health.** `/healthz` answers 200 while frames keep coming from the GPU and 503
   when they stop. The Kubernetes probes use it.

[architecture.md](architecture.md) goes further into the kernel side.

## Kubernetes with Helm

```bash
helm install flappy-kernel deploy/helm/flappy-kernel
kubectl logs -f deploy/flappy-kernel                 # follow the first start
kubectl port-forward svc/flappy-kernel 8080:80       # then open http://localhost:8080
```

The chart creates one Deployment (one pod, one GPU) and one Service. By default the
pod runs the stock `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` image, so there is no
image to build. On each start the pod:

1. downloads the arcade's source from `source.url` (a release archive);
2. runs `deploy/install.sh`, which installs Mirage prebuilt with the patch from the
   GitHub release, in about a minute, and builds Mirage from source if no prebuilt one
   matches;
3. runs `deploy/serve.sh`, which loads the model, compiles the megakernel and serves.

The pod reports Ready once frames come from the GPU. The startup probe allows 30
minutes, which covers downloading the model on the first start.

| Value | Default | Meaning |
|---|---|---|
| `source.url` | this release's source archive | where the pod downloads the arcade from; `""` runs the image's own copy (see below) |
| `image.repository`, `image.tag` | `pytorch/pytorch`, `2.8.0-cuda12.8-cudnn9-devel` | container image |
| `env` | `MODEL`, `MAX_SEQ`, `LLM_WORKERS`, `COTENANT_BLOCKS` | environment of `serve.sh` ([configuration.md](configuration.md)) |
| `serverArgs` | `[]` | extra web server options, e.g. `[--allow-origin, https://arcade.example.com, --trust-proxy]` |
| `modelCache.existingClaim` | `""` | a PersistentVolumeClaim that keeps the model across restarts |
| `modelCache.hostPath` | `""` | or a directory on the node; with neither, each restart downloads the model again |
| `service.type`, `service.port` | `ClusterIP`, `80` | `LoadBalancer` or `NodePort` to reach it without port-forward |
| `resources` | 16 CPUs, 64 GiB, 1 GPU | requests and limits |
| `runtimeClassName`, `nodeSelector`, `tolerations` | empty | to place the pod on an H100 node |

For example, keeping the model on a volume and pinning the pod to H100 nodes:

```bash
helm install flappy-kernel deploy/helm/flappy-kernel \
  --set modelCache.existingClaim=hf-cache \
  --set nodeSelector."nvidia\.com/gpu\.product"=NVIDIA-H100-80GB-HBM3
```

**Your own image.** To skip the install at every start, or if the pod cannot reach
GitHub, build the repository's `Dockerfile`, push it to your registry, and install
with `--set image.repository=<registry>/flappy-kernel --set image.tag=<tag> --set source.url=`.

**Upgrade and remove:** `helm upgrade flappy-kernel deploy/helm/flappy-kernel` restarts
the pod with the new settings; `helm uninstall flappy-kernel` removes everything the
chart created.

## Reaching it from elsewhere

- **`kubectl port-forward`** is the simplest way and needs no other setup. It is slower
  (each WebSocket message passes through the Kubernetes API server), so the game reacts
  less quickly than over a direct connection.
- **`service.type=LoadBalancer`** (or `NodePort`) gives the arcade its own address.
  Phones on the same network can then open it.
- **An Ingress** works if it passes WebSockets through (most do). The page is then
  served under your host name, so add `--allow-origin https://<your host>` and
  `--trust-proxy` to `serverArgs`. With `--trust-proxy`, the per-address limits use
  each player's address rather than the proxy's.

The arcade has no login. Before putting it on a public address, put it behind your own
access control.

## Docker

```bash
docker build -t flappy-kernel .
docker run --gpus all --shm-size 16g -p 8080:8080 -v hf-cache:/root/.cache/huggingface flappy-kernel
```

The build runs `deploy/install.sh`, so the image contains Mirage and the compiled
`kernel/` tools. The `hf-cache` volume keeps the model between runs.

## A plain machine or an existing GPU pod

On a Linux machine with an H100, CUDA 12.8 and a Python environment with PyTorch 2.8
(for example the `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` image, a venv or a conda
env), run as root:

```bash
deploy/install.sh        # packages, Mirage (prebuilt or from source), kernel/
deploy/serve.sh          # serves on port 8080
```

To copy your working tree into a GPU pod that is already running, which is handy while
changing the code:

```bash
deploy/setup_pod.sh <pod>                        # copies the repository, runs install.sh
kubectl exec <pod> -- /root/fk/deploy/serve.sh
kubectl port-forward pod/<pod> 8080:8080
```

`install.sh` options (environment): `MIRAGE` (where Mirage goes, `/root/mirage`),
`MIRAGE_PREBUILT=0` (always build from source), `RELEASES` (where the prebuilt archive
is downloaded from), `MAX_JOBS` (parallel compile jobs when building Mirage).

## Check that it works

```bash
curl -s localhost:8080/healthz                       # {"stalled": false}
npm install && npx playwright install chrome         # once
node tests/e2e.test.mjs http://localhost:8080 --llm  # drives Chrome against the arcade
```

The end-to-end test checks frames, flaps, a full race, the one-game-per-address limit,
live training and an LLM answer.

## Troubleshooting

- **"The GPU process is restarting" on the page, and `/healthz` returns 503:** the
  Mirage process stopped producing frames. Its output is in the container log.
  `serve.sh` exits when either process ends, so Docker (`--restart`) or Kubernetes
  restarts both.
- **"... resident blocks, but this GPU has N SMs"** at startup: the GPU has fewer than
  132 SMs (an H100 PCIe has 114). Lower `LLM_WORKERS` and `COTENANT_BLOCKS` so that
  `LLM_WORKERS + 4 + COTENANT_BLOCKS` fits.
- **Mirage stops producing tokens:** another CUDA process (or MPS) is sharing the GPU.
  Mirage's persistent kernels need the GPU to themselves ([results.md](results.md)).
- **The page loads but the WebSocket is refused (403):** the page is served under
  another origin. Add `--allow-origin https://your.host` (Helm: `serverArgs`; otherwise
  an argument to `serve.sh`).
- **The pod is never Ready:** `kubectl logs deploy/flappy-kernel` shows which step it is
  on. Without `modelCache`, every restart downloads the model again.
