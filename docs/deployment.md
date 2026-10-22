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

## What the GPU and the cluster need

- **One whole H100 SXM or NVL (132 SMs) per arcade.** The GPU must not be shared:
  Mirage's persistent kernel stops producing valid tokens when another process runs
  on it. A MIG slice, a time-sliced GPU or MPS will not work. Clusters running the
  NVIDIA device plugin or GPU Operator expose whole GPUs as `nvidia.com/gpu` by default.
  If yours shares GPUs, pin the pod to nodes that do not; GPU Feature Discovery labels
  them `nvidia.com/gpu.sharing-strategy=none`. An H100 PCIe (114 SMs) needs smaller
  settings ([Troubleshooting](#troubleshooting)).
- **A pod running as root.** The pod installs system packages at start, so a namespace
  that enforces the `restricted` Pod Security level rejects it. `baseline` allows it,
  unless you use `modelCache.hostPath`.
- **Network access from the pod:** GitHub (the source and the prebuilt Mirage:
  `github.com` and the hosts it redirects downloads to), Ubuntu's package mirrors, PyPI, and `huggingface.co` on the first
  start. A source build of Mirage also needs `sh.rustup.rs`, `crates.io` and the
  Mirage repository. Proxies set with `HTTP_PROXY`/`HTTPS_PROXY` in `env` are used.
  Without network access, use your own image and a model cache filled beforehand, and
  set `env.HF_HUB_OFFLINE: "1"`.

## Kubernetes with Helm

```bash
helm install flappy-kernel deploy/helm/flappy-kernel
kubectl logs -f deploy/flappy-kernel                 # follow the first start
kubectl port-forward svc/flappy-kernel 8080:80       # then open http://localhost:8080
```

The chart creates one Deployment (one pod, one GPU) and one Service. By default the
pod runs the stock `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` image, so there is no
image to build. On each start the pod:

1. downloads the arcade's source for the chart's version from GitHub;
2. runs `deploy/install.sh`, which installs system and Python packages and Mirage. It
   downloads Mirage prebuilt with the patch from the same release (about a minute in
   all); without one, it builds Mirage from source (about 5 minutes on 64 cores, longer
   on fewer);
3. runs `deploy/serve.sh`, which loads the model, compiles the megakernel and serves.

On one H100 node with the model already cached, the pod was Ready in under two
minutes. The pod reports Ready once frames come from the GPU; the startup probe waits
up to `startupTimeoutMinutes` (60).

| Value | Default | Meaning |
|---|---|---|
| `source.enabled`, `source.url` | `true`, the chart version's GitHub archive | download and install the arcade at start; `false` runs your own image's copy (see below) |
| `image.repository`, `image.tag`, `imagePullSecrets` | `pytorch/pytorch`, `2.8.0-cuda12.8-cudnn9-devel`, none | container image |
| `env` | `MODEL`, `MAX_SEQ`, `LLM_WORKERS`, `COTENANT_BLOCKS` | environment of `serve.sh` ([configuration.md](configuration.md)) and `install.sh` (below) |
| `extraEnv` | `[]` | more environment in the container's own form, e.g. `HF_TOKEN` from a Secret |
| `serverArgs` | `[]` | extra web server options, e.g. `[--allow-origin, https://arcade.example.com, --trust-proxy]` |
| `modelCache.existingClaim` | `""` | a PersistentVolumeClaim that keeps the model |
| `modelCache.size`, `modelCache.storageClass` | `""` | or let the chart create one (e.g. `40Gi`); it is kept when you uninstall |
| `modelCache.hostPath` | `""` | or a directory on the node. With none of the three, each new pod downloads the model again |
| `service.type`, `service.port` | `ClusterIP`, `80` | `LoadBalancer` or `NodePort` to reach it without port-forward |
| `resources` | 16 CPUs, 64 GiB (96 GiB limit), 1 GPU | requests and limits |
| `startupTimeoutMinutes` | `60` | how long the first start may take |
| `runtimeClassName` | `""` | `nvidia` where the NVIDIA runtime is not the node's default (some k3s, RKE2 and containerd setups) |
| `nodeSelector`, `affinity`, `tolerations` | empty | place the pod on an H100 node |

A typical install on a mixed GPU cluster keeps the model on a volume, accepts the
GPU nodes' taint and requires an H100:

```yaml
# values.yaml, for helm install flappy-kernel deploy/helm/flappy-kernel -f values.yaml
modelCache: {size: 40Gi}
tolerations: [{key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}]
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - {key: nvidia.com/gpu.product, operator: In, values: [NVIDIA-H100-80GB-HBM3, NVIDIA-H100-NVL]}
```

**Your own image.** To skip the install at every start, or when the pod cannot reach
the internet, build the repository's `Dockerfile`, push it to your registry, and install
with `--set image.repository=<registry>/flappy-kernel --set image.tag=<tag>
--set source.enabled=false` (and `imagePullSecrets` if the registry needs them).

**Upgrade and remove:** `helm upgrade flappy-kernel deploy/helm/flappy-kernel` restarts
the pod with the new settings; `helm uninstall flappy-kernel` removes everything the
chart created except the model volume.

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

Needs the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
docker build -t flappy-kernel .
docker run -d --restart unless-stopped --gpus '"device=0"' --shm-size 16g -p 8080:8080 \
  -v hf-cache:/root/.cache/huggingface flappy-kernel
```

The build runs `deploy/install.sh`, so the image contains Mirage and the compiled
`kernel/` tools; it needs no GPU. The image is for x86-64 only: on an Apple Silicon Mac,
build with `docker build --platform linux/amd64` (slow; better on the GPU machine).
`--gpus '"device=0"'` gives the arcade one whole GPU; `--restart` restarts it if it stops;
the `hf-cache` volume keeps the model between runs.

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

`install.sh` can run again; an installed Mirage is kept. Its settings (environment):
`MIRAGE` (where Mirage goes, `/root/mirage`), `MIRAGE_PREBUILT=0` (always build from
source), `RELEASES` (where the prebuilt archive is downloaded from; the release named
in `VERSION`), `MAX_JOBS` (parallel compile jobs when building Mirage; by default the
CPU count, at most one per 4 GB of memory).

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
  on. A source build of Mirage or a slow model download can outlast
  `startupTimeoutMinutes`; raise it, set a `modelCache`, or use your own image.
- **The pod stays Pending:** no node has a free `nvidia.com/gpu` that matches
  `nodeSelector`/`affinity`, or the GPU nodes' taint is not in `tolerations`
  (`kubectl describe pod` says which).
