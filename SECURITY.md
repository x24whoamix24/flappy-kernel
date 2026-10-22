# Security

The arcade's web server (`server/app.py`) is meant to be reachable by players over the
internet, so reports about it are welcome.

## Reporting a vulnerability

Please do not open a public issue. Use GitHub's private vulnerability reporting
("Report a vulnerability" under the repository's Security tab) with a description, the
steps to reproduce, and the impact you expect.

## What the server does and does not protect

- **In place:** per-address limits on connections, games, questions (280 characters,
  20 queued; `--question-interval` adds a per-address interval, off by default) and WebSocket messages per second; chat-template markers
  removed from questions; an origin check on WebSocket connections; `--trust-proxy`
  reads the client's address from the entry your proxy appends to `X-Forwarded-For`.
- **Not in place:** authentication, TLS (terminate it in a reverse proxy), and any
  moderation. Players' questions and the LLM's answers are shown to every viewer.
- **Trust boundary:** the server and the GPU process share one memory segment
  (`/dev/shm`). Anything that can write it controls the game slots and the LLM
  feed, so run both in one container or host that nothing else can reach.

Put the arcade behind your own access control before exposing it publicly.
