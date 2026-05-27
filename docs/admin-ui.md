# Admin UI

A local, password-gated web dashboard for operating the reranker, served by the same FastAPI app at **`/admin`**. It shows live device (GPU/MPS/CPU) memory quota, the batch queue (running + waiting), throughput/latency, lets you view and edit `config.yml`, and tails logs.

## Enabling it

The admin UI is **disabled unless a password is set**. It is an env-only secret:

```bash
export RERANKER_ADMIN_PASSWORD='choose-a-strong-password'
./run.sh
```

With no `RERANKER_ADMIN_PASSWORD`, every `/admin/*` route returns `503 Admin UI not configured`.

Optional knobs:

| Setting | Env | Default | Effect |
|---|---|---|---|
| Admin password | `RERANKER_ADMIN_PASSWORD` | unset | Enables `/admin`; gates all admin routes. Never written to `config.yml`. |
| Session lifetime | `RERANKER_ADMIN_SESSION_TTL_HOURS` | `12` | How long a login stays valid. |

Install the admin extra if you manage deps via extras: `pip install '.[admin]'` (adds `jinja2`, `python-multipart`). For NVIDIA utilization/temperature/power on the dashboard, also `pip install '.[gpu-metrics]'` (adds `pynvml`); without it the dashboard still shows memory used/total/free.

## Access & security model

- **Bind locally.** Run the service on `127.0.0.1` (or behind your VPN/SSH tunnel). The password is defense-in-depth, not a substitute for network isolation.
- **Login** at `/admin/login` sets an HMAC-signed, `HttpOnly`, `SameSite=Strict` session cookie. The signing key is derived from the password, so changing the password invalidates all existing sessions.
- **Brute-force throttle:** 5 failed logins per IP within 5 minutes returns `429`.
- The introspection endpoints `/stats` and `/info` also require the bearer `RERANKER_API_KEY` when one is configured; `/health`, `/ready`, `/live` stay open for probes. `/docs` can be disabled with `RERANKER_ENABLE_DOCS=false`.

## Pages

- **Dashboard** (`/admin`) — device quota bar (used / total / free, plus util/temp/power on NVIDIA when `pynvml` is present), p50/p95 latency, throughput (pairs/s), and two live tables: **batches running** (`batch_id`, requests, pairs, elapsed) and **requests waiting** (`request_id`, docs, waited). Refreshes every 2 s via HTMX polling.
- **Config** (`/admin/config`) — every setting with its current value, **source** (`env` / `yaml` / `default`), and an **apply badge** (`hot` = applied by reloading the engine, `restart` = needs a process restart). Secrets are shown as `***set***`.
- **Logs** (`/admin/logs`) — live tail of the rotating files in `log_dir`, with a substring filter.

## Applying config changes

- `POST /admin/api/config` with `{"updates": {"<setting>": <value>}}` writes the change into `config.yml` and, for **hot** settings (batch sizing, timeouts, log level, precision, model/device, …), reloads the engine in-process so it takes effect immediately. Unknown keys and secrets are rejected with `400`.
- Settings marked **restart** (`host`, `port`, `workers`) require a process restart. `POST /admin/api/restart` triggers the graceful drain (in-flight requests finish, new ones get `503`) and exits so supervisord respawns the service with the new config.

## Endpoints (all under `/admin`, all require the session cookie)

| Method | Path | Purpose |
|---|---|---|
| GET | `/admin` | Dashboard page |
| GET | `/admin/config`, `/admin/logs` | Config / Logs pages |
| POST | `/admin/login`, `/admin/logout` | Session login / logout |
| GET | `/admin/api/resources` | Device memory + throughput + latency JSON |
| GET | `/admin/api/queue` | `{waiting, running}` snapshot |
| GET | `/admin/api/config` | Effective config snapshot |
| POST | `/admin/api/config` | Apply config edits (+ engine reload) |
| POST | `/admin/api/restart` | Graceful drain + process restart |
| GET | `/admin/api/logs/tail?lines=&q=` | Recent log lines, filtered |

The dashboard's live widgets are served as HTML partials under `/admin/partials/*` (polled by HTMX). All assets (HTMX) are vendored under `src/admin/static/`, so the UI works fully offline.
