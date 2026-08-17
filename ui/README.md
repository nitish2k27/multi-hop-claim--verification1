# ui/

What FastAPI serves at `/`. Two files matter:

| Path | What it is | When it is served |
|---|---|---|
| `index.html` | Single-file fallback UI, no build step | When `dist/` is absent |
| `dist/` | Vite build output from [`frontend/`](../frontend/) | Whenever it exists |

`serve/api.py:162` prefers `dist/index.html` and falls back to `index.html`,
so a clone with no Node toolchain still gets a working interface. The React
build is an upgrade, not a requirement.

`dist/` is gitignored — it is generated.

## Building the React frontend

```bash
cd frontend
npm install
npm run build      # writes ../ui/dist/
```

Then start the API and open it:

```bash
uvicorn serve.api:app --reload
# http://127.0.0.1:8000
```

## Developing against a live reload

```bash
cd frontend && npm run dev      # http://localhost:5173
```

`frontend/vite.config.js` proxies `/verify`, `/health` and `/download` to
port 8000, so the browser sees a single origin and never issues a CORS
preflight. The CORS middleware in `serve/api.py` exists only for anyone who
bypasses that proxy.

## How the page talks to the backend

`POST /verify/stream` with a multipart body, answered as server-sent events —
one frame per completed graph node, then a terminal frame carrying the full
result. `frontend/src/api.js` uses `fetch` rather than `EventSource` because
EventSource cannot POST and this endpoint accepts a file upload.

[`docs/execution-trace.html`](../docs/execution-trace.html) follows one claim
through the whole path, file by file.
