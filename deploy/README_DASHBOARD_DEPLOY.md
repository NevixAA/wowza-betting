# Wowza dashboard - SLIM deployment

Owner-approved 2026-10-26. Streamlit Community Cloud could no longer finish a fresh clone of this repo
(~710 MB of history plus a ~619 MB working tree): the app log looped `Cloning repository...` every ~6.5
minutes and the dashboard never came up. The dashboard now deploys from a small **generated** repo,
`NevixAA/wowza-dashboard`, that carries the dashboard **code** only. The **data** stays in this repo and
is downloaded at runtime from its public raw URLs.

Nothing about production changes. v9 keeps committing its output exactly as before; the dashboard reads
the same files, only fetched over HTTPS instead of cloned.

## How it works

| piece | file | what it does |
|---|---|---|
| Manifest | `deploy/dashboard_manifest.json` | Every code file the dashboard imports and every v9 data file it reads, with sizes. Generated, never hand-edited. |
| Manifest generator | `scripts/dashboard_manifest.py` | `--write` renders every page with Streamlit's `AppTest` (every tab, widget interactions) under an audit hook that records each file opened, stat'd or read as parquet, plus the static import closure of `app.py` and every routed page. `--check` fails if something is missing. `--static` is the fast check. |
| Guard test | `tests/test_dashboard_manifest.py` | Fails if a page imports a module, or names a v9 data file, that the manifest does not carry. Also tests the mirror and the builder. |
| Runtime mirror | `dashboard_data/remote_mirror.py` | In slim mode only: downloads the manifest's data files into the same relative paths, so every page keeps reading local paths unchanged. |
| Slim tree builder | `scripts/build_dashboard_repo.py OUT_DIR` | Copies the manifest's code, the `deploy/SLIM` marker, `deploy/requirements-dashboard.txt` (as `requirements.txt`), `.streamlit/config.toml`, a README and a `.gitignore` for the mirrored data. No data, no models. |

### Slim mode

On when the marker file `deploy/SLIM` exists (the builder writes it; this repo never has it) or when
`WOWZA_SLIM=1`. `WOWZA_SLIM=0` forces it off. It is **impossible in a full v9 checkout**: if
`.github/workflows/predict.yml` exists the mirror refuses to run whatever the environment says, because
mirroring into a full checkout would overwrite tracked output files with main's copies.

### Data freshness

* `app.py` calls `remote_mirror.ensure_fresh()` before navigation on every rerun.
* First call in a fresh process (cold start): it **waits** for one full sync (spinner "Fetching the latest
  Wowza data..."; budget 120 s, `WOWZA_MIRROR_COLD_BUDGET_S`), because `config.py` and the pages read files
  at import time.
* After that: at most **one sync every 10 minutes per process** (`WOWZA_MIRROR_TTL_MIN`), always in a
  **background thread**. A render never waits for it; stale data is served meanwhile.
* A sync is cheap:
  1. one GitHub API call reads main's head SHA - unchanged since the last sync means **zero downloads**;
  2. moved, and the compare API answers - **only the manifest files that changed** are downloaded;
  3. API unavailable (the anonymous API allows 60 calls/hour per IP, and Streamlit Cloud's egress IPs are
     shared) - every file is re-requested with `If-None-Match` on its recorded ETag, so unchanged files
     answer `304` with no body.
  Downloads are pinned to one commit SHA when it is known, so a sync is a consistent snapshot.
* Every file is written to a temp file and atomically renamed. Any failure keeps the old copy.
* Pages still cache their reads with `st.cache_data` (TTL 2-60 min, as before), so a new file shows up
  within roughly one page-cache TTL of being mirrored.
* What a page shows as "fresh" still comes from the data **content** (newest record, recorded timestamps),
  never from file mtime. The sync record `deploy/.mirror_sync.json` (sync time, SHA, ETags) is reported in
  the admin sidebar ("Data mirror: synced ... UTC · v9 abc1234") and is never used to judge the data.
* Pro and v11 files are unchanged: `dashboard_data/core.resolve_file` already fetches them from their own
  public repos (30-minute disk cache).

## What does not work slim (and what the app does instead)

| feature | why | slim behaviour |
|---|---|---|
| Fantasy -> Club squads ("Load club squads") | reads `player_history.parquet`, a ~173 MB directory, ~1.35 GB in RAM - over the app's memory | shows "Club squads read the full player history, which this deployment does not carry"; every other Fantasy tool works |
| Tips -> "Admin actions" (Run Predict Now / Update Results Now) | run `pipeline.py` / `update_results.py` in place | hidden (they appear only when `pipeline.py` exists) |
| Live -> "Run Live Scan Now" | runs `src/live_scanner.py` | hidden (only when that file exists) |
| Overview -> "Weekly research panel" lock status | the live research-lock check fingerprints workflows, threshold files and a config subprocess | uses the lock result CI recorded in `output/system_health.json` (written daily by `evidence_refresh.yml`); a missing record stays RED |

## Regenerating and pushing the slim repo

Whenever dashboard code changes (a new page, a new import, a new data file):

```bash
# in a v9 checkout, with the dashboard's venv
python scripts/dashboard_manifest.py --write      # ~3 minutes; re-traces every page
python -m pytest tests/test_dashboard_manifest.py -q
git add deploy/dashboard_manifest.json && git commit -m "dashboard manifest"   # commit in v9

# build into a clone of the slim repo (everything except .git/ is replaced)
git clone https://github.com/NevixAA/wowza-dashboard ../wowza-dashboard     # once
python scripts/build_dashboard_repo.py ../wowza-dashboard
cd ../wowza-dashboard && git add -A && git commit -m "build from v9 <sha>" && git push
```

`git add -A` is safe in the slim repo only: it is generated, and its `.gitignore` excludes everything the
mirror downloads (`/output/`, `/config/...json`, `/models/...`, `deploy/.mirror_sync.json`, secrets).

A data-only change in v9 needs **no** rebuild: the mirror picks it up within 10 minutes. A new data file
that the manifest does not list is **not** mirrored until the manifest is regenerated and the slim repo
rebuilt - which is exactly what `tests/test_dashboard_manifest.py` exists to catch.

First creation of the slim repo (main session / owner): create an empty public repo
`NevixAA/wowza-dashboard` (no README, no licence), then from the built tree
`git init -b main && git add -A && git commit -m "Initial slim build" && git remote add origin https://github.com/NevixAA/wowza-dashboard.git && git push -u origin main`.
It must be **public** or the Streamlit app must be granted access to it; the data it reads is v9's public
raw URLs either way, so no token is ever needed.

## Streamlit Community Cloud - owner steps

Streamlit Community Cloud does not let an existing app switch to a different repository, so the slim app
is deployed as a new app that takes over the old URL:

1. Open the current app at share.streamlit.io -> the app's **⋮ -> Settings**.
   * **General**: note the app URL (subdomain) and the Python version.
   * **Secrets**: copy the whole secrets box somewhere private (a password manager, not a chat or a file in
     a repo).
2. **⋮ -> Delete** the old app (this frees its subdomain; the v9 repo is not touched).
3. **Create app -> Deploy a public app from GitHub**:
   * Repository `NevixAA/wowza-dashboard`, branch `main`, main file path `app.py`.
   * **App URL**: the same subdomain as before, so bookmarks keep working.
   * **Advanced settings**: Python **3.11** (what the slim build was proven on); paste the secrets from step 1.
4. Deploy. The first boot installs the packages, then shows "Fetching the latest Wowza data..." for a few
   seconds while the mirror downloads ~16 MB, then the Home page.

If the old app cannot be deleted first, deploy the new app on a new subdomain, check it, then delete the
old one and (in the new app's Settings -> General) change the subdomain back.

### Secrets the app reads (keys only - copy the values from the existing app)

| key | used by | needed? |
|---|---|---|
| `PUBLIC_MODE` | `utils/access.is_public` - read-only mode | optional (default `false`) |
| `[auth]` `redirect_uri`, `cookie_secret`, `client_id`, `client_secret`, `server_metadata_url` | `st.login` (Google OIDC) for the Free / Pro / Admin tiers | only when tiers are switched on (dormant without it). `redirect_uri` must be `https://<app>.streamlit.app/oauth2callback` of the app's actual URL - update it, and the Google OAuth client's authorised redirect URIs, if the subdomain changes |
| `[access]` `admins`, `pro` | `dashboard_access` - email lists per tier | only with `[auth]` |
| `GOOGLE_API_KEY` or `ANTHROPIC_API_KEY` (top-level; Streamlit exports top-level secrets as environment variables) | Agent Analysis page (`agent/agent_runner.py`) | optional; the page explains itself without one |

The app needs **no** secret to read data: every v9, Pro and v11 file is fetched from public raw URLs.
Never put a GitHub token in the secrets for this.

Optional environment knobs (top-level secrets also become env vars): `WOWZA_MIRROR_TTL_MIN` (default 10),
`WOWZA_MIRROR_COLD_BUDGET_S` (120), `WOWZA_MIRROR_WORKERS` (8), `WOWZA_MIRROR_BRANCH` (`main`).
