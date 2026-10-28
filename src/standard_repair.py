"""
Standard v9 forensic repair — research entry point (2026-10-28). READ-ONLY w.r.t. production.
=============================================================================================

This module is how the Pro reconciliation (agent R2) trains and scores the two recipes on
identical data, from a v9 checkout, as `V9_LOCAL`:

    LEGACY_V9_REPRO   recipe "legacy_v9": the exact production trainer and feature builder,
                      defects included (batch-median imputation, closing-price overround, meta
                      fitted on its own test block, misaligned weights, pooled side models).
    V9_REPAIRED       recipe "v9_repaired": persisted train-only imputer, one canonical
                      train/serve feature builder (src/features_repaired.py), league-scoped
                      team stats, four-block chronological stacking, weights attached to rows.

and the three market-feature arms of the closing-price experiment (all on the repaired recipe,
identical chronological blocks, only the price feature differs):

    STANDARD_NO_MARKET_FEATURES                       price_mode="none"
    STANDARD_TRUE_ASOF_MARKET_FEATURES                price_mode="asof"  (football-data NON-C
                                                      prices, class FD_PREMATCH_APPROX; needs
                                                      --fd-raw)
    CURRENT_STANDARD_CLOSING_CONTAMINATED_REPRODUCTION price_mode="closing" (AvgC>2.5 & co.)

NOTHING HERE WRITES A SERVING FILE OR CALLS src/promotion_gate.py. Every artifact goes to an
explicit `--out` directory; `save_research_payload` refuses the serving paths.

CLI (run from the v9 root):

    python -m src.standard_repair train  --arm V9_REPAIRED --out DIR [--fd-raw DIR] [--target over25]
    python -m src.standard_repair train  --arm LEGACY_V9_REPRO --out DIR
    python -m src.standard_repair sides  --recipe v9_repaired --out DIR     # pooled/std/nf x btts/over15/over35
    python -m src.standard_repair score  --payload P.pkl --frame F.parquet --out scores.csv
    python -m src.standard_repair board-delta --repaired-payload P.pkl --out DIR [--nf-payload P.pkl]

Every artifact carries a provenance block (generated_at, v9 commit, recipe and feature versions,
frame hash, price mode/source).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]

ARMS = {
    "LEGACY_V9_REPRO": {"recipe": "legacy_v9", "price_mode": "closing"},
    "V9_REPAIRED": {"recipe": "v9_repaired", "price_mode": "none"},
    "STANDARD_NO_MARKET_FEATURES": {"recipe": "v9_repaired", "price_mode": "none"},
    "STANDARD_TRUE_ASOF_MARKET_FEATURES": {"recipe": "v9_repaired", "price_mode": "asof"},
    "CURRENT_STANDARD_CLOSING_CONTAMINATED_REPRODUCTION": {"recipe": "v9_repaired",
                                                           "price_mode": "closing"},
}
SIDE_TARGETS = ("btts", "over15", "over35")
TRACKS = ("pooled", "standard", "newformat")

#: football-data league codes for the raw-file reader (same codes data_loader downloads).
FD_CODES = {"Championship": "E1", "League One": "E2", "League Two": "E3", "National League": "EC",
            "Bundesliga 2": "D2", "Ligue 2": "F2", "La Liga 2": "SP2", "Serie B": "I2",
            "Greek Super League": "G1", "Portuguese Primeira Liga": "P1",
            "Scottish Championship": "SC1", "Scottish Premiership": "SC0",
            "Belgian First Division A": "B1", "Dutch Eredivisie": "N1",
            "Turkish Super Lig": "T1", "Scottish League One": "SC2", "Scottish League Two": "SC3"}
#: ENTRY = football-data pre-match columns (no "C" infix). CLOSING = the "C" columns.
FD_ENTRY_OVER = ["Avg>2.5", "B365>2.5", "P>2.5", "Max>2.5"]
FD_ENTRY_UNDER = ["Avg<2.5", "B365<2.5", "P<2.5", "Max<2.5"]
FD_CLOSE_OVER = ["AvgC>2.5", "B365C>2.5", "PC>2.5", "MaxC>2.5"]
FD_CLOSE_UNDER = ["AvgC<2.5", "B365C<2.5", "PC<2.5", "MaxC<2.5"]
ENTRY_CLASS = "FD_PREMATCH_APPROX"
CLOSING_CLASS = "FD_CLOSING"


# ── provenance ──────────────────────────────────────────────────────────────────────────────

def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=20).stdout.strip() or "unknown"
    except Exception:                                                  # noqa: BLE001
        return "unknown"


def frame_hash(df: pd.DataFrame, cols=("date", "league", "home_team", "away_team",
                                       "home_goals", "away_goals")) -> str:
    c = [x for x in cols if x in df.columns]
    h = pd.util.hash_pandas_object(df[c].astype(str), index=False).to_numpy()
    return hashlib.sha256(h.tobytes()).hexdigest()[:16]


def provenance(**extra) -> dict:
    from src.features_repaired import FEATURE_RECIPE_VERSION
    from src.imputer import IMPUTER_VERSION
    from src.model import MODEL_RECIPE_VERSION
    p = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
         "v9_commit": _git_sha(), "feature_recipe_version": FEATURE_RECIPE_VERSION,
         "imputer_version": IMPUTER_VERSION, "model_recipe_version": MODEL_RECIPE_VERSION,
         "team_canonicalisation": "data_loader._canonicalise_clubs + team_names.resolve"}
    p.update(extra)
    return p


# ── prices: entry vs closing, never confused ────────────────────────────────────────────────

def _first(raw: pd.DataFrame, cols: list) -> tuple[pd.Series, pd.Series]:
    """First column (in preference order) that has a value, PER ROW; returns (value, column)."""
    val = pd.Series(np.nan, index=raw.index)
    src = pd.Series(None, index=raw.index, dtype=object)
    for c in cols:
        if c not in raw.columns:
            continue
        s = pd.to_numeric(raw[c], errors="coerce")
        take = val.isna() & s.notna() & (s > 1.0)
        val[take] = s[take]
        src[take] = c
    return val, src


def fd_entry_closing(raw: pd.DataFrame) -> pd.DataFrame:
    """Split one raw football-data file into ENTRY and CLOSING O/U 2.5 prices.

    Entry comes ONLY from non-C columns (class FD_PREMATCH_APPROX: football-data snapshots these
    on the Friday/Tuesday before kickoff). Closing comes ONLY from *C* columns. A file with only
    closing columns yields NaN entry prices — a closing price can never become an entry.
    """
    out = pd.DataFrame(index=raw.index)
    out["odds_over25_entry"], so = _first(raw, FD_ENTRY_OVER)
    out["odds_under25_entry"], su = _first(raw, FD_ENTRY_UNDER)
    out["odds_over25_close"], sc = _first(raw, FD_CLOSE_OVER)
    out["odds_under25_close"], sd = _first(raw, FD_CLOSE_UNDER)
    out["entry_price_source"] = so.fillna("")
    out["close_price_source"] = sc.fillna("")
    has_e = out["odds_over25_entry"].notna() & out["odds_under25_entry"].notna()
    has_c = out["odds_over25_close"].notna() & out["odds_under25_close"].notna()
    out["entry_price_class"] = np.where(has_e, ENTRY_CLASS, "MISSING")
    out["close_price_class"] = np.where(has_c, CLOSING_CLASS, "MISSING")
    out["entry_price_is_real"] = has_e
    out["close_price_is_real"] = has_c
    return out


def load_fd_entry_prices(fd_raw_dir) -> pd.DataFrame:
    """Read every `{CODE}_{YYZZ}.csv` in `fd_raw_dir` (football-data raw files, e.g. Pro's
    experiments/ou_info_inputs_2026_10/fd_raw) -> one row per match with entry AND closing."""
    inv = {v: k for k, v in FD_CODES.items()}
    frames = []
    for p in sorted(Path(fd_raw_dir).glob("*_*.csv")):
        code = p.stem.rsplit("_", 1)[0]
        lg = inv.get(code)
        if not lg:
            continue
        try:
            raw = pd.read_csv(p, encoding="utf-8-sig", on_bad_lines="skip", low_memory=False)
        except UnicodeDecodeError:
            # some football-data files are Windows-1252 (e.g. byte 0x92, a curly apostrophe)
            raw = pd.read_csv(p, encoding="cp1252", encoding_errors="replace",
                              on_bad_lines="skip", low_memory=False)
        if "Date" not in raw.columns:
            continue
        px = fd_entry_closing(raw)
        px["date"] = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce").dt.normalize()
        px["league"] = lg
        px["fd_home"] = raw.get("HomeTeam", pd.Series(np.nan, index=raw.index)).astype(str)
        px["fd_away"] = raw.get("AwayTeam", pd.Series(np.nan, index=raw.index)).astype(str)
        px["fd_file"] = p.name
        frames.append(px[px["date"].notna()])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def attach_entry_prices(frame: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Add odds_over25_entry / odds_under25_entry (+ class/source) to a training frame, matched
    on (date, league, normalised home, normalised away). Never touches odds_over25 (closing)."""
    from src.data_loader import _norm_club
    f = frame.copy()
    if prices is None or prices.empty:
        f["odds_over25_entry"] = np.nan
        f["odds_under25_entry"] = np.nan
        f["entry_price_class"] = "MISSING"
        return f
    pk = prices.assign(_d=pd.to_datetime(prices["date"]).dt.normalize(),
                       _h=prices["fd_home"].map(_norm_club), _a=prices["fd_away"].map(_norm_club))
    pk = pk.drop_duplicates(["_d", "league", "_h", "_a"], keep="last")
    f["_d"] = pd.to_datetime(f["date"]).dt.normalize()
    f["_h"] = f["home_team"].astype(str).map(_norm_club)
    f["_a"] = f["away_team"].astype(str).map(_norm_club)
    cols = ["odds_over25_entry", "odds_under25_entry", "entry_price_class", "entry_price_source",
            "odds_over25_close", "odds_under25_close", "close_price_class"]
    f = f.drop(columns=[c for c in cols if c in f.columns])
    f = f.merge(pk[["_d", "league", "_h", "_a"] + cols], on=["_d", "league", "_h", "_a"],
                how="left")
    f["entry_price_class"] = f["entry_price_class"].fillna("MISSING")
    return f.drop(columns=["_d", "_h", "_a"])


# ── frames ──────────────────────────────────────────────────────────────────────────────────

def load_history() -> pd.DataFrame:
    """Production loader, COVID exclusion exactly as pipeline.mode_train applies it."""
    from src.data_loader import load_all_matches
    raw = load_all_matches()
    if config.EXCLUDE_COVID_SEASONS:
        raw = raw[~raw["season"].isin(config.COVID_SEASONS)]
    return raw


def legacy_weights(df: pd.DataFrame) -> pd.Series:
    """pipeline.mode_train._get_weights, as an index-aligned Series (season-LABEL keyed — note
    the map has no 2025/26 or 2026/27 entry, so the two newest seasons get the 1.0 default while
    2024/25 gets 4.0; reported, not changed)."""
    w = df["season"].map(lambda s: config.TRAINING_DECAY_WEIGHTS.get(s, config.DEFAULT_DECAY_WEIGHT)
                         ).fillna(config.DEFAULT_DECAY_WEIGHT)
    return pd.Series(w.to_numpy(dtype=float), index=df.index)


#: Recency weights by AGE of the canonical season (0 = the newest season in the frame). Same
#: shape as config.TRAINING_DECAY_WEIGHTS had when it was typed (newest season 4.0, then 3/2/1.5),
#: but derived from the match date, so it can never freeze at a hand-typed season key.
DERIVED_AGE_WEIGHTS = {0: 4.0, 1: 3.0, 2: 2.0, 3: 1.5}


def derived_weights(df: pd.DataFrame, as_of=None) -> pd.Series:
    """Weight from the match DATE (bug: TRAINING_DECAY_WEIGHTS has no 2025/26 / 2026/27 key and
    standard rows also carry calendar-year labels, so the newest seasons fell back to 1.0 while
    2024/25 got 4.0). Season = canonical date-derived key; age is counted from `as_of`
    (default: the frame's newest date). COVID seasons keep weight 0 as in the legacy map."""
    from src.features_repaired import season_key
    d = pd.to_datetime(df["date"])
    sk = season_key(d, df["league"].astype(str))
    ref = pd.Timestamp(as_of) if as_of is not None else d.max()
    ref_sk = season_key(pd.Series([ref] * len(df), index=df.index), df["league"].astype(str))
    start = sk.str.slice(0, 4).astype(int)
    age = ref_sk.str.slice(0, 4).astype(int) - start
    w = age.map(lambda a: DERIVED_AGE_WEIGHTS.get(int(a), config.DEFAULT_DECAY_WEIGHT)).astype(float)
    covid = sk.isin({"2019/20", "2020/21"})
    w[covid] = 0.0
    return pd.Series(w.to_numpy(dtype=float), index=df.index)


def prepare_history(raw: pd.DataFrame, recipe: str, alias_map=None, identity: bool = True):
    """Legacy: the frame as production builds it. Repaired: club identities canonicalised and
    duplicate fixtures removed (src/identity_repair) BEFORE any feature is built.
    `alias_map` = dict or path to an identity/1.0 JSON (e.g. Pro's identity_aliases.json)."""
    if recipe == "legacy_v9" or not identity:
        return raw, {"identity": "none"}
    from src.identity_repair import canonicalise_and_dedupe, load_alias_map
    amap = load_alias_map(alias_map) if isinstance(alias_map, (str, Path)) else alias_map
    d, rep = canonicalise_and_dedupe(raw, amap)
    d.attrs["identity"] = f"{rep['identity_version']}:{'explicit+derived' if amap else 'derived'}"
    rep ={k: v for k, v in rep.items() if k != "quarantine"} | {
        "quarantine_rows": rep["quarantine"][["date", "league", "home_team", "away_team",
                                              "home_goals", "away_goals"]].astype(str)
        .to_dict("records")}
    return d, rep


def feature_frame(raw: pd.DataFrame, recipe: str, price_mode: str = "closing",
                  fd_prices: pd.DataFrame | None = None, team_scope: str = "league") -> pd.DataFrame:
    """All-league feature frame. Legacy is built over ALL leagues exactly as mode_train does
    (its rolling stats cross leagues); repaired is league-scoped (or explicit chronological
    club scope) so the scope cannot leak."""
    from src.feature_engineering import build_features
    from src.features_repaired import build_features_repaired
    if recipe == "legacy_v9":
        return build_features(raw)
    src = attach_entry_prices(raw, fd_prices) if price_mode == "asof" else raw
    out = build_features_repaired(src, price_mode=price_mode, team_scope=team_scope)
    out.attrs["team_scope"] = team_scope
    out.attrs["identity"] = raw.attrs.get("identity", "none")
    return out


def track_rows(feat: pd.DataFrame, track: str, target: str = "over25") -> pd.DataFrame:
    """mode_train's `valid` filter, then the track. Raises if a track would mix formats."""
    valid = feat.dropna(subset=["over25", "home_scored_last5"])
    if target != "over25":
        valid = valid.dropna(subset=[target])
    if track == "standard":
        out = valid[valid["league"].isin(config.STANDARD_FORMAT_LEAGUES)]
    elif track == "newformat":
        out = valid[valid["league"].isin(config.NEW_FORMAT_LEAGUES)]
    elif track == "pooled":
        out = valid[valid["league"].isin(set(config.STANDARD_FORMAT_LEAGUES)
                                         | set(config.NEW_FORMAT_LEAGUES))]
    else:
        raise ValueError(track)
    assert_track_pure(out, track)
    return out


def assert_track_pure(df: pd.DataFrame, track: str) -> None:
    """Test 13: Standard and New-Format rows can never silently pool in a track-specific model."""
    lg = set(df["league"].astype(str))
    if track == "standard" and lg - set(config.STANDARD_FORMAT_LEAGUES):
        raise ValueError(f"standard track contains non-standard leagues: "
                         f"{sorted(lg - set(config.STANDARD_FORMAT_LEAGUES))}")
    if track == "newformat" and lg - set(config.NEW_FORMAT_LEAGUES):
        raise ValueError(f"newformat track contains non-NF leagues: "
                         f"{sorted(lg - set(config.NEW_FORMAT_LEAGUES))}")


def feature_cols(recipe: str, price_mode: str, track: str = "standard") -> list:
    """Model inputs. Legacy = the serving payload lists (standard 42; NF = mode_train's NF drop)."""
    from src.features_repaired import STANDARD_SERVING_FEATURES, feature_cols_for
    base = list(STANDARD_SERVING_FEATURES)
    if track == "newformat":
        nf_drop = {"home_xg_last5", "away_xg_last5", "home_insidebox_last5", "away_insidebox_last5",
                   "home_ht_scored_last5", "away_ht_scored_last5", "home_ht_conceded_last5",
                   "away_ht_conceded_last5", "home_ht_attack_str", "away_ht_attack_str",
                   "combined_ht_goals_avg", "home_ht_over05_rate", "away_ht_over05_rate",
                   "home_ht_over15_rate", "away_ht_over15_rate"}
        base = [c for c in base if c not in nf_drop]
    if recipe == "legacy_v9":
        return base
    return feature_cols_for(price_mode, base)


# ── train / score ───────────────────────────────────────────────────────────────────────────

def train_track(feat: pd.DataFrame, recipe: str, price_mode: str, track: str = "standard",
                target: str = "over25", weights: str | None = None) -> dict:
    """Train one payload. Legacy: production's `train()` with mode_train's weights, unchanged.
    Repaired: `train_repaired` with weights attached to their rows; `weights` = "derived"
    (repaired default: from the match date) or "legacy" (the season-label map)."""
    from src.model import train, build_payload
    rows = track_rows(feat, track, target)
    cols = feature_cols(recipe, price_mode, "newformat" if track == "newformat" else "standard")
    weights = weights or ("legacy" if recipe == "legacy_v9" else "derived")
    w = legacy_weights(rows) if weights == "legacy" else derived_weights(rows)
    if recipe == "legacy_v9":
        res = train(rows, target=target, sample_weight=w.to_numpy(), feature_cols=cols)
    else:
        res = train(rows, target=target, sample_weight=w, feature_cols=cols, recipe="v9_repaired")
    payload = build_payload(res, target)
    payload["research"] = provenance(recipe=recipe, price_mode=price_mode, track=track,
                                     target=target, rows=int(len(rows)), weights=weights,
                                     team_scope=str(feat.attrs.get("team_scope", "league")),
                                     identity=str(feat.attrs.get("identity", "none")),
                                     frame_hash=frame_hash(rows),
                                     price_source=("FD_PREMATCH_APPROX" if price_mode == "asof"
                                                   else "FD_CLOSING_PREFERRED" if price_mode == "closing"
                                                   else "NONE"))
    return payload


def save_research_payload(payload: dict, path) -> Path:
    """Write a research payload. Refuses every serving model path (and the models/ dir itself)."""
    path = Path(path).resolve()
    serving = {Path(p).resolve() for p in (
        config.MODEL_FILE_STANDARD, config.MODEL_FILE_NEWFORMAT, config.MODEL_FILE_BTTS,
        config.MODEL_FILE_OVER15, config.MODEL_FILE_OVER35, config.HT_MODEL_FILE_05,
        config.HT_MODEL_FILE_15, config.MODEL_FILE_STANDARD_REPAIRED)}
    if path in serving or path.parent == Path(config.MODELS_DIR).resolve():
        raise PermissionError(f"{path} is a serving location; research payloads go elsewhere")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    meta = {k: v for k, v in payload.items() if k not in ("models", "imputer")}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return path


def score(payload: dict, frame: pd.DataFrame) -> pd.Series:
    from src.model import predict_proba
    return predict_proba(frame, payload=payload)


def train_side_tracks(feat: pd.DataFrame, recipe: str, out_dir, price_mode: str = "none",
                      targets=SIDE_TARGETS, weights: str | None = None) -> pd.DataFrame:
    """Bug 9: pooled vs Standard-only vs NF-only side-market CHALLENGERS, written to `out_dir`
    only. Each is scored on its own track's rows of the pooled model's holdout window too, so the
    three can be compared on identical rows. Returns a summary table."""
    from src.model import predict_proba, four_block_split
    out_dir = Path(out_dir)
    rows = []
    for target in targets:
        payloads = {}
        for track in TRACKS:
            try:
                rr = track_rows(feat, track, target)
                if len(rr) < config.BACKTEST_MIN_TRAIN:
                    log.warning(f"[{target}/{track}] only {len(rr)} rows — skipped")
                    continue
                p = train_track(feat, recipe, price_mode, track, target, weights)
                save_research_payload(p, out_dir / f"{recipe}_{track}_{target}.pkl")
                payloads[track] = p
            except Exception as e:                                     # noqa: BLE001
                log.warning(f"[{target}/{track}] failed: {e}")
        if "pooled" not in payloads:
            continue
        # identical evaluation rows: the pooled model's holdout, split by track
        pooled_rows = track_rows(feat, "pooled", target)
        hold = four_block_split(pooled_rows, target)["holdout"]
        for track_eval in ("standard", "newformat"):
            ev = hold[hold["league"].isin(config.STANDARD_FORMAT_LEAGUES if track_eval == "standard"
                                          else config.NEW_FORMAT_LEAGUES)]
            if ev.empty:
                continue
            y = ev[target].to_numpy(float)
            for model_track in ("pooled", track_eval):
                if model_track not in payloads:
                    continue
                p = np.clip(predict_proba(ev, payload=payloads[model_track]).to_numpy(float),
                            1e-15, 1 - 1e-15)
                rows.append({"target": target, "eval_track": track_eval, "model": model_track,
                             "recipe": recipe, "n": int(len(ev)),
                             "eval_dates": f"{ev['date'].min()}..{ev['date'].max()}",
                             "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
                             "brier": float(np.mean((p - y) ** 2)),
                             "note": ("pooled holdout rows; a track model may have trained on "
                                      "rows inside this window only if its own chronology differs"
                                      " — compare on prospective rows for promotion")})
    tab = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out_dir / f"side_track_comparison_{recipe}.csv", index=False)
    return tab


# ── board delta (measure only) ──────────────────────────────────────────────────────────────

def board_delta(predictions_csv, historical: pd.DataFrame, repaired_std: dict | None,
                repaired_nf: dict | None = None, price_mode: str = "none") -> pd.DataFrame:
    """Per-fixture probability effect on the committed board. Never writes production files.

      p_served            the committed predictions.csv p_over25
      p_legacy_rescored   serving payload on the board's own features, Standard batch as predict
                          does it (sanity: must equal p_served)
      p_legacy_alone      the same fixture scored ALONE (bug 1 effect = alone - rescored)
      p_repaired          repaired payload on repaired upcoming features
    """
    from src.model import load_models, predict_proba
    from src.feature_engineering import build_upcoming_features
    board = pd.read_csv(predictions_csv)
    std_mask = board["league"].isin(config.STANDARD_FORMAT_LEAGUES)
    nf_mask = board["league"].isin(config.NEW_FORMAT_LEAGUES)
    pay_std = load_models(model_file=config.MODEL_FILE_STANDARD)
    pay_nf = load_models(model_file=config.MODEL_FILE_NEWFORMAT)
    out = board[["date", "league", "home_team", "away_team", "odds_over25", "odds_under25",
                 "p_over25"]].rename(columns={"p_over25": "p_served"}).copy()
    out["track"] = np.where(std_mask, "standard", np.where(nf_mask, "newformat", "other"))
    out["p_legacy_rescored"] = np.nan
    out["p_legacy_alone"] = np.nan
    for mask, pay in ((std_mask, pay_std), (nf_mask, pay_nf)):
        if not mask.any():
            continue
        sub = board[mask]
        out.loc[mask, "p_legacy_rescored"] = predict_proba(sub, payload=pay).to_numpy()
        out.loc[mask, "p_legacy_alone"] = [float(predict_proba(sub.iloc[[i]], payload=pay).iloc[0])
                                           for i in range(len(sub))]
    # side markets: predict scores std+nf in ONE batch
    side_mask = std_mask | nf_mask
    for t, f in config.SIDE_MARKETS.items():
        if f.exists() and side_mask.any() and f"p_{t}" in board.columns:
            sp = load_models(model_file=f)
            sub = board[side_mask]
            out[f"p_{t}_served"] = board[f"p_{t}"]
            out.loc[side_mask, f"p_{t}_std_only_batch"] = np.nan
            if std_mask.any():
                out.loc[std_mask, f"p_{t}_std_only_batch"] = predict_proba(
                    board[std_mask], payload=sp).to_numpy()
            out.loc[side_mask, f"p_{t}_alone"] = [
                float(predict_proba(sub.iloc[[i]], payload=sp).iloc[0]) for i in range(len(sub))]
    out["bug1_delta"] = out["p_legacy_alone"] - out["p_legacy_rescored"]
    out["p_repaired"] = np.nan
    ident = board[["date", "league", "home_team", "away_team", "odds_over25", "odds_under25"]]
    for mask, pay in ((std_mask, repaired_std), (nf_mask, repaired_nf)):
        if pay is None or not mask.any():
            continue
        from src.features_repaired import build_upcoming_features_repaired
        rs = pay.get("research") or {}
        feats = build_upcoming_features_repaired(ident[mask], historical,
                                                 price_mode=rs.get("price_mode", price_mode),
                                                 team_scope=rs.get("team_scope", "league"))
        out.loc[mask, "p_repaired"] = predict_proba(feats, payload=pay).to_numpy()
    out["repaired_minus_served"] = out["p_repaired"] - out["p_served"]
    return out


def _summ(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {"n": 0}
    a = s.abs()
    return {"n": int(len(s)), "mean": round(float(s.mean()), 5),
            "mean_abs": round(float(a.mean()), 5), "median_abs": round(float(a.median()), 5),
            "p95_abs": round(float(a.quantile(0.95)), 5), "max_abs": round(float(a.max()), 5),
            "share_gt_1pp": round(float((a > 0.01).mean()), 4),
            "share_gt_5pp": round(float((a > 0.05).mean()), 4)}


# ── CLI ─────────────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.standard_repair")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--arm", choices=sorted(ARMS), required=True)
    t.add_argument("--target", default="over25")
    t.add_argument("--track", default="standard", choices=TRACKS)
    t.add_argument("--out", required=True)
    t.add_argument("--fd-raw", default=None, help="dir of raw football-data CSVs (asof arm)")
    t.add_argument("--frame-out", default=None, help="also write the feature frame (parquet)")
    s = sub.add_parser("sides")
    s.add_argument("--recipe", choices=config.MODEL_RECIPES, default="v9_repaired")
    s.add_argument("--price-mode", choices=config.MARKET_FEATURE_MODES, default="none")
    s.add_argument("--out", required=True)
    sc = sub.add_parser("score")
    sc.add_argument("--payload", required=True)
    sc.add_argument("--frame", required=True)
    sc.add_argument("--out", required=True)
    b = sub.add_parser("board-delta")
    b.add_argument("--predictions", default=str(config.OUTPUT_DIR / "predictions.csv"))
    b.add_argument("--repaired-payload", default=None)
    b.add_argument("--nf-payload", default=None)
    b.add_argument("--out", required=True)
    for p_ in (t, s, b):
        p_.add_argument("--alias-map", default=None,
                        help="identity/1.0 alias JSON (e.g. Pro output/standard_repair/"
                             "identity_aliases.json); repaired recipe only")
        p_.add_argument("--no-identity", action="store_true",
                        help="repaired recipe WITHOUT identity canonicalisation (ablation)")
        p_.add_argument("--team-scope", choices=("league", "club"), default="league")
    for p_ in (t, s):
        p_.add_argument("--weights", choices=("derived", "legacy"), default=None,
                        help="repaired default: derived from match date")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if a.cmd == "train":
        arm = ARMS[a.arm]
        raw, idrep = prepare_history(load_history(), arm["recipe"], a.alias_map, not a.no_identity)
        fd = load_fd_entry_prices(a.fd_raw) if (arm["price_mode"] == "asof" and a.fd_raw) else None
        if arm["price_mode"] == "asof" and fd is None:
            log.warning("asof arm without --fd-raw: entry prices are NaN, bookmaker_overround is a "
                        "constant (= the NO_MARKET arm in effect)")
        feat = feature_frame(raw, arm["recipe"], arm["price_mode"], fd, a.team_scope)
        if a.frame_out:
            feat.to_parquet(a.frame_out, index=False)
        p = train_track(feat, arm["recipe"], arm["price_mode"], a.track, a.target, a.weights)
        p["research"]["arm"] = a.arm
        p["research"]["identity_report"] = {k: v for k, v in idrep.items() if k != "aliases"}
        if fd is not None:
            rr = track_rows(feat, a.track, a.target)
            p["research"]["entry_price_coverage"] = round(
                float(rr.get("odds_over25_entry", pd.Series(dtype=float)).notna().mean()), 4)
        path = save_research_payload(p, Path(a.out) / f"{a.arm}_{a.track}_{a.target}.pkl")
        print(json.dumps({"payload": str(path), **{k: v for k, v in p.items()
                                                    if k in ("served_holdout_metrics", "blocks",
                                                             "research")}}, indent=2, default=str))
    elif a.cmd == "sides":
        raw, _ = prepare_history(load_history(), a.recipe, a.alias_map, not a.no_identity)
        feat = feature_frame(raw, a.recipe, a.price_mode, None, a.team_scope)
        tab = train_side_tracks(feat, a.recipe, a.out, a.price_mode, weights=a.weights)
        print(tab.to_string(index=False))
    elif a.cmd == "score":
        with open(a.payload, "rb") as f:
            pay = pickle.load(f)
        fr = pd.read_parquet(a.frame) if a.frame.endswith(".parquet") else pd.read_csv(a.frame)
        fr = fr.copy()
        fr["p"] = score(pay, fr).to_numpy()
        fr.to_csv(a.out, index=False)
    elif a.cmd == "board-delta":
        def _ld(p):
            if not p:
                return None
            with open(p, "rb") as f:
                return pickle.load(f)
        raw, _ = prepare_history(load_history(), "v9_repaired", a.alias_map, not a.no_identity)
        tab = board_delta(a.predictions, raw, _ld(a.repaired_payload), _ld(a.nf_payload))
        Path(a.out).mkdir(parents=True, exist_ok=True)
        tab.to_csv(Path(a.out) / "board_delta.csv", index=False)
        summ = {"provenance": provenance(predictions=a.predictions)}
        for tr in ("standard", "newformat"):
            m = tab["track"] == tr
            summ[tr] = {"bug1_alone_vs_batch": _summ(tab.loc[m, "bug1_delta"]),
                        "rescore_reproduces_served": _summ(tab.loc[m, "p_legacy_rescored"]
                                                           - tab.loc[m, "p_served"]),
                        "repaired_minus_served": _summ(tab.loc[m, "repaired_minus_served"])}
            for t_ in config.SIDE_MARKETS:
                if f"p_{t_}_alone" in tab.columns:
                    summ[tr][f"{t_}_alone_vs_served_batch"] = _summ(
                        tab.loc[m, f"p_{t_}_alone"] - tab.loc[m, f"p_{t_}_served"])
                    summ[tr][f"{t_}_std_only_batch_vs_served"] = _summ(
                        tab.loc[m, f"p_{t_}_std_only_batch"] - tab.loc[m, f"p_{t_}_served"])
        (Path(a.out) / "board_delta_summary.json").write_text(json.dumps(summ, indent=2),
                                                               encoding="utf-8")
        print(json.dumps(summ, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
