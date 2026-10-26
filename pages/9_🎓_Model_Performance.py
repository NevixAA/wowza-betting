"""Model performance — is the model any good, separately from whether the bets made money?

WHY THIS PAGE IS SEPARATE FROM P/L. "The model is accurate" and "the bets are profitable" are two
different claims and collapsing them is the single most expensive confusion in this estate. A
model can be accurate and unprofitable (the price already knew), or profitable and badly
calibrated (variance). This page only asks the first question.

IT LEADS WITH THE BAD NEWS. The strongest result Wowza has about its own model is a negative one,
and it is shown first, at full size, with its n and its z. A dashboard that buries that is not
measuring — it is marketing.

Visual redesign 2026-10-25 (stage 3): one card per finding (one main number, its n and the evidence kind
beside it), section headers instead of dividers, and the v11 / Pro reads cached for five minutes (they were
re-fetched on every widget interaction). PRESENTATION ONLY — every value from the same adapter call.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard_access import require as _require_tier  # noqa: E402
_require_tier("admin")  # tier guard (dashboard_access.PAGES), 2026-10-18

import dashboard_ui as ui  # noqa: E402
from src.tier_labels import TIER_LABEL  # noqa: E402
from dashboard_data import fantasy, pro, v9, v11  # noqa: E402

st.set_page_config(page_title="Wowza | Model performance", page_icon="🎓", layout="wide")


@st.cache_data(ttl=300, show_spinner="Reading v11 and Pro…")
def _remote() -> dict:
    """The two remote reads (v11 residual, Pro walk-forward), once per five minutes."""
    r = v11.residual() if v11.available() else {"available": False, "why": v11.absent_reason()}
    if pro.available():
        sl = pro.shadow_learning()
        pro_absent = None
    else:
        sl, pro_absent = {}, pro.absent_reason()
    return {"residual": r, "shadow_learning": sl, "pro_absent": pro_absent}


cal = v9.calibration()
R = _remote()
ui.page_header("Model performance", "Accuracy, not profit. The two are measured separately on purpose.",
               eyebrow_text="Research Lab · model accuracy",
               freshness=ui.status_chip("RESEARCH", label="RESEARCH · NOT A P/L PAGE"))

# ═══ 1 · calibration: the load-bearing negative result ═══════════════════════════════════════════
ui.section_header("1 · Does the model mean what it says?")
if not cal.get("available"):
    ui.empty_state("NO_DATA", cal.get("why", "not computable"))
else:
    st.markdown(
        "**A worked example.** Take every bet where the model said *this happens about 55% of the "
        "time*. If the model is honest, about 55 of every 100 of those should win. That is the "
        "whole test — and it needs no model file, because the ledger stores both the edge and the "
        "price, and Wowza's edge is defined as `model probability − 1/odds`."
    )
    _ev = ui.evidence_badge("LIVE LEDGER", cal["n"], state="SETTLED")
    k = st.columns(3)
    ui.metric_card("Overconfident by", f"+{cal['gap_pp']:.2f}pp", title="Calibration gap",
                   supporting=[("z", f"{cal['z']}"), ("Settled sample", f"{cal['n']:,}")],
                   status=ui.status_chip("FAIL", label="OVERCONFIDENT"), evidence=_ev, n=cal["n"], container=k[0])
    ui.metric_card("Model claimed", f"{cal['claimed']:.4f}", title="Average stated probability",
                   supporting=[("WIN or LOSS rows", f"{cal['n']:,}")], container=k[1])
    ui.metric_card("Actually happened", f"{cal['realised']:.4f}", title="Realised rate",
                   supporting=[("on settled bets", f"{cal['n']:,}")], container=k[2])

    st.markdown(
        f"**In plain words.** Across {cal['n']:,} settled bets the model said things would happen "
        f"{cal['claimed'] * 100:.1f}% of the time. They happened {cal['realised'] * 100:.1f}% of "
        f"the time. On a hundred bets it expects to win about {cal['claimed'] * 100:.0f} and wins "
        f"about {cal['realised'] * 100:.0f}. A z of {cal['z']} means that gap is far too large to "
        f"be luck."
    )

    b = cal["bins"].copy()
    b["label"] = b["bin"] + b["reliable"].map({True: "", False: "  ⚠ thin"})
    c1, c2 = st.columns([3, 2])
    with c1:
        st.altair_chart(ui.calibration(b, "claimed", "realised", "n",
                                       title="Claimed vs realised — the dashed line is honesty"),
                        width="stretch")
        st.caption("Every point below the diagonal is the model claiming more than it delivered. "
                   "Point size is n, so a confident-looking dot on 2 bets cannot masquerade as "
                   "evidence.")
    with c2:
        ui.eyebrow("Gap by confidence band")
        ui.table(b[["label", "n", "claimed", "realised", "gap_pp"]]
                 .rename(columns={"label": "claimed band", "gap_pp": "overconfidence (pp)"})
                 .round(4))
        thin = int((~b["reliable"]).sum())
        if thin:
            st.markdown(ui.status_chip("SMALL_SAMPLE", label=f"{thin} THIN BAND(S) · n < 20"))
            st.caption(f"⚠ {thin} band(s) hold fewer than 20 bets and are marked. They are kept "
                       "visible because dropping them would make the curve look tidier than the "
                       "evidence is.")

    # ── the finding that actually bites ───────────────────────────────────────────────────────
    ui.section_header("Does the tier ladder carry information?")
    bt = cal.get("by_tier") or {}
    if bt:
        d = pd.DataFrame([{"tier": k, **v} for k, v in bt.items()]).sort_values("gap_pp")
        d["tier"] = d["tier"].map(lambda t: TIER_LABEL.get(str(t).upper(), t))   # display words only
        st.markdown(
            f"**The claim being tested.** The tier is the size of the gap between our probability and the "
            f"price ({TIER_LABEL['SNIPER']} largest, {TIER_LABEL['VALUABLE']} smallest). If a bigger gap "
            "meant a better bet, the overconfidence should be smallest at the top."
        )
        cc1, cc2 = st.columns([3, 2])
        with cc1:
            long = d.melt(id_vars=["tier", "n"], value_vars=["claimed", "realised"],
                          var_name="series", value_name="value")
            st.altair_chart(
                ui.compare_bars(long, "tier", "value", "series",
                                order=["claimed", "realised"],
                                title="Claimed vs realised win rate, by tier",
                                y_title="probability"),
                width="stretch")
        with cc2:
            worst = d.iloc[-1]
            st.markdown(
                f"**It does not — it is inverted.** The worst-calibrated tier is "
                f"**{worst['tier']}**, overconfident by **{worst['gap_pp']:.2f}pp** on "
                f"n={int(worst['n']):,}, against {d.iloc[0]['gap_pp']:.2f}pp for "
                f"{d.iloc[0]['tier']}. The tier that gets the full stake is the one whose "
                f"confidence is least deserved."
            )
            st.markdown(
                "**Why this is the finding, not a footnote.** If the tiers were sorting real "
                "signal from weak signal, the gaps would shrink as you climb. They grow. A tier "
                "that does not separate outcomes is a label, not a signal — and it is currently "
                "driving stake size."
            )
            ui.table(d.rename(columns={"gap_pp": "overconfidence (pp)"}))

# ═══ 2 · the market comparison ══════════════════════════════════════════════════════════════════
ui.section_header("2 · Does the model beat the price?")
st.markdown(
    "Calibration says whether the model is honest with itself. This asks a harder question: "
    "once you already know what the bookmakers think, does the model add anything? The market "
    "price is a forecast too — made by people with money on it."
)
r = R["residual"]
if not r.get("available"):
    ui.empty_state("NOT_CONNECTED", r.get("why", "v11 not reachable"))
else:
    st.markdown(ui.status_chip("SHADOW", label=f"{v11.STATE} · MEASURED BY V11, WHICH LOGS AND NEVER STAKES"))
    d = pd.DataFrame(r["table"])
    base_col = next((c for c in d.columns if c.startswith("brier") and "wowza" not in c), None)
    with_col = next((c for c in d.columns if c.startswith("brier") and "wowza" in c), None)
    if base_col and with_col and "n" in d.columns:
        n = pd.to_numeric(d["n"], errors="coerce").fillna(0)
        w = n / n.sum() if n.sum() else n
        bm = float((pd.to_numeric(d[base_col], errors="coerce") * w).sum())
        bmw = float((pd.to_numeric(d[with_col], errors="coerce") * w).sum())
        c = st.columns(3)
        ui.metric_card("What Wowza adds", f"{bm - bmw:+.5f}", title="Brier improvement over the market",
                       supporting=[("n", f"{int(n.sum()):,}")],
                       status=ui.status_chip("NO_DATA", label="INDISTINGUISHABLE FROM ZERO", tone="NEUTRAL")
                       if abs(bm - bmw) < 0.001 else "",
                       evidence=ui.evidence_badge("SHADOW LOG", int(n.sum())), container=c[0])
        ui.metric_card("Market alone", f"{bm:.4f}", title="Brier — 0 is perfect, 0.25 is a coin flip",
                       container=c[1])
        ui.metric_card("Market + Wowza", f"{bmw:.4f}", title="Same fixtures, same scale", container=c[2])
        st.markdown(
            f"**Read it plainly.** Adding Wowza to the market's own de-vigged price changes the "
            f"score by {bm - bmw:+.5f} on {int(n.sum()):,} fixtures. That is indistinguishable "
            f"from zero. On this evidence the model knows nothing the price has not already "
            f"priced — which is exactly why v11 defaults to NO_BET rather than to a tier."
        )
    with st.expander("Residual table — every segment"):
        ui.table(d)

# ═══ 3 · does retraining help ═══════════════════════════════════════════════════════════════════
ui.section_header("3 · Does retraining help?")
if R["pro_absent"]:
    ui.empty_state("NOT_CONNECTED", R["pro_absent"])
else:
    sl = R["shadow_learning"]
    if not sl.get("available"):
        ui.empty_state("NO_DATA", sl.get("why", "no walk-forward results"))
    else:
        v = sl["variants"]
        st.markdown(
            "This is the one clearly positive result on the page, and it is positive in the way "
            "real effects are: small, and consistent across independent months."
        )
        best = sl.get("best_by_logloss")
        items = sorted(v.items())
        for i in range(0, len(items), 3):
            for c, (name, m) in zip(st.columns(3), items[i:i + 3]):
                ui.metric_card("Log loss", f"{m['log_loss']:.5f}", title=name.title(),
                               supporting=[("AUC", f"{m.get('auc', float('nan')):.4f}")],
                               status=ui.status_chip("PASS", label="BEST BY LOG LOSS") if name == best else "",
                               evidence=ui.evidence_badge("WALK-FORWARD", None, span=f"{sl['months']} months"),
                               container=c)
        if "frozen" in v and "canonical" in v:
            st.markdown(
                f"**Retraining wins.** A model retrained on the canonical store scores "
                f"{v['canonical']['log_loss']:.5f} against {v['frozen']['log_loss']:.5f} for one "
                f"left frozen, across {sl['months']} walk-forward months. The consequence is a "
                f"rule: **never delete old data.** Every window, decay and filter experiment that "
                f"threw history away lost to simply keeping it."
            )

# ═══ 4 · fantasy, on its own scale ══════════════════════════════════════════════════════════════
ui.section_header("4 · Fantasy — against a public benchmark")
b = fantasy.benchmark()
if not b.get("available"):
    ui.empty_state("NO_DATA", b.get("why", "no benchmark yet"))
else:
    st.markdown(ui.status_chip("PAPER", label="PAPER · FANTASY IS SCORED IN FPL POINTS, NEVER ADDED TO BETTING UNITS"))
    c = st.columns(3)
    ui.metric_card("Gameweeks won", f"{b['gameweeks_wowza_better']} of {b['gameweeks']}", title="Against FPL's ep_next",
                   supporting=[("Verdict", b["verdict"])],
                   status=ui.status_chip("FAIL", label="BEHIND THE BENCHMARK") if b["gameweeks_wowza_better"] == 0 else "",
                   evidence=ui.evidence_badge("GAMEWEEK BENCHMARK", None, span=f"{b['gameweeks']} GW"),
                   container=c[0])
    ui.metric_card("Wowza error", ui.points(b["wowza_mae"]), title="Mean absolute error", container=c[1])
    ui.metric_card("FPL's own ep_next", ui.points(b["fpl_mae"]), title="The public benchmark", container=c[2])
    st.markdown(
        f"**{b['verdict']}.** FPL publishes its own expected-points number, and it is more "
        f"accurate than ours: {b['fpl_mae']:.3f} points of error against our "
        f"{b['wowza_mae']:.3f}, on the same players in the same gameweeks. Our projection also "
        f"runs {b['wowza_bias']:+.3f} points high on average. This stays on the page because a "
        f"benchmark you only show when you win is not a benchmark."
    )
    st.caption("What Wowza adds that ep_next does not: an explicit start probability, a "
               "fixture-by-fixture split, and a stated bias. Those are useful even when the "
               "headline number loses.")
    d = pd.DataFrame(b["table"])
    long = d.melt(id_vars=["gw", "n"], value_vars=["wowza_mae", "fpl_mae"],
                  var_name="series", value_name="value")
    st.altair_chart(ui.compare_bars(long, "gw", "value", "series",
                                    order=["wowza_mae", "fpl_mae"],
                                    title="Mean absolute error by gameweek — lower is better",
                                    y_title="points of error"),
                    width="stretch")
    with st.expander("Benchmark table — every gameweek"):
        ui.table(d)
