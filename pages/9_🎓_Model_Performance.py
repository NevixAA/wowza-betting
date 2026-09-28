"""Model performance — is the model any good, separately from whether the bets made money?

WHY THIS PAGE IS SEPARATE FROM P/L. "The model is accurate" and "the bets are profitable" are two
different claims and collapsing them is the single most expensive confusion in this estate. A
model can be accurate and unprofitable (the price already knew), or profitable and badly
calibrated (variance). This page only asks the first question.

IT LEADS WITH THE BAD NEWS. The strongest result Wowza has about its own model is a negative one,
and it is shown first, at full size, with its n and its z. A dashboard that buries that is not
measuring — it is marketing.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard_data import fantasy, pro, ui, v9, v11

st.set_page_config(page_title="Wowza | Model performance", page_icon="🎓", layout="wide")
st.title("🎓 Model performance")
st.caption("Accuracy, not profit. The two are measured separately on purpose.")

# ── calibration: the load-bearing negative result ─────────────────────────────────────────────
st.header("1 · Does the model mean what it says?")
cal = v9.calibration()
if not cal.get("available"):
    st.info(cal.get("why", "not computable"))
else:
    st.markdown(
        "**A worked example.** Take every bet where the model said *this happens about 55% of the "
        "time*. If the model is honest, about 55 of every 100 of those should win. That is the "
        "whole test — and it needs no model file, because the ledger stores both the edge and the "
        "price, and Wowza's edge is defined as `model probability − 1/odds`."
    )
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        ui.kpi("Model claimed", f"{cal['claimed']:.4f}", "average stated probability")
    with k2:
        ui.kpi("Actually happened", f"{cal['realised']:.4f}", f"on {cal['n']:,} settled bets")
    with k3:
        ui.kpi("Overconfident by", f"+{cal['gap_pp']:.2f}pp", f"z = {cal['z']}",
               tone=ui.polarity()[1])
    with k4:
        ui.kpi("Settled sample", f"{cal['n']:,}", "WIN or LOSS rows")

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
                        use_container_width=False)
        st.caption("Every point below the diagonal is the model claiming more than it delivered. "
                   "Point size is n, so a confident-looking dot on 2 bets cannot masquerade as "
                   "evidence.")
    with c2:
        st.markdown("**Gap by confidence band**")
        ui.table(b[["label", "n", "claimed", "realised", "gap_pp"]]
                 .rename(columns={"label": "claimed band", "gap_pp": "overconfidence (pp)"})
                 .round(4))
        thin = int((~b["reliable"]).sum())
        if thin:
            st.caption(f"⚠ {thin} band(s) hold fewer than 20 bets and are marked. They are kept "
                       "visible because dropping them would make the curve look tidier than the "
                       "evidence is.")

    # ── the finding that actually bites ───────────────────────────────────────────────────────
    st.subheader("Does the tier ladder carry information?")
    bt = cal.get("by_tier") or {}
    if bt:
        d = pd.DataFrame([{"tier": k, **v} for k, v in bt.items()]).sort_values("gap_pp")
        st.markdown(
            "**The claim being tested.** SNIPER is supposed to be the strongest signal, VALUABLE "
            "the weakest. If the ladder means anything, the overconfidence gap should be smallest "
            "at the top."
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
                use_container_width=False)
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

st.divider()

# ── the market comparison ─────────────────────────────────────────────────────────────────────
st.header("2 · Does the model beat the price?")
st.markdown(
    "Calibration says whether the model is honest with itself. This asks a harder question: "
    "once you already know what the bookmakers think, does the model add anything? The market "
    "price is a forecast too — made by people with money on it."
)
r = v11.residual() if v11.available() else {"available": False, "why": v11.absent_reason()}
if not r.get("available"):
    ui.absent(r.get("why", "v11 not reachable"))
else:
    ui.state_banner(v11.STATE, "Measured by v11, which logs and never stakes.")
    d = pd.DataFrame(r["table"])
    base_col = next((c for c in d.columns if c.startswith("brier") and "wowza" not in c), None)
    with_col = next((c for c in d.columns if c.startswith("brier") and "wowza" in c), None)
    if base_col and with_col and "n" in d.columns:
        n = pd.to_numeric(d["n"], errors="coerce").fillna(0)
        w = n / n.sum() if n.sum() else n
        bm = float((pd.to_numeric(d[base_col], errors="coerce") * w).sum())
        bmw = float((pd.to_numeric(d[with_col], errors="coerce") * w).sum())
        c1, c2, c3 = st.columns(3)
        with c1:
            ui.kpi("Market alone", f"{bm:.4f}", "Brier — 0 is perfect, 0.25 is a coin flip")
        with c2:
            ui.kpi("Market + Wowza", f"{bmw:.4f}", "same fixtures, same scale")
        with c3:
            ui.kpi("What Wowza adds", f"{bm - bmw:+.5f}", f"n = {int(n.sum()):,}",
                   tone=ui.NEUTRAL if abs(bm - bmw) < 0.001 else ui.polarity()[0])
        st.markdown(
            f"**Read it plainly.** Adding Wowza to the market's own de-vigged price changes the "
            f"score by {bm - bmw:+.5f} on {int(n.sum()):,} fixtures. That is indistinguishable "
            f"from zero. On this evidence the model knows nothing the price has not already "
            f"priced — which is exactly why v11 defaults to NO_BET rather than to a tier."
        )
    ui.table(d)

st.divider()

# ── does retraining help ──────────────────────────────────────────────────────────────────────
st.header("3 · Does retraining help?")
if not pro.available():
    ui.absent(pro.absent_reason())
else:
    sl = pro.shadow_learning()
    if not sl.get("available"):
        st.info(sl.get("why", "no walk-forward results"))
    else:
        v = sl["variants"]
        st.markdown(
            "This is the one clearly positive result on the page, and it is positive in the way "
            "real effects are: small, and consistent across independent months."
        )
        cols = st.columns(len(v))
        for c, (name, m) in zip(cols, sorted(v.items())):
            with c:
                ui.kpi(name.title(), f"{m['log_loss']:.5f}",
                       f"log loss · AUC {m.get('auc', float('nan')):.4f}",
                       tone=ui.polarity()[0] if name == sl.get("best_by_logloss") else None)
        if "frozen" in v and "canonical" in v:
            st.markdown(
                f"**Retraining wins.** A model retrained on the canonical store scores "
                f"{v['canonical']['log_loss']:.5f} against {v['frozen']['log_loss']:.5f} for one "
                f"left frozen, across {sl['months']} walk-forward months. The consequence is a "
                f"rule: **never delete old data.** Every window, decay and filter experiment that "
                f"threw history away lost to simply keeping it."
            )

st.divider()

# ── fantasy, on its own scale ─────────────────────────────────────────────────────────────────
st.header("4 · Fantasy — against a public benchmark")
b = fantasy.benchmark()
if not b.get("available"):
    st.info(b.get("why", "no benchmark yet"))
else:
    ui.state_banner("PAPER", "Fantasy is scored in FPL points. Never added to betting units.")
    c1, c2, c3 = st.columns(3)
    with c1:
        ui.kpi("Wowza error", ui.points(b["wowza_mae"]), "mean absolute error")
    with c2:
        ui.kpi("FPL's own ep_next", ui.points(b["fpl_mae"]), "the public benchmark")
    with c3:
        ui.kpi("Gameweeks won", f"{b['gameweeks_wowza_better']} of {b['gameweeks']}",
               b["verdict"], tone=ui.polarity()[1] if b["gameweeks_wowza_better"] == 0 else None)
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
                    use_container_width=False)
    ui.table(d)
