# BTTS edge validation — is it broad, league-specific, timing-specific, or variance?

Date: 2026-10-01. Data: `output/side_bets_ledger.csv`, staked tiers (SNIPER + MARKSMAN) only.
Cells: `output/pro_btts_validation.csv`.

## Verdict

**League-specific, and the whole sample is five weeks old.**

BTTS is the best-looking thing in production — **+30.60u on 100 staked bets, +30.6% ROI**. It is
also 64% Argentina, and outside Argentina there is **no demonstrated edge**.

| | n | P/L | ROI | 95% CI on ROI/bet |
|---|---|---|---|---|
| **Argentina Primera** | 64 | **+26.01u** | **+40.6%** | **[+0.066, +0.708] — excludes zero** |
| everything else | 36 | +4.58u | +12.7% | **[−0.155, +0.402] — includes zero** |

Argentina carries **85% of the P/L on 64% of the bets**. Strip it out and the remaining 36 bets
cannot be distinguished from luck.

## The cuts

**By league.** Only Argentina has both a usable sample and an interval excluding zero. Brazil
(n=13, +27.1%) and Serie B (n=9, +10.3%) are too small to support a conclusion in either
direction. Japan is negative on 5.

**By model track.** new-format **+34.1% on 90 bets** (CI [+0.009, +0.628], just excludes zero);
standard **−0.7% on 10**. BTTS is effectively a new-format phenomenon — which is also where
Argentina sits, so these two cuts are not independent evidence. They are the same finding seen
twice.

**By odds band.** Positive in every band, 1.8 through 2.5, and the hit rate beats break-even in
all of them. This is the one cut that looks genuinely broad rather than concentrated. The
`(2.5, 99]` band at +157% is 4 bets and means nothing.

**By month.** August +39.4% (n=23), September +28.0% (n=77). Both positive, so it is not a
single hot week — but two months inside one five-week window is not two independent periods.

**By tier.** MARKSMAN +37.9% (n=33), SNIPER +27.0% (n=67). The lower tier outperforming the
higher one is the same tier-inversion the main O/U track shows, and is further reason not to
read tier as signal strength.

## What this is not

It is **not** evidence of a permanent BTTS edge. Three reasons:

1. **Five weeks.** 2026-08-22 to 2026-09-28. No regime change, no winter, no season turnover.
2. **One league.** The only cell whose confidence interval excludes zero is Argentina, and that
   is also the cell that was most likely to look good by chance, because it has the most bets.
3. **Searched, not predicted.** These cells were examined *after* seeing the P/L. Sixteen cells
   were tested; at a 5% threshold roughly one would clear by chance alone even if nothing were
   real. Nothing here was preregistered.

## What would settle it

Registered in `registry/preregistered_hypotheses.json` before any further data arrives:

> **H-BTTS-01.** BTTS staked selections in Argentina Primera División achieve ROI > 0 with a
> 95% block-bootstrap lower bound above zero, on at least 150 settled bets occurring **after
> 2026-10-01**.
>
> **H-BTTS-02.** The same, for BTTS staked selections **excluding** Argentina. This is the
> hypothesis that decides whether BTTS is a market edge or a league artifact.

If H-BTTS-01 confirms and H-BTTS-02 fails, BTTS is an Argentina phenomenon and should be treated
as one — sized accordingly, and never extrapolated to other leagues. If both fail, the current
+30.60u was variance.

At ~22 staked BTTS bets a week, 150 Argentina bets is roughly **17 weeks** — late January 2027.

## Production recommendation

`RESEARCH`. No change to BTTS selection, staking or thresholds.

The temptation is to raise BTTS weight now because it is the only profitable market in the
estate. That is precisely the §23 trap: a good five-week run is evidence generation, not proof.
The cheapest correct action is to keep taking the bets exactly as now and let the preregistered
test run.
