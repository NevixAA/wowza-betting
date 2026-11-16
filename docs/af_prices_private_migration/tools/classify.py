import re
# (repo, path regex, class, provider, dataset label)
RULES = [
 ("wowza-betting", r"^output/book_quotes/", "A","API-Football (source=api_football)","book_quotes"),
 ("wowza-betting", r"^output/specials_quotes/", "A","API-Football","specials_quotes"),
 ("wowza-betting", r"^output/prop_quotes/", "A","mixed: API-Football + OddsAPI (source column)","prop_quotes"),
 ("wowza-betting", r"^output/player_prop_odds_history(\.csv|_archive/)", "A","mixed: API-Football + OddsAPI (source column)","player_prop_odds_history"),
 ("wowza-betting", r"^output/newformat_odds_history\.csv$", "A","API-Football Bet365 (nf_odds_capture)","newformat_odds_history"),
 ("wowza-betting", r"^output/standard_sidemarket_odds_history\.csv$", "A","API-Football Bet365 (std_odds_capture)","standard_sidemarket_odds_history"),
 ("wowza-betting", r"^output/odds_heartbeat/", "A","API-Football (capture heartbeat)","odds_heartbeat"),
 ("wowza-betting", r"^output/book_odds_snapshots(\.csv|_archive/)", "O","OddsAPI (predict)","book_odds_snapshots"),
 ("wowza-betting", r"^output/exchange_quotes/", "O","OddsAPI exchanges","exchange_quotes"),
 ("wowza-betting", r"^output/standard_odds_history\.csv$", "O","OddsAPI (predict drift archive)","standard_odds_history"),
 ("wowza-betting", r"^output/newformat_odds_dense\.csv$", "O","OddsAPI (predict drift archive)","newformat_odds_dense"),
 ("wowza-betting", r"^odds_history_v9\.json$", "O","OddsAPI (predict)","odds_history_v9.json"),
 ("wowza-betting", r"^output/sharp_history/", "O","OddsAPI (sharp_tracker)","sharp_history"),
 ("wowza-betting", r"^output/worldcup_history\.json$", "O","OddsAPI (worldcup tracker)","worldcup_history"),
 ("wowza-betting", r"^output/fd_history\.parquet$", "F","football-data.co.uk","fd_history"),
 ("wowzaV9-Pro", r"^data/season_2026_27/scout_odds/", "A","API-Football","scout_odds"),
 ("wowzaV9-Pro", r"^data/season_2026_27/scout_specials_quotes/", "A","API-Football","scout_specials_quotes"),
 ("wowzaV9-Pro", r"^data/season_2026_27/live_odds_snapshots/", "A","API-Football (live /odds/live)","live_odds_snapshots"),
 ("wowzaV9-Pro", r"^data/season_2026_27/market_snapshots/", "A","mixed: imported from v9 predictions + capture histories (odds_source)","market_snapshots"),
 ("wowzaV9-Pro", r"^data/season_2026_27/book_odds_snapshots/", "O","OddsAPI (imported from v9 book_odds_snapshots.csv; source=v9:book_odds on every row)","pro_book_odds_snapshots"),
 ("wowzaV9-Pro", r"^output/(monthly_learning_by_track|price_trader/slippage_study|valuable_deep_dive/valuable_data_quality)\.csv$", "C","aggregate","aggregate"),
 ("wowza-betting", r"^output/league_market_gate_research\.csv$", "C","aggregate","aggregate"),
 ("wowzaV9-Pro", r"^docs/league_scout_af_odds_probe", "A","API-Football probe","league_scout_af_odds_probe"),
]
def classify(repo, path):
    for r, rx, c, prov, ds in RULES:
        if r == repo and re.search(rx, path):
            return c, prov, ds
    # default: a per-row price inside our own output (ledger / tips / research frame)
    d = re.sub(r"\d{4}-\d{2}(-\d{2})?", "*", path); d = re.sub(r"run=[^/]*", "run=*", d)
    if repo == "wowzaV9-Pro" and path.startswith("data/season_2026_27/"):
        d = "/".join(path.split("/")[:3]) + "/"
    return "B", "own record (price as taken / as observed; upstream mix of API-Football and OddsAPI)", d
