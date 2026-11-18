"""Build the harness's committed feature frame once (offline) and cache it for the EFL diagnosis.
Writes only to the --out folder (outside the repo). No prices are printed."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.retrain_harness import protocol as P

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
cfg = P.load_config()
feat, info = P.load_committed_frame(cfg)
feat.to_parquet(out / "feat.parquet", index=False)
(out / "frame_info.json").write_text(json.dumps(info, indent=2, default=str))
print(info)
