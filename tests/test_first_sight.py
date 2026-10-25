"""The shared first-sight store (src/research/first_sight.py): immutable across reruns, refuses
post-kickoff rows, detects tampering, and the two migrated writers keep their exact behaviour."""
import pandas as pd

from src.research.first_sight import FirstSightStore, frozen_digest

COLS = ["fid", "market", "observed", "kickoff", "p", "result"]


def _rows(*specs):
    return pd.DataFrame([dict(zip(COLS, s)) for s in specs])


def _store(p, **kw):
    return FirstSightStore(p, key=["fid", "market"], columns=COLS, observed_col="observed",
                           kickoff_col="kickoff", frozen_cols=COLS[:5], **kw)


A = ("f1", "ou25", "2026-10-25T10:00:00Z", "2026-10-25T15:00:00Z", 0.51, "")
B = ("f2", "ou25", "2026-10-25T10:00:00Z", "2026-10-25T16:00:00Z", 0.40, "")


def test_reruns_are_byte_identical_and_earliest_wins(tmp_path):
    p = tmp_path / "log.csv"
    s = _store(p)
    assert s.append(_rows(A, B)) == 2
    first = p.read_bytes()
    later = ("f1", "ou25", "2026-10-25T11:00:00Z", "2026-10-25T15:00:00Z", 0.99, "")
    for _ in range(3):
        assert s.append(_rows(later, B)) == 0
    assert p.read_bytes() == first
    assert pd.read_csv(p)["p"].tolist() == [0.51, 0.40]


def test_existing_bytes_are_a_prefix_after_append(tmp_path):
    p = tmp_path / "log.csv"
    s = _store(p)
    s.append(_rows(A))
    before = p.read_bytes()
    s.append(_rows(B))
    after = p.read_bytes()
    assert after.startswith(before) and after.count(b"fid,") == 1          # header written once


def test_within_batch_first_row_wins(tmp_path):
    p = tmp_path / "log.csv"
    dup = ("f1", "ou25", "2026-10-25T10:05:00Z", "2026-10-25T15:00:00Z", 0.77, "")
    assert _store(p).append(_rows(A, dup)) == 1
    assert pd.read_csv(p)["p"].tolist() == [0.51]


def test_post_kickoff_rows_are_refused(tmp_path):
    p = tmp_path / "log.csv"
    at = ("f3", "ou25", "2026-10-25T15:00:00Z", "2026-10-25T15:00:00Z", 0.5, "")
    after = ("f4", "ou25", "2026-10-25T15:01:00Z", "2026-10-25T15:00:00Z", 0.5, "")
    no_ko = ("f5", "ou25", "2026-10-25T15:01:00Z", "", 0.5, "")
    assert _store(p).append(_rows(at, after, no_ko, A)) == 2
    assert pd.read_csv(p)["fid"].tolist() == ["f5", "f1"]


def test_refused_late_row_does_not_block_an_earlier_valid_row_for_the_same_key(tmp_path):
    p = tmp_path / "log.csv"
    late = ("f1", "ou25", "2026-10-25T16:00:00Z", "2026-10-25T15:00:00Z", 0.9, "")
    assert _store(p).append(_rows(late, A)) == 1
    assert pd.read_csv(p)["p"].tolist() == [0.51]


def test_settlement_of_grading_columns_keeps_the_frozen_digest(tmp_path):
    p = tmp_path / "log.csv"
    s = _store(p)
    s.append(_rows(A, B))
    prior = s.integrity()
    d = s.read()
    d.loc[0, "result"] = "WIN"                                   # grading-only column
    d.to_csv(p, index=False, lineterminator="\n")
    r = s.integrity(prior={"rows": prior["rows"], "frozen_digest": prior["frozen_digest"]})
    assert r["prior_match"] is True and r["status"] == "OK"


def test_editing_a_frozen_value_is_detected(tmp_path):
    p = tmp_path / "log.csv"
    s = _store(p)
    s.append(_rows(A, B))
    prior = s.integrity()
    d = s.read()
    d.loc[0, "p"] = "0.6"                                        # rewrite history
    d.to_csv(p, index=False, lineterminator="\n")
    r = s.integrity(prior={"rows": prior["rows"], "frozen_digest": prior["frozen_digest"]})
    assert r["prior_match"] is False and r["status"] == "TAMPERED"


def test_removed_rows_are_detected(tmp_path):
    p = tmp_path / "log.csv"
    s = _store(p)
    s.append(_rows(A, B))
    prior = s.integrity()
    s.read().iloc[:1].to_csv(p, index=False, lineterminator="\n")
    assert s.integrity(prior=prior)["prior_match"] is False


def test_digest_sidecar_fails_closed_on_tamper(tmp_path):
    p = tmp_path / "log.csv"
    s = _store(p, digest=True)
    assert s.append(_rows(A)) == 1 and s.sidecar.exists() and s.verify()["ok"] is True
    d = s.read()
    d.loc[0, "p"] = "0.1"
    d.to_csv(p, index=False, lineterminator="\n")
    assert s.verify()["ok"] is False
    tampered = p.read_bytes()
    assert s.append(_rows(B)) == 0 and p.read_bytes() == tampered       # nothing appended


def test_duplicate_keys_from_a_concurrent_writer_are_reported(tmp_path):
    """Documented assumption: one writer at a time. If two processes both appended the same key,
    integrity() makes the breach visible instead of silent."""
    p = tmp_path / "log.csv"
    s = _store(p)
    s.append(_rows(A))
    _rows(A).to_csv(p, mode="a", header=False, index=False, lineterminator="\n")    # a racing writer
    r = s.integrity()
    assert r["duplicate_keys"] == 1 and r["status"] == "WARN"


def test_frozen_digest_ignores_absent_vs_present_confusion():
    d = pd.DataFrame({"a": ["1"], "b": ["2"]})
    assert frozen_digest(d, ["a", "b"]) != frozen_digest(d, ["a", "c"])
    assert frozen_digest(d, ["a"], n=0) != frozen_digest(d, ["a"], n=1)


def test_append_never_raises(tmp_path):
    s = _store(tmp_path / "missing_dir" / "x" / "log.csv")
    assert s.append(_rows(A)) == 1
    bad = FirstSightStore(tmp_path, key=["fid"], columns=["fid"])                   # path is a directory
    assert bad.append(pd.DataFrame({"fid": ["x"]})) == 0


def test_migrated_writers_use_the_shared_store(tmp_path):
    from src import edge_engine_shadow as EES
    from src import shadow_experiments as SE
    assert EES.first_sight_store(tmp_path / "e.csv").key == EES.KEY
    st = SE.first_sight_store(tmp_path / "x.csv")
    assert st.key == SE.KEY and st.frozen_cols == SE.FIRST_SIGHT_COLS
