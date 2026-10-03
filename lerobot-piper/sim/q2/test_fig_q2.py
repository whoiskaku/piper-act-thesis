import json

import numpy as np

import fig_q2 as FIG
import q2_run as RUN
import q2_testing as T


def _fake_fold(out, eps, steps):
    """一折种子 0 的终训产物（只有作图要读的字段）：零响应 / H / 两种候选的逐集预测与 num、den。"""
    rng = np.random.default_rng(0)
    rows = lambda s: [{"name": e.name, "num": s * (i + 1), "den": 10.0} for i, e in enumerate(eps)]
    k = lambda e: e.name.replace("/", "__")
    base = RUN.nom_final_base(out, "ab7", 0)
    base.parent.mkdir(parents=True)
    np.savez(base.with_suffix(".npz"), **{f"{p}__{k(e)}": rng.normal(size=(e.n, 3)) for e in eps for p in ("nom", "H")})
    base.with_suffix(".json").write_text(json.dumps({"zero": rows(1.0), "H": rows(2.0)}))
    for form, s in steps.items():
        b = RUN.inc_final_base(out, form, "ab7", 0)
        b.parent.mkdir(parents=True)
        np.savez(b.with_suffix(".npz"), **{f"cand__{k(e)}": rng.normal(size=(e.n, 3)) for e in eps})
        b.with_suffix(".json").write_text(json.dumps({"cand": rows(0.5), "step_star": s}))


def test_fold_rows_reads_saved_predictions_and_per_episode_rho(tmp_path, monkeypatch):
    eps = [T.synthetic(s, "ab7", "D", "corr") for s in range(3)]
    monkeypatch.setattr(RUN, "load_eps", lambda folds, kinds: eps)
    _fake_fold(tmp_path, eps, {"nonlinear": 0, "condlinear": 250})
    rows, steps = FIG.fold_rows(tmp_path, "ab7")
    z = np.load(RUN.nom_final_base(tmp_path, "ab7", 0).with_suffix(".npz"))
    assert [r["name"] for r in rows] == [e.name for e in eps] and steps == {"nonlinear": 0, "condlinear": 250}
    assert np.array_equal(rows[1]["pred"]["zero"], z["nom__ab7__D_ep1"])
    assert rows[1]["rho"] == {"zero": 0.2, "H": 0.4, "nonlinear": 0.1, "condlinear": 0.1}


def test_plot_picks_median_and_hardest_by_zero_response_and_writes_png(tmp_path):
    rng = np.random.default_rng(1)
    rows = []
    for i in range(5):
        e = T.synthetic(i, "ab7", "D", "corr")
        rows.append({"name": e.name, "ep": e, "pred": {p: rng.normal(0, 0.01, (e.n, 3)) for p in FIG.PREDS},
                     "rho": {p: 1.0 + i for p in FIG.PREDS}})
    med, hard = FIG.pick(rows)
    assert (med["name"], hard["name"]) == (rows[2]["name"], rows[4]["name"])
    p = tmp_path / "f.png"
    FIG.plot(rows, "ab7", {"nonlinear": 0, "condlinear": 250}, p)
    assert p.exists() and p.stat().st_size > 10_000


def test_refuses_to_draw_from_an_unverified_run(tmp_path, capsys):
    assert FIG.main(["--out", str(tmp_path)]) == 4
    assert "不作图" in capsys.readouterr().out and not (tmp_path / "fig").exists()


def test_not_part_of_the_code_fingerprint():
    import q2_train as TR
    assert "fig_q2.py" not in TR.CODE_FILES_Q2
