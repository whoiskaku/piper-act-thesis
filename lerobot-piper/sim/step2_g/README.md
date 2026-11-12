# Step 2 G — M0a role-separated synthetic contract

This directory currently contains only an offline synthetic contract test. It
does **not** implement the real Piper model G, calibrate MuJoCo, or establish a
model-quality result.

M0 checks that the future calibration pipeline can:

- validate explicit parameter bounds and a fixed command-scale gauge;
- fit only a deterministic calibration split;
- evaluate an independently seeded, role-separated synthetic holdout after fitting;
- emit deterministic JSON artifacts with configuration, seeds, data roles, and
  SHA-256 provenance.

Run it without a robot or network:

```bash
python -m sim.step2_g.run_m0 --output-dir /tmp/step2-g-m0
```

The command writes `calibration_artifact.json`, `holdout_evaluation.json`, and
`manifest.json` and refuses to overwrite an existing artifact. The holdout is
separated only by the calibration API and data-role label. Its seed, labels and
scores remain visible, so this is **not** the sealed M0.3 evaluation described
in the preregistration draft.

This scaffold does not yet ingest real raw logs, predict IMU, simulate noise or
missing frames, use an externally held evaluator, or define a numerical M0 pass
threshold. Those remain draft work in `reports/PREREG_STEP2_G_DRAFT.md`. The
zero-error synthetic recovery is a software-contract check on a known
second-order plant; it must not be reported as performance of the eventual real
G model or as M0.3/M0.4 having passed.
