#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Upsert a run into the UI registry (build/ui_runs/index.json) so runs are VISIBLE-BY-DEFAULT
# in the web UI — a row appears the moment a run starts, and the deploy/report stage upserts
# the artifact paths later. Never removes other runs; sets created_at once (on first insert).
# Runs standalone (stdlib + pyyaml); safe to call from inside the NGC container.
import argparse, json, os, re, time
try:
    import yaml
except Exception:
    yaml = None


def slug(s):
    return (re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")[:48] or "run")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)                 # run name -> runs/<run>, key custom_<run>
    ap.add_argument("--config")                             # yaml: model_id / model_short_name / epochs
    ap.add_argument("--label")
    ap.add_argument("--data-dir"); ap.add_argument("--eval-labels")
    ap.add_argument("--orig-dir"); ap.add_argument("--ft-dir")
    ap.add_argument("--report"); ap.add_argument("--comparison"); ap.add_argument("--samples-dir")
    ap.add_argument("--ft-onnx"); ap.add_argument("--ft-engine")
    ap.add_argument("--blurb")
    ap.add_argument(
        "--require-baseline", action="store_true",
        help="refuse registration unless original engine_metrics.json and perf.json are non-empty",
    )
    ap.add_argument("--index", default="build/ui_runs/index.json")
    a = ap.parse_args()

    cfg = {}
    if a.config and yaml and os.path.exists(a.config):
        try: cfg = yaml.safe_load(open(a.config)) or {}
        except Exception: cfg = {}
    run = a.run; key = f"custom_{slug(run)}"
    idx = []
    if os.path.exists(a.index):
        try: idx = json.load(open(a.index))
        except Exception: idx = []
    prev = next((e for e in idx if e.get("key") == key), None)

    # External training runners call this more than once. An omitted artifact argument means
    # "keep the registered path", not "replace it with a convention-derived default". The old
    # behavior silently rewrote a valid baseline path at AutoML start, making the Baseline tab
    # disappear even though the artifacts were present.
    def path_value(field, explicit, default):
        return explicit or (prev or {}).get(field) or default

    model_id = cfg.get("model_id", "model"); short = cfg.get("model_short_name") or model_id.split("/")[-1]
    data_dir = path_value("data_dir", a.data_dir, f"data/{run}")
    orig_dir = path_value("orig_dir", a.orig_dir, f"models/{run}_orig")
    if a.require_baseline:
        missing = [
            os.path.join(orig_dir, "eval", name)
            for name in ("engine_metrics.json", "perf.json")
            if not os.path.isfile(os.path.join(orig_dir, "eval", name))
            or os.path.getsize(os.path.join(orig_dir, "eval", name)) == 0
        ]
        if missing:
            ap.error(
                "baseline is required before training; missing or empty: "
                + ", ".join(missing)
                + ". Run DeepStream Stages 1-2 first or pass --orig-dir."
            )
    fields = {
        "key": key, "kind": "custom", "label": a.label or f"{short} — {run}",
        "model_id": model_id, "model_label": model_id.split("/")[-1],
        "dataset_id": run, "dataset_label": run, "blurb": a.blurb or "auto-registered run.",
        "editable": False, "deploy_recipe": True, "config_generate": False,
        "assets_dir": ".claude/skills/deepstream-eval-and-finetune/examples/rtdetr-aerial-sheep/assets",
        "config_yaml": None, "data_dir": data_dir, "run_dir": f"runs/{run}",
        "orig_dir": orig_dir, "ft_dir": path_value("ft_dir", a.ft_dir, f"models/{run}_ft"),
        "eval_split": "valid", "eval_labels": a.eval_labels or f"{data_dir}/labels.txt",
        "orig_onnx": "rtdetr_orig.onnx", "orig_engine": "rtdetr_orig_ds.engine",
        "ft_onnx": path_value("ft_onnx", a.ft_onnx, f"{run}_ft.onnx"),
        "ft_engine": path_value("ft_engine", a.ft_engine, f"{run}_ft_ds.engine"),
        "canvas": [640, 640], "n_eval": 200, "bbox_format": "xywh", "resize_mode": "stretch",
        "epochs": int(cfg.get("num_train_epochs", 20)),
        "comparison": path_value("comparison", a.comparison, f"reports/{run}_comparison.json"),
        "samples_dir": path_value("samples_dir", a.samples_dir, f"reports/{run}_samples"),
        "report_pdf": path_value("report_pdf", a.report, f"reports/{run}_report.pdf"),
        "report_basename": f"{run}_report.pdf",
        "sample_overlays_extra": [], "hparams": {},
    }
    if prev:
        prev.update(fields)                       # keep existing created_at
    else:
        fields["created_at"] = time.time(); idx.append(fields)
    os.makedirs(os.path.dirname(a.index) or ".", exist_ok=True)
    json.dump(idx, open(a.index, "w"), indent=2)
    print(f"[ui_register] upserted {key} in {a.index} ({'updated' if prev else 'new'})")


if __name__ == "__main__":
    main()
