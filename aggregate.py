"""Turn the v2 eval JSONs into one comparison table + results/summary.csv. Runs locally, no GPU.

Retention = mean over {MMLU (5 subjects averaged), IFEval, HumanEval} of (run score / base score).
GSM8K is excluded from retention: it is the trained skill, so it is the gain axis, not a retained one.
Refuses to compare runs that were scored with different eval settings."""
import csv
import json
import os
import subprocess

RAW = "results/raw"

# (label, result file stem on the Volume, group). Order = row order. Missing files are skipped.
RUNS = [
    ("base",               "v2_baseline",                          "base"),
    ("LoRA naive 5e-5",    "v2_naive_b0_lr5e-5_s300_ckpt300",      "lora"),
    ("LoRA naive 2e-4",    "v2_naive_b0_lr2e-4_s300_ckpt300",      "lora"),
    ("LoRA KL.04 2e-4",    "v2_kl_b0.04_lr2e-4_s300_ckpt300",      "lora"),
    ("full naive @100",    "v2_full_b0_lr1e-5_s300_ckpt100",       "full"),
    ("full naive @200",    "v2_full_b0_lr1e-5_s300_ckpt200",       "full"),
    ("full naive @300",    "v2_full_b0_lr1e-5_s300_ckpt300",       "full"),
    ("full KL.01 @200",    "v2_full_b0.01_lr1e-5_s300_ckpt200",    "full_kl"),
    ("full KL.01 @300",    "v2_full_b0.01_lr1e-5_s300_ckpt300",    "full_kl"),
    ("full KL.04 @200",    "v2_full_b0.04_lr1e-5_s300_ckpt200",    "full_kl"),
    ("full KL.04 @300",    "v2_full_b0.04_lr1e-5_s300_ckpt300",    "full_kl"),
    ("full KL.2 @200",     "v2_full_b0.2_lr1e-5_s300_ckpt200",     "full_kl"),
    ("full KL.2 @300",     "v2_full_b0.2_lr1e-5_s300_ckpt300",     "full_kl"),
]

# Must match HEADLINE in eval.py (one metric per task, chosen once).
HEADLINE = {
    "gsm8k": "exact_match,flexible-extract",
    "ifeval": "prompt_level_loose_acc,none",
    "humaneval_instruct": "pass@1,create_test",
}
MMLU_KEY = "acc,none"
SETTINGS = ("eval_version", "limit", "batch_size", "apply_chat_template", "lm_eval_version", "tasks")


def fetch(stem):
    """Download one result file from the Modal Volume if we don't have it. Returns the path or None."""
    os.makedirs(RAW, exist_ok=True)
    path = f"{RAW}/{stem}.json"
    if not os.path.exists(path):
        subprocess.run(["modal", "volume", "get", "al-tax", f"eval_results/{stem}.json", path, "--force"],
                       capture_output=True)
    return path if os.path.exists(path) else None


def summarize(data):
    r = data["results"]
    mmlu = [v[MMLU_KEY] for t, v in r.items() if t.startswith("mmlu_")]
    return {
        "gsm8k_flex": r["gsm8k"][HEADLINE["gsm8k"]],
        "gsm8k_strict": r["gsm8k"]["exact_match,strict-match"],
        "mmlu": sum(mmlu) / len(mmlu),
        "ifeval": r["ifeval"][HEADLINE["ifeval"]],
        "humaneval": r["humaneval_instruct"][HEADLINE["humaneval_instruct"]],
    }


def main():
    loaded = {}
    for label, stem, group in RUNS:
        path = fetch(stem)
        if path is None:
            print(f"(skipping {label}: {stem}.json not on the Volume yet)")
            continue
        loaded[label] = (json.load(open(path)), group)

    base_meta = loaded["base"][0]["meta"]
    for label, (data, _) in loaded.items():
        for key in SETTINGS:
            assert data["meta"].get(key) == base_meta.get(key), f"{label}: setting '{key}' differs from base"
    print(f"{len(loaded)} runs, all scored with identical v2 settings\n")

    rows = {label: summarize(d) for label, (d, _) in loaded.items()}
    base = rows["base"]
    header = ["run", "group", "gsm8k_flex", "gsm8k_strict", "mmlu", "ifeval", "humaneval", "retention", "gsm8k_delta"]
    out = []
    for label, s in rows.items():
        retention = (s["mmlu"] / base["mmlu"] + s["ifeval"] / base["ifeval"] + s["humaneval"] / base["humaneval"]) / 3
        out.append([label, loaded[label][1], s["gsm8k_flex"], s["gsm8k_strict"], s["mmlu"], s["ifeval"],
                    s["humaneval"], retention, s["gsm8k_flex"] - base["gsm8k_flex"]])

    print(f"{'run':18s}{'GSM8K flex':>11s}{'strict':>8s}{'MMLU':>8s}{'IFEval':>8s}{'HumanEval':>10s}{'retention':>11s}{'GSM8K d':>9s}")
    for r in out:
        print(f"{r[0]:18s}{r[2]:11.3f}{r[3]:8.3f}{r[4]:8.3f}{r[5]:8.3f}{r[6]:10.3f}{r[7]*100:10.1f}%{r[8]:+9.3f}")

    os.makedirs("results", exist_ok=True)
    with open("results/summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(out)
    print("\nwrote results/summary.csv")


if __name__ == "__main__":
    main()
