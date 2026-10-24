"""Modal app: the v2 evaluation. Scores a model (base, LoRA adapter, or full checkpoint) on a fixed battery and
writes totals and per-question outputs to the Volume.

Run with:  modal run eval.py --run-name NAME [--adapter-path PATH | --model-id PATH]
"""

import modal
app = modal.App("alignment-eval")

vol = modal.Volume.from_name("al-tax", create_if_missing=True)

# Eval settings: fixed ONCE. Every run (baseline, naive, KL, sweep) must use identical values,
# or the retention numbers aren't comparable.
APPLY_CHAT_TEMPLATE = True
BATCH_SIZE = 8
EVAL_VERSION = "v2"  # v1 = batch 1, limit 50. v2 = batch 8, limit 200. Never compare across versions.
MMLU_SUBJECTS = [  # deliberately far from math, spread across MMLU's categories
    "high_school_us_history",              # humanities
    "high_school_biology",                 # STEM, non-math
    "philosophy",                          # humanities
    "high_school_government_and_politics", # social sciences
    "clinical_knowledge",                  # other
]
MMLU_TASKS = [f"mmlu_{s}" for s in MMLU_SUBJECTS]
DEFAULT_TASKS = ["gsm8k", *MMLU_TASKS, "ifeval", "humaneval_instruct"]  # instruct variant pairs with chat template

# ONE headline metric per task (the y-axis of the trade-off scatter is built from these).
HEADLINE = {
    "gsm8k": "exact_match,flexible-extract",
    **{t: "acc,none" for t in MMLU_TASKS},
    "ifeval": "prompt_level_loose_acc,none",
    "humaneval_instruct": "pass@1,create_test",
}

image = (
   modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "lm-eval[ifeval]==0.4.13", "transformers==5.17.0",
          "torch==2.13.0", "accelerate==1.15.0", "peft==0.21.0"
    ).env({"HF_ALLOW_CODE_EVAL": "1"})
)


@app.function(image=image, gpu="A10G", volumes={"/vol": vol}, timeout=60*60)
def run_eval(model_id, tasks, limit, run_name, adapter_path=None, max_length = None):
  import json, lm_eval, os
  from importlib.metadata import version

  model_args = f"pretrained={model_id},dtype=bfloat16"
  if adapter_path:
    model_args += f",peft={adapter_path}"
  if max_length:
    model_args += f",max_length={max_length}"
  

  meta = {
    "model_id": model_id,
    "adapter_path": adapter_path,
    "tasks": tasks.split(","),
    "limit": limit,
    "apply_chat_template": APPLY_CHAT_TEMPLATE,
    "batch_size": BATCH_SIZE,
    "eval_version": EVAL_VERSION,
    "lm_eval_version": version("lm_eval"),
    "max_length": max_length,
  }

  results = lm_eval.simple_evaluate(
    model="hf",
    model_args=model_args,
    tasks=tasks.split(","),
    limit=limit,                  
    device="cuda:0",
    batch_size=BATCH_SIZE,
    apply_chat_template=APPLY_CHAT_TEMPLATE,
    confirm_run_unsafe_code=True,
  )

  output_dir = '/vol/eval_results'
  os.makedirs(output_dir, exist_ok=True)
  output_path = f'{output_dir}/{run_name}.json'

  with open(output_path, "w") as f:
    json.dump({"meta": meta, "results": results['results']}, f, indent=2)

  # Per-question outputs, kept separate from the totals file. Lets you read real generations and
  # run paired comparisons between two runs on the same questions.
  samples_dir = '/vol/eval_samples'
  os.makedirs(samples_dir, exist_ok=True)

  def slim(sample):
    out = {k: v for k, v in sample.items() if k not in ("doc", "arguments", "doc_hash", "prompt_hash", "target_hash")}
    try:
      out["prompt_tail"] = str(sample["arguments"][0][0])[-600:]
    except Exception:
      out["prompt_tail"] = None
    return out

  slim_samples = {task: [slim(s) for s in rows] for task, rows in results.get("samples", {}).items()}
  with open(f'{samples_dir}/{run_name}.json', "w") as f:
    json.dump(slim_samples, f, default=str)
  vol.commit()
  return results['results']


@app.local_entrypoint()
def main(model_id: str = "Qwen/Qwen2.5-1.5B-Instruct", tasks: str = ",".join(DEFAULT_TASKS), limit: int = 200, run_name: str = "v2_baseline", adapter_path: str = "", max_length: int = 0):
  print(run_eval.remote(model_id, tasks, limit, run_name, adapter_path or None, max_length or None))
