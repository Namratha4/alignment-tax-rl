"""Step 6: baseline eval on Modal. Decoupled from training -- reads a model
(base model, or later a checkpoint path), scores it, writes JSON to the Volume."""

# TODO 1: import modal; app = modal.App("alignment-eval"); reuse the SAME volume
#         (modal.Volume.from_name("al-tax", create_if_missing=True) -- must match train.py's name)
# TODO 2: image = debian_slim + pip_install("lm-eval==0.4.13", "transformers==5.17.0",
#           "torch==2.13.0", "accelerate==1.15.0")   (pin versions, same habit as train.py)
# TODO 3: run_eval(model_id, tasks, limit, run_name) on gpu="A10G", volumes={"/vol": vol}, timeout=60*30
#           inside: import lm_eval and json INSIDE the function
#           results = lm_eval.simple_evaluate(...)   -- see syntax above
#           save results["results"] as JSON to f"/vol/eval_results/{run_name}.json"
#             (os.makedirs the parent dir first)
#           vol.commit()
#           return results["results"]   -- so main() can print it
# TODO 4: local entrypoint main(model_id="Qwen/Qwen2.5-1.5B-Instruct",
#           tasks="gsm8k,mmlu_high_school_mathematics,ifeval", limit=50, run_name="baseline")
#           calls run_eval.remote(...) and prints the result
#
# Knowledge checks -- answer before running:
#   1. Why pass limit=50 instead of no limit, given this same function gets re-run after
#      EVERY training variant later (Phase 2, 3, 4)? Think in terms of total GPU-hours
#      across the whole project, not just this one run.
#   2. Why must every future call to this function use the SAME tasks/limit as this
#      baseline run? What would happen to the trade-off scatter if run 3 used limit=200
#      but the baseline used limit=50?
#   3. Why save to a JSON file on the Volume instead of just printing the dict? Think about
#      Phase 4, where you'll need to combine numbers from many separate eval runs into one
#      chart, without re-running any of them.
#
# Run with:  modal run eval.py --run-name baseline


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


# ---- Step 9: drift probe (wikitext perplexity) ----
# Accuracy on 50 questions only registers big changes. Perplexity on generic text is
# continuous: it moves when the model moves, even if no benchmark answer flips.
# TODO 1: if run_eval gets an optional max_length (default None). If set, append
#         f",max_length={max_length}" to model_args. Add it to `meta` too.
#         WHY: wikitext slides a window of max_length tokens over each article, and the
#         default window is the model's full context -- with a 150K-token vocabulary
#         that is an out-of-memory crash on the A10G. Do NOT set it globally: it would
#         change the other tasks and invalidate your baseline.
# TODO 2: main() gets max_length: int = 0 (keep the type hint!) and passes
#         `max_length or None` through.
# TODO 3: trial first, it's cheap:
#           modal run eval.py --tasks wikitext --max-length 2048 --limit 5 --run-name probe_trial
#         then the real probes (all use limit 50 and max-length 2048, never change them):
#           base:   modal run eval.py --tasks wikitext --max-length 2048 --run-name probe_base
#           naive:  ... --run-name probe_naive_lr5e-5_ckpt300 --adapter-path /vol/checkpoints/naive_b0_lr5e-5_s300/checkpoint-300
# TODO 4: read the metric keys off the output (expect word_perplexity / byte_perplexity /
#         bits_per_byte, each followed by ",none"). Pick ONE as the probe headline.
#
# Knowledge checks:
#   1. Lower perplexity is better. If the tuned model's perplexity on Wikipedia goes UP vs
#      base, what does that say about the model, and why can't accuracy see it?
#   2. Entropy collapsed during training (0.35 -> 0.08). Predict: does wikitext perplexity
#      go up, down, or stay flat for the tuned model? Write your prediction BEFORE running.
#   3. Perplexity rising doesn't prove "capability lost". What else could it mean, and what
#      would you check next?


# ---- Step 7: add code eval (humaneval) ----
# TODO 1: image: add .env({"HF_ALLOW_CODE_EVAL": "1"}) to the image chain
# TODO 2: simple_evaluate(...): add confirm_run_unsafe_code=True
# TODO 3: main(): add "humaneval" to the default tasks string
# TODO 4: run it. Reusing run_name="baseline" overwrites the 3-task file with a 4-task one --
#         that's intended (one JSON per run, all four tasks in it).
# TODO 5: after it finishes, find humaneval's metric key in the output, then add a module-level
#         dict HEADLINE = {task: metric_key} with ONE chosen metric per task (the convention
#         for every future comparison). Known keys so far:
#           gsm8k -> "exact_match,flexible-extract"   mmlu_high_school_mathematics -> "acc,none"
#           ifeval -> "prompt_level_loose_acc,none"   humaneval -> (read it off the output)
#
# Knowledge checks:
#   1. HumanEval runs code the MODEL wrote. What's the risk, and what in our setup contains it?
#   2. pass@1 on 50 problems: what's the smallest change in score you can even observe,
#      and what does that imply about reading a 2-point difference between two runs?
#   3. Why is a single HEADLINE metric per task needed for the trade-off scatter (the y-axis
#      is "average retained capability")? What goes wrong if you average strict-match for
#      one task with loose-match for another?
  
  
  