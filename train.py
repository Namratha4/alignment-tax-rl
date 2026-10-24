"""Step 5a: Modal training image + environment check. No training yet."""

# TODO 1: import modal; create the App
# TODO 2: vol = modal.Volume.from_name("al-tax", create_if_missing=True)
# TODO 3: image = debian_slim(python 3.12) + pip_install(trl[vllm]==1.14.0, peft==0.21.0, datasets, wandb)
#                 + .env({"HF_HOME": "/vol/hf-cache"}) + .add_local_python_source(...)
# TODO 4: check_env() on gpu="A100-80GB": import torch, trl, peft, transformers, vllm INSIDE the function;
#         return their versions + torch.cuda.get_device_name(0)
# TODO 5: local entrypoint that calls check_env.remote() and prints the result
#
# Run with:  modal run train.py     (first run builds the image -- slow, several minutes; costs cents)

import modal

app = modal.App('alignment-train')

vol = modal.Volume.from_name("al-tax", create_if_missing=True)

image = (
   modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "trl[vllm]==1.14.0", "vllm==0.30.0", "transformers==5.17.0",
"peft==0.21.0", "datasets==5.0.1", "wandb==0.30.0", "accelerate==1.15.0",
    )
    .env({
        "HF_HOME": "/vol/hf-cache",
    })
    .add_local_python_source("gsm8k_reward", "grpo_config")
)

@app.function(image=image, gpu="A100-80GB", volumes={"/vol": vol}, timeout=60 * 30)
def check_env():
  import torch, trl, peft, transformers, vllm
  return {
    "torch": torch.__version__,
    "trl": trl.__version__,
    "peft": peft.__version__,
    "transformers": transformers.__version__,
    "vllm": vllm.__version__,
    "gpu": torch.cuda.get_device_name(0),
  }

@app.function(image=image, gpu="A100-80GB", volumes={"/vol": vol}, timeout=2 * 60 * 60, secrets=[modal.Secret.from_name("wandb-secret")])  

def train(run_name: str, beta: float, max_steps: int, learning_rate:float, save_steps: int, full_ft: bool):
  from trl import GRPOTrainer
  import gsm8k_reward, grpo_config, glob, os
  from gsm8k_reward import correctness_reward, format_reward

  ds = gsm8k_reward.build_dataset()
  args = grpo_config.make_training_args(run_name, beta, max_steps, learning_rate, save_steps, output_root="/vol/checkpoints", save_only_model=full_ft)
  peft_config = grpo_config.peft_config

  from transformers import TrainerCallback
  class CommitOnSave(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
      vol.commit()

  if full_ft:
    peft_config = None
  trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    reward_funcs=[correctness_reward, format_reward],
    args = args,
    train_dataset=ds,
    peft_config = peft_config,
    callbacks=[CommitOnSave()]
    
  )
  checkpoint_dirs = glob.glob(os.path.join(args.output_dir, "checkpoint-*"))
  # full-FT checkpoints are weights-only (no optimizer state), so they can't be resumed cleanly
  resume_from_checkpoint = len(checkpoint_dirs) > 0 and not full_ft
  trainer.train(resume_from_checkpoint=resume_from_checkpoint)
  vol.commit()
  
@app.local_entrypoint()
def main(run_name: str, beta: float = 0.0,  max_steps: int = 20, learning_rate: float = 1e-5, save_steps: int = 5, full_ft: bool = False):
  # result = check_env.remote()
  train.remote(run_name, beta, max_steps, learning_rate, save_steps, full_ft)
  # print(result)


# ---- Step 8: parameterize + make long runs survivable ----
# TODO 1: train(...) and main(...) take max_steps: int, learning_rate: float, save_steps: int
#         and pass them to grpo_config.make_training_args. Keep the TYPE HINTS on main --
#         that's what makes Modal parse --max-steps etc. as numbers (the eval.py bug).
# TODO 2: raise train's timeout. 300 steps ~ 48 min; a timeout kills the run and loses
#         anything not committed. Pick something with real headroom (e.g. 3 hours).
# TODO 3: commit checkpoints DURING training, not just at the end. Write a TrainerCallback:
#             from transformers import TrainerCallback
#             class CommitOnSave(TrainerCallback):
#                 def on_save(self, args, state, control, **kwargs):
#                     vol.commit()
#         and pass callbacks=[CommitOnSave()] to GRPOTrainer.
# TODO 4: resume if a checkpoint already exists. trainer.train(resume_from_checkpoint=True)
#         continues from the newest checkpoint-* in output_dir, but raises an error if there
#         is none. So check first (glob.glob(f"{args.output_dir}/checkpoint-*")) and pass a bool.
#
# Run naming: put the config in the run name so folders never collide, e.g.
#   modal run train.py --run-name naive_b0_lr5e-5_s300 --beta 0 --learning-rate 5e-5 --max-steps 300 --save-steps 50
#
# Knowledge checks:
#   1. Why does TODO 3 matter if train() already calls vol.commit() at the end?
#      What exactly is lost if the container dies at step 250 without it?
#   2. With resume enabled, what happens if you launch the SAME run name twice by accident?
#      Is that good or bad for an experiment you want to be a clean, fresh run?
#   3. The 20-step run at lr=1e-5 barely moved the model. If you raise the learning rate a lot
#      with beta=0, what are the two different things that could happen to eval scores?
#      (one is the "forgetting" you want to measure, the other is a different failure)


  
    


# ---- Step 5b: training function (write below) ----
# TODO 1: pin the versions in pip_install above (trl[vllm], vllm, transformers, peft, datasets, wandb, accelerate)
# TODO 2: train(run_name, beta) on A100-80GB, mount the Volume, secrets=[modal.Secret.from_name("wandb-secret")]
#           inside: import GRPOTrainer, and your modules (gsm8k_reward, grpo_config) INSIDE the function
#           ds = build_dataset(); args = make_training_args(run_name, beta, output_root="/vol/checkpoints")
#           trainer = GRPOTrainer(model=..., reward_funcs=[...], args=args, train_dataset=ds, peft_config=peft_config)
#           trainer.train(); vol.commit()
# TODO 3: change main() to take run_name and beta and call train.remote(...)
#
# Run with:  modal run train.py --run-name naive_beta0 --beta 0     (~10-20 min on an A100, roughly $1)