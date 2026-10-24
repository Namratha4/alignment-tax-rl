"""Modal app: GRPO training on an A100 (LoRA, or --full-ft for all parameters), with checkpoint committing and resume.

Run with:  modal run train.py --run-name NAME --beta B --learning-rate LR --max-steps 300 --save-steps 100 [--full-ft]
"""

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
  train.remote(run_name, beta, max_steps, learning_rate, save_steps, full_ft)
