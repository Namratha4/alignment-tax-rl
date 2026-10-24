"""Step 4 (pass 1): LoRA + GRPO config for the naive (beta ~ 0) run. Config objects only, no training."""

# TODO 1: import LoraConfig (peft) and GRPOConfig (trl)
# TODO 2: lora_config -- r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"
# TODO 3: training_args = GRPOConfig(...) with:
#           output_dir            (a path; later this will live on the Volume)
#           num_generations       G, completions per prompt
#           per_device_train_batch_size   counts COMPLETIONS -- must be divisible by num_generations
#           gradient_accumulation_steps   more prompts per optimizer update
#           beta                  KL coefficient -- near 0 for the naive run
#           learning_rate
#           max_steps             small (~20) for the throwaway run
#           use_vllm=True, vllm_mode="colocate"
#           report_to="wandb"
# TODO 4: __main__ block that prints both objects, so you can check they construct
#
# Questions to be ready for:
#   - how many unique prompts does one optimizer update see with your values?
#   - why beta near 0 for this run (which row of the experiment matrix is it)?
#
# Run with:  python grpo_config.py   (needs `pip install trl peft` locally; no GPU needed)

from peft import LoraConfig
from trl import GRPOConfig
import argparse

peft_config = LoraConfig(
  r = 16,
  lora_alpha = 32,
  target_modules="all-linear",
  task_type="CAUSAL_LM"
)

def make_training_args(run_name, beta, max_steps=20, learning_rate=1e-5, save_steps=5, output_root="outputs", save_only_model=False):
  return GRPOConfig(
    output_dir=f"{output_root}/{run_name}",
    num_generations=4,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    beta=beta,                    
    learning_rate=learning_rate,
    max_steps=max_steps,
    save_steps=save_steps,             
    use_vllm=True,
    vllm_mode="colocate",
    report_to="wandb",
    save_only_model=save_only_model,
  )

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--run-name", default = "local_test")
  parser.add_argument("--beta",type=float, default = 0.0)
  parser.add_argument("--max-steps",type=int, default = 20)
  parser.add_argument("--learning-rate",type=float, default = 1e-05)
  parser.add_argument("--save-steps",type=int, default = 5)
  parser.add_argument("--save-only-model", action="store_true")
  args = parser.parse_args()

  training_args = make_training_args(args.run_name,args.beta, args.max_steps, args.learning_rate, args.save_steps, save_only_model=args.save_only_model)

  print(peft_config)
  print(training_args)


# ---- Step 8: parameterize for Phase 2 ----
# TODO 1: make_training_args takes max_steps, learning_rate, save_steps as parameters
#         (give them defaults so old calls still work), and uses them in GRPOConfig.
# TODO 2: add --max-steps (int), --lr (float), --save-steps (int) to the argparse block
#         in __main__ and pass them through, so you can check the printed config locally.
#
# Useful numbers: one optimizer step = 16 prompts x 4 completions = 64 completions
# (you worked this out earlier). GSM8K train has 7473 prompts. So one epoch ~ how many steps?
  
  
  
  
  
  
   