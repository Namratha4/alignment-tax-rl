"""LoRA config and GRPO training arguments. beta, steps, learning rate and save interval are parameters.

Print a config locally (no GPU):  python grpo_config.py --run-name t --beta 0.04
"""

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
