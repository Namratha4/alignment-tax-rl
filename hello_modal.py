"""Hello-GPU: run a function on a Modal T4 and report the GPU name and host it ran on.

Run with:  modal run hello_modal.py
"""

import modal
import platform

app = modal.App("alignment-tax")

image = modal.Image.debian_slim().pip_install("torch")

@app.function(gpu="T4", image=image)
def my_fn():
  import torch
  gpu_name = (
    torch.cuda.get_device_name(0) if torch.cuda.is_available()
    else "No GPU Available"
  )
  # print(gpu_name)
  return (gpu_name, platform.node())


@app.local_entrypoint()
def main():
  result = my_fn.remote()
  print(result)
