'''
# Step 1: Hello-GPU on Modal
#
# TODO 1: import modal (and torch/platform where you use them), create the App
# TODO 2: define an image with torch installed
# TODO 3: cloud function on a T4 -> returns (GPU name, hostname)
# TODO 4: local entrypoint that calls it with .remote() and prints the result
#
# Run with:  modal run hello_modal.py
'''

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

  
