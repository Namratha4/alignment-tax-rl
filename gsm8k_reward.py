"""Step 3: GSM8K dataset builder and verifiable reward functions."""

# TODO 1: extract_answer(text) -> str | None    (contents of <answer>...</answer>)
# TODO 2: correctness_reward(completions, answer, **kwargs) -> list[float]
#         (normalize commas / $ before comparing; 1.0 match, 0.0 otherwise)
# TODO 3: format_reward(completions, **kwargs) -> list[float]   (small reward if tag present)
# TODO 4: build_dataset() -> Dataset with "prompt" (chat messages) and "answer" (text after ####)
# TODO 5: __main__ block testing with fake completions: correct, wrong, no tags, "$1,234" style
#
# Run with:  python gsm8k_reward.py
import re
from datasets import load_dataset

def extract_answer(text):
  res = None
  match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
  if match:
    res = match.group(1)
  return res

def correctness_reward(completions, answer, **kwargs) -> list[float]:
  result = []
  for comp, ans in zip(completions,answer):
    content = comp[0]['content']
    ans = ans.replace('$','').replace(',','').strip()
    
    if extract_answer(content) and extract_answer(content).replace('$','').replace(',','').strip() == ans:
      result.append(1.0)
    else:
      result.append(0.0)
  return result
      

def format_reward(completions, **kwargs):
  score = []
  for comp in completions:
    text = comp[0]['content']
    match = extract_answer(text)
    if match:
      score.append(0.1)
    else:
      score.append(0)
  return score

def build_dataset():
  ds = load_dataset("openai/gsm8k", "main", split="train")

  def to_prompt(row):
    return {
      "prompt": [
        {"role": "system", "content":"You are helpful math assistant. Solve this quant question with step by step reasoning and come up with single solution. Do not hallucinate. After solving make sure to return the answer only in opening and closing tags which will be in this format : <answer> x </answer>"},
         {
         "role": "user", "content":row["question"]
         },
      ],
      "answer" : row["answer"].split('####')[-1].strip()
    }
  ds = ds.map(to_prompt)
  return ds

if __name__ == "__main__":
  ds = build_dataset()
  print(ds[0])
  completions = [
    [{"role": "assistant", "content": "...5 + 7 = 12... <answer>12</answer>"}],   # prompt A, sample 1
    [{"role": "assistant", "content": "...I think it's 15 <answer>15</answer>"}], # prompt A, sample 2
    [{"role": "assistant", "content": "...<answer>$1,234</answer>"}],             # prompt B, sample 1
    [{"role": "assistant", "content": "no answer here at all"}],                  # prompt B, sample 2
]
  fake_answers = ["12", "12", "1234", "5"]
  print(correctness_reward(completions, fake_answers))
  print(format_reward(completions))
  
  
  