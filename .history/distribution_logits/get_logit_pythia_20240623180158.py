# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv

load_dotenv()

sns.set_style("darkgrid")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
auth_token = os.environ.get("HF_TOKEN")

# Initialize tokenizer and model for Pythia
model_id = "EleutherAI/pythia-160m"  # Change to the desired Pythia model
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    truncation_side="left",
    padding_side="right",
)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map={"": 0},
)

# %%
import pandas as pd

df = pd.read_csv("distribution_logits/prompts_small.csv")
df.columns = ["prompt"]
prompts = df["prompt"].tolist()
prompts = prompts[:10]
print(prompts)

# %%
def prompt_to_tokens(tokenizer, system_prompt, instruction, model_output):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    dialog_content = B_SYS + system_prompt + E_SYS + instruction.strip()
    dialog_tokens = tokenizer.encode(
        f"{B_INST} {dialog_content.strip()} {E_INST} {model_output.strip()}"
    )
    return torch.tensor(dialog_tokens).unsqueeze(0)

system_prompt = "You are a helpful, honest and concise assistant."

logits_list = []
for prompt in prompts:
    if "chat-hf" in model_id:
        inputs = prompt_to_tokens(tokenizer, system_prompt, prompt, "")
    else:
        inputs = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

    inputs = inputs.to(model.device)

    with torch.no_grad():
        outputs = model(inputs)
        logits = outputs.logits
        logits_list.append(logits)

print(logits_list[0].shape)

# %%
# Normalization and sorting functions remain unchanged
# ...

# %%
# Save and plotting code remains unchanged
# ...

# %%
for i in range(len(prompts)):
    print(f"Top and worst 15 tokens for prompt: {prompts[i]}")
    # Token printing code remains unchanged
    # ...

# %%
# Plotting code remains unchanged
# ...