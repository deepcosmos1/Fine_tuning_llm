import os
from datasets import load_dataset
from huggingface_hub import snapshot_download, login, interpreter_login
import asyncio
import pandas as pd
from sentence_transformers import SentenceTransformer
import random

try:
    interpreter_login()
except:
    pass

cot_df = load_dataset("diabolic6045/flanv2_cot_alpeca", split="train")
cot_df = cot_df.select(random.sample(range(len(cot_df)), 50000))
cot_df.to_csv(os.path.join("/src", "ChainOfThoughts.csv"))

df1 = pd.read_csv(os.path.join("/src", "ChainOfThoughts.csv"))
df1.to_json(os.path.join("/src", "ChainOfThoughts.json"), orient="records", index=False)

fc_df = load_dataset("Trelis/function_calling_v3", split="train")
fc_df = fc_df.rename_column("functionList", "instruction")
fc_df = fc_df.rename_column("userPrompt", "input")
fc_df = fc_df.rename_column("assistantResponse", "output")
fc_df.to_csv(os.path.join("/src", "FunctionCalling.csv"))

df1 = pd.read_csv(os.path.join("/src", "FunctionCalling.csv"))
df1.to_json(os.path.join("/src", "FunctionCalling.json"), orient="records", index=False)

MODEL_DIR = "gte_embedding"
emb_model = SentenceTransformer("thenlper/gte-small")
emb_model.save(MODEL_DIR)

MODEL_DIR = "LLama-2-7b-hf"
snapshot_download(repo_id="meta-llama/Llama-2-7b-hf", local_dir=MODEL_DIR)

async def setup():
    if os.path.split(os.getcwd())[-1] != "axolotl":
        proc = await asyncio.create_subprocess_shell(
            "git clone -b main --depth 1 https://github.com/Arindam0231/axolotl.git",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            error_message = stderr.decode() if stderr else 'Unknown error'
        else:
            os.chdir("axolotl")

os.system("clear")
asyncio.run(setup())
print("\n\n\n\n\n\n\n\n\n!!!SETUP COMPLETE")
