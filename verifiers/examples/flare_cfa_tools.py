from openai import OpenAI

import os
import sys
sys.path.append(os.path.abspath("/mnt/data/oanaf/verifiers"))


import verifiers as vf
from verifiers.tools import python
from verifiers.tools import search
from verifiers.utils import preprocess_dataset
from verifiers.prompts import SEARCH_FEW_SHOT


"""
Evaluating multi-turn reasoning before/after training.

CUDA_VISIBLE_DEVICES=0,1 vllm serve 'Qwen/Qwen2.5-7B-Instruct' --tensor_parallel_size 2 --max_model_len 8192 --dtype bfloat16 \
    --gpu_memory_utilization 0.9 --enable_prefix_caching \
    --host 0.0.0.0 --port 8001

uv run verifiers/examples/math_eval.py
"""

TOOL_PROMPT = """
Think step-by-step inside <reasoning>...</reasoning> tags, then either call a tool inside <tool>...</tool> tags, or give your final answer inside <answer>...</answer> tags.

You have access to the following tools to help solve problems:

{tool_descriptions}

Tools can be called by writing a JSON command inside <tool> tags with:
- "name": the name of the tool to use
- "args": the arguments for the tool

You will then see the tool's output inside <result> tags. You may call tools multiple times if needed.

The <answer>...</answer> tags should contain only your final answer.

"""


ZERO_SHOT_PROMPT = """
You are a CFA (chartered financial analyst) taking a test to evaluate your
knowledge of finance. You will be given a question along with three possible
answers (A, B, and C). Indicate the correct answer (A, B, or C).
"""

dataset = preprocess_dataset("flare-cfa")
print(dataset[0])


# vf_env = vf.MultiTurnEnv(
#     eval_dataset=dataset,
#     system_prompt=ZERO_SHOT_PROMPT,
#     few_shot=[],
#     #tools=[search, python],
#     max_steps=3
# )
# print(vf_env.system_prompt)

vf_env = vf.ToolEnv(
    eval_dataset=dataset, #.select(range(5)),
    system_prompt=f"{ZERO_SHOT_PROMPT}\n\n{TOOL_PROMPT}",
    few_shot=[],  #SEARCH_FEW_SHOT, #[]
    tools=[search],
    max_steps=2
)
print(vf_env.system_prompt)





model_name = "Qwen/Qwen2.5-7B-Instruct"
base_url = "http://10.0.6.217:8001/v1"
client = OpenAI(base_url=base_url, api_key="EMPTY")
vf_env.eval_api(client, model_name, max_concurrent=20, sampling_args={"temperature": 0.0})

