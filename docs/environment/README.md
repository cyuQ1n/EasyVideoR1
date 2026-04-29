# Qwen3.5 RL Environment Exports

This directory stores the tested runtime environment for the `easyvideor1-for-qwen3.5` training branch.

Files:

- `easyvideor1-for-qwen3.5.conda.yaml`: full `conda env export`
- `easyvideor1-for-qwen3.5.conda-explicit.txt`: explicit Conda package lock
- `easyvideor1-for-qwen3.5.pip-freeze.txt`: `pip freeze` from the tested environment
- `easyvideor1-for-qwen3.5.summary.txt`: concise runtime summary

Tested stack summary:

- Python `3.11.14`
- CUDA `12.9`
- PyTorch `2.10.0+cu129`
- Transformers `5.5.4`
- vLLM `0.19.1`
- Ray `2.54.0`
- qwen-vl-utils `0.0.14`
- flash-attn `2.8.3`
- flash-linear-attention `0.4.2`
- torchcodec `0.11.1`

Validation:

- `vllm serve /path/to/Qwen3.5-2B --port 8101 --tensor-parallel-size 1 --max-model-len 262144 --reasoning-parser qwen3`
- an OpenAI-compatible client request against the local vLLM server

Notes:

- The environment was validated for Qwen3.5 inference and this branch's RL training stack.
- The exported files are the authoritative record for reproducing the tested environment on this branch.
