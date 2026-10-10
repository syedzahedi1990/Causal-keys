"""Capture pinned revisions and per-file hashes (LFS sha256 / git blob id) of every Part-B model repo and of the
ungated mirrors whose bytes equal the gated official files."""
import json, time
from huggingface_hub import HfApi
api = HfApi()
repos = ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3",
         "allenai/OLMo-2-1124-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "google/gemma-2-9b-it",
         "microsoft/phi-4", "tiiuae/Falcon3-7B-Instruct", "NousResearch/Meta-Llama-3.1-8B-Instruct",
         "modularai/Llama-3.1-8B-Instruct-GGUF", "unsloth/gemma-2-9b-it", "thr3a/gemma-2-9b-it",
         "dnhkng/RYS-Gemma-2-9b-it", "google/gemma-2-27b-it", "unsloth/gemma-2-27b-it", "mistralai/Mistral-Small-24B-Instruct-2501"]
man = {}
for r in repos:
    for attempt in range(6):
        try:
            info = api.model_info(r, files_metadata=True)
            break
        except Exception as e:
            print(r, "retry", attempt, str(e)[:80]); time.sleep(65)
    files = {s.rfilename: {("sha256" if s.lfs else "git_blob"): (s.lfs.sha256 if s.lfs else s.blob_id), "size": s.size}
             for s in info.siblings if s.rfilename.endswith((".json", ".safetensors", ".model", ".jinja"))
             and not s.rfilename.startswith("original/")}
    man[r] = {"revision": info.sha, "last_modified": str(info.last_modified), "gated": info.gated, "files": files}
    gb = sum(v["size"] or 0 for k, v in files.items() if k.endswith(".safetensors")) / 1e9
    print(f"{r:45s} {info.sha} {str(info.last_modified)[:10]} gated={info.gated} {gb:.1f} GB", flush=True)
    time.sleep(3)
json.dump(man, open("/tmp/claude-0/-home-user-Causal-keys/7f2a5307-16d9-55d5-8797-c2900bcd44b4/scratchpad/s8design/partB/hf_manifest.json", "w"), indent=1)
# official vs mirror check
def cmp(off, mir, names):
    o, m = man[off]["files"], man[mir]["files"]
    return {n: (n in o and n in m and o[n] == m[n]) for n in names}
st = lambda r: [k for k in man[r]["files"] if k.endswith(".safetensors")] + ["model.safetensors.index.json"]
print("llama weights NousResearch:", all(cmp("meta-llama/Llama-3.1-8B-Instruct", "NousResearch/Meta-Llama-3.1-8B-Instruct", st("meta-llama/Llama-3.1-8B-Instruct")).values()))
print("llama cfg/tok NousResearch:", cmp("meta-llama/Llama-3.1-8B-Instruct", "NousResearch/Meta-Llama-3.1-8B-Instruct", ["config.json", "generation_config.json", "tokenizer.json", "special_tokens_map.json", "tokenizer_config.json"]))
print("llama tok modularai:", cmp("meta-llama/Llama-3.1-8B-Instruct", "modularai/Llama-3.1-8B-Instruct-GGUF", ["config.json", "tokenizer.json", "tokenizer_config.json"]))
print("gemma9 weights unsloth:", all(cmp("google/gemma-2-9b-it", "unsloth/gemma-2-9b-it", st("google/gemma-2-9b-it")).values()))
print("gemma9 tok unsloth:", cmp("google/gemma-2-9b-it", "unsloth/gemma-2-9b-it", ["tokenizer.json", "tokenizer.model", "special_tokens_map.json", "tokenizer_config.json", "config.json"]))
print("gemma9 cfg thr3a:", cmp("google/gemma-2-9b-it", "thr3a/gemma-2-9b-it", ["config.json"]), "tokcfg dnhkng:", cmp("google/gemma-2-9b-it", "dnhkng/RYS-Gemma-2-9b-it", ["tokenizer_config.json", "generation_config.json"]))
print("gemma27 weights unsloth:", all(cmp("google/gemma-2-27b-it", "unsloth/gemma-2-27b-it", st("google/gemma-2-27b-it")).values()),
      cmp("google/gemma-2-27b-it", "unsloth/gemma-2-27b-it", ["tokenizer.json", "tokenizer_config.json", "config.json"]))
