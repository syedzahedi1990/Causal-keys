"""Critic pilot (tokenizers only, no model, no story seed touched): which alternative container/location words are
single tokens as ' w' (lowercase, leading space) in all 8 Part-B models + the Yi fallback, i.e. usable as a second
location lexicon for a lexically fresh population. Also checks ' w,' and ' w.' contexts (list and sentence end)."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoTokenizer
MODELS = ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3",
          "allenai/OLMo-2-1124-7B-Instruct", "unsloth/Meta-Llama-3.1-8B-Instruct", "unsloth/gemma-2-9b-it",
          "microsoft/phi-4", "tiiuae/Falcon3-7B-Instruct", "01-ai/Yi-1.5-9B-Chat"]
WORDS = ["bin", "crate", "tray", "jar", "bucket", "chest", "trunk", "sack", "bowl", "safe", "vase", "pot", "case",
         "cart", "locker", "pouch", "barrel", "cupboard", "fridge", "oven", "desk", "bed", "sofa", "car", "garage",
         "bag", "suitcase", "wardrobe", "pocket", "tin", "can", "jug", "kettle", "pan", "sink", "tub", "attic",
         "basement", "kitchen", "garden", "room", "hall", "office", "table", "chair", "box", "basket", "shelf",
         "drawer", "cabinet", "closet"]
ok = {w: [] for w in WORDS}
toks = {}
for m in MODELS:
    try:
        toks[m] = AutoTokenizer.from_pretrained(m)
    except Exception as e:
        print("LOAD FAIL", m, type(e).__name__, str(e)[:100])
for w in WORDS:
    for m, t in toks.items():
        a = t(" " + w, add_special_tokens=False).input_ids
        b = t("the " + w + ",", add_special_tokens=False).input_ids
        c = t("the " + w + ".", add_special_tokens=False).input_ids
        single = len(a) == 1 and a[0] in b and a[0] in c
        ok[w].append(single)
print("models loaded:", len(toks))
good = [w for w in WORDS if all(ok[w])]
print("single-token ' w' in all loaded models (and stable before ',' and '.'):", good)
print("fails:", {w: [m.split('/')[1][:12] for m, s in zip(toks, ok[w]) if not s] for w in WORDS if not all(ok[w])})
