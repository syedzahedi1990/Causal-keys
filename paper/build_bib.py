"""Build references.bib from arXiv metadata (no hand-typed bibliographic data for arXiv papers).

Each entry: our citation key -> arXiv id. Titles/authors/years come from the arXiv API; the script prints
them so they can be checked against the intended paper. Non-arXiv sources are added verbatim below.
"""
import re
import time
import urllib.request
from pathlib import Path

ARXIV = {
    "geiger2021inducing": "2112.00826", "geiger2024finding": "2303.02536", "makelov2024subspace": "2311.17030",
    "wu2024reply": "2401.12631", "lieberum2023chinchilla": "2307.09458",
    "wiegreffe2025answer": "2407.15018", "tulchinskii2024wise": "2410.02343", "ok2026lost": "2601.14152",
    "wong2026decide": "2601.03914", "sun2026persuaded": "2605.09314", "oh2026rebinding": "2606.08644",
    "wu2026record": "2609.24635", "cheng2026steering": "2604.08524", "zou2026introspection": "2609.35108",
    "opielka2026causality": "2602.22424", "gao2026encodings": "2608.22985", "liu2026wrong": "2609.39243",
    "shih2026steering": "2606.29522", "steele2026routing": "2607.11945", "steele2026spaces": "2607.10248",
    "feng2023binding": "2310.17191", "mueller2025mib": "2504.13151", "wang2022ioi": "2211.00593",
    "sclar2023quantifying": "2310.11324", "heimersheim2024patching": "2404.15255", "zhang2023patching": "2309.16042",
    "vig2020causal": "2004.12265", "venkatesh2026steering": "2602.06801", "hewitt2019control": "1909.03368",
    "huang2024ravel": "2402.17700", "arora2024causalgym": "2402.12560", "li2026bucketing": "2605.02234",
    "yang2024qwen25": "2412.15115", "olmo2025two": "2501.00656", "bojieli2026notes": "2606.17107",
    "pustovit2026packs": "2604.03270", "zhou2026ordinal": "2610.00910", "feucht2025dualroute": "2504.03022",
    "ma2025addressbook": "2512.10547", "tan2024steering": "2407.12404",
    "olsson2022context": "2209.11895",
    # v5 additions (each checked against its arxiv.org/abs page: title, authors, date, comments)
    "kobayashi2020norm": "2004.10102",      # Attention is Not Only a Weight (EMNLP 2020)
    "mcdougall2023copy": "2310.04625",      # Copy Suppression: Comprehensively Understanding an Attention Head
    "merullo2024reuse": "2310.08744",       # Circuit Component Reuse Across Tasks in Transformer Language Models (ICLR 2024)
    "dai2024binding": "2409.05448",         # Representational Analysis of Binding in Language Models
    "prakash2024finetuning": "2402.14811",  # Fine-Tuning Enhances Existing Mechanisms (ICLR 2024)
    "gurarieh2025mixing": "2510.06182",     # Mixing Mechanisms (ICLR 2026; OpenReview UJ2UUjT2ko)
    "robinson2023mcsb": "2210.12353",       # Leveraging LLMs for Multiple Choice Question Answering (ICLR 2023)
    "turner2023actadd": "2308.10248",       # Steering Language Models With Activation Engineering
    "panickssery2023caa": "2312.06681",     # Steering Llama 2 via Contrastive Activation Addition
    "bussmann2024batchtopk": "2412.06410",  # BatchTopK Sparse Autoencoders
    "rajpurkar2016squad": "1606.05250",     # SQuAD (EMNLP 2016)
    "longpre2021conflicts": "2109.05052",   # Entity-Based Knowledge Conflicts in Question Answering (EMNLP 2021)
    "grattafiori2024llama3": "2407.21783",  # The Llama 3 Herd of Models
    "gemma2024gemma2": "2408.00118",        # Gemma 2: Improving Open Language Models at a Practical Size
    "abdin2024phi4": "2412.08905",          # Phi-4 Technical Report
    "liu2024kivi": "2402.02750",            # KIVI (ICML 2024); appendix only
    "yang2025qwen3": "2505.09388",          # Qwen3 Technical Report
    "jiang2023mistral": "2310.06825",       # Mistral 7B
}

# Corrections to arXiv author metadata (diacritics, team names, name order), applied after fetching.
AUTHOR_FIX = {
    "opielka2026causality": [("Opiełka", "Opie{\\l}ka")],
    "yang2024qwen25": [(" Qwen and  : and ", "")],
    "olmo2025two": [("Team OLMo", "{Team OLMo}")],
    "li2026bucketing": [("Li Puyin", "Puyin Li")],
    "gemma2024gemma2": [("Gemma Team", "{Gemma Team}")],
    "gurarieh2025mixing": [("Gur-Arieh", "{Gur-Arieh}")],
    "prakash2024finetuning": [("Rott Shaham", "{Rott Shaham}")],
}

MANUAL = r"""
@inproceedings{prakash2025lookbacks,
  title = {{Language Models Use Lookbacks to Track Beliefs}},
  author = {Nikhil Prakash and Natalie Shapira and Arnab Sen Sharma and Christoph Riedl and Yonatan Belinkov and Tamar {Rott Shaham} and David Bau and Atticus Geiger},
  booktitle = {International Conference on Learning Representations},
  year = {2026},
  note = {arXiv:2505.14685},
  url = {https://openreview.net/forum?id=6gO6KTRMpG}
}
@misc{elhage2021framework,
  title        = {A Mathematical Framework for Transformer Circuits},
  author       = {Elhage, Nelson and Nanda, Neel and Olsson, Catherine and Henighan, Tom and Joseph, Nicholas and Mann, Ben and Askell, Amanda and Bai, Yuntao and Chen, Anna and Conerly, Tom and DasSarma, Nova and Drain, Dawn and Ganguli, Deep and Hatfield-Dodds, Zac and Hernandez, Danny and Jones, Andy and Kernion, Jackson and Lovitt, Liane and Ndousse, Kamal and Amodei, Dario and Brown, Tom and Clark, Jack and Kaplan, Jared and McCandlish, Sam and Olah, Chris},
  year         = {2021},
  howpublished = {Transformer Circuits Thread},
  url          = {https://transformer-circuits.pub/2021/framework/index.html}
}
@misc{tii2024falcon3,
  title        = {{Falcon3-7B-Instruct}},
  author       = {{Technology Innovation Institute}},
  year         = {2024},
  howpublished = {Hugging Face model card},
  url          = {https://huggingface.co/tiiuae/Falcon3-7B-Instruct}
}
@misc{andyrdt2025saes,
  title        = {Residual stream {SAEs} for {Qwen2.5-7B-Instruct}},
  author       = {{andyrdt}},
  year         = {2025},
  howpublished = {Hugging Face repository andyrdt/saes-qwen2.5-7b-instruct, revision c37e53c4},
  url          = {https://huggingface.co/andyrdt/saes-qwen2.5-7b-instruct}
}
@misc{anonymous2026fitted,
  title  = {Beyond the Fitted Answer: Causal Dissection and Downstream Consequences of Learned Activation Interventions},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review at ICLR 2027}
}
@misc{kamath2025attention,
  title  = {Tracing Attention Computation Through Feature Interactions},
  author = {Kamath, Harish and Ameisen, Emmanuel and others},
  year   = {2025},
  howpublished = {Transformer Circuits Thread},
  url    = {https://transformer-circuits.pub/2025/attention-qk/index.html}
}
@misc{mistral2025small,
  title  = {{Mistral-Small-24B-Instruct-2501}},
  author = {{Mistral AI}},
  year   = {2025},
  howpublished = {Model card},
  url    = {https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501}
}
"""


def fetch(ids):
    url = "https://export.arxiv.org/api/query?id_list=" + ",".join(ids) + "&max_results=100"
    return urllib.request.urlopen(url, timeout=60).read().decode()


def main():
    keys = list(ARXIV)
    xml = ""
    for i in range(0, len(keys), 20):
        xml += fetch([ARXIV[k] for k in keys[i:i + 20]])
        time.sleep(3)
    entries = {}
    for e in xml.split("<entry>")[1:]:
        aid = re.search(r"<id>http://arxiv.org/abs/([^<v]+)", e).group(1)
        title = " ".join(re.search(r"<title>(.*?)</title>", e, re.S).group(1).split())
        authors = re.findall(r"<name>(.*?)</name>", e)
        year = re.search(r"<published>(\d{4})", e).group(1)
        entries[aid] = (title, authors, year)
    out = []
    for k in keys:
        aid = ARXIV[k]
        if aid not in entries:
            raise SystemExit(f"arXiv id not found: {k} {aid}")
        title, authors, year = entries[aid]
        auth = " and ".join(authors[:12]) + (" and others" if len(authors) > 12 else "")
        for old, new in AUTHOR_FIX.get(k, []):
            auth = auth.replace(old, new)
        out.append(f"@article{{{k},\n  title = {{{{{title}}}}},\n  author = {{{auth}}},\n  journal = {{arXiv preprint arXiv:{aid}}},\n  year = {{{year}}}\n}}")
        print(f"{k:26s} {aid}  {year}  {title[:90]}")
    Path(__file__).with_name("references.bib").write_text("\n".join(out) + "\n" + MANUAL)


if __name__ == "__main__":
    main()
