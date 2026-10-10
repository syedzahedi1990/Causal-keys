"""Part E: additions to paper/build_bib.py for paper v5.

Every arXiv id below was checked on 2026-10-10 against the arxiv.org/abs page (citation_title / citation_author /
citation_date meta tags); titles are given here only for the reviewer of this patch, since build_bib.py fetches them.
Venues come from the arXiv 'Comments' field unless marked otherwise.
"""

# Merge into ARXIV in paper/build_bib.py.
ARXIV_ADD = {
    # --- the six works the panel named as missing (Elhage et al. 2021 is not on arXiv; see NON_ARXIV below)
    "gurarieh2025mixing": "2510.06182",     # Mixing Mechanisms: How Language Models Retrieve Bound Entities In-Context. Gur-Arieh, Geva, Geiger. ICLR 2026 (arXiv comment)
    "dai2024binding": "2409.05448",         # Representational Analysis of Binding in Language Models (v1 title: "... in Large Language Models"). Dai, Heinzerling, Inui. EMNLP 2024 (ACL Anthology author page; Anthology URL not opened)
    "prakash2024finetuning": "2402.14811",  # Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking. Prakash, Rott Shaham, Haklay, Belinkov, Bau. ICLR 2024 (arXiv comment)
    "variengien2023leap": "2312.10091",     # Look Before You Leap: A Universal Emergent Decomposition of Retrieval Tasks in Language Models. Variengien, Winsor.
                                            #   ICLR 2025 version titled "Look Before You Leap: Universal Emergent Mechanism for Retrieval in Language Models" (iclr.cc poster 28935)
    "merullo2024reuse": "2310.08744",       # Circuit Component Reuse Across Tasks in Transformer Language Models. Merullo, Eickhoff, Pavlick. ICLR 2024 (arXiv comment)
    # --- further works the rewrite cites (all verified the same way)
    "merullo2024talking": "2406.09519",     # Talking Heads: Understanding Inter-layer Communication in Transformer Language Models. NeurIPS 2024
    "mcdougall2023copy": "2310.04625",      # Copy Suppression: Comprehensively Understanding an Attention Head
    "robinson2023mcsb": "2210.12353",       # Leveraging Large Language Models for Multiple Choice Question Answering. ICLR 2023
    "geva2023dissecting": "2304.14767",     # Dissecting Recall of Factual Associations in Auto-Regressive Language Models. EMNLP 2023
    "kobayashi2020norm": "2004.10102",      # Attention is Not Only a Weight: Analyzing Transformers with Vector Norms. EMNLP 2020
    "springer2024echo": "2402.15449",       # Repetition Improves Language Model Embeddings. ICLR 2025
    "turner2023actadd": "2308.10248",       # Steering Language Models With Activation Engineering (Part C, E1/E2)
    "panickssery2023caa": "2312.06681",     # Steering Llama 2 via Contrastive Activation Addition (Part C, E1/E2)
    "bussmann2024batchtopk": "2412.06410",  # BatchTopK Sparse Autoencoders (Part C, E3; the andyrdt SAEs are BatchTopK)
    "liu2024kivi": "2402.02750",            # KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache. ICML 2024 (Discussion; Part A E5)
    "rajpurkar2016squad": "1606.05250",     # SQuAD: 100,000+ Questions for Machine Comprehension of Text (Part A)
    "longpre2021conflicts": "2109.05052",   # Entity-Based Knowledge Conflicts in Question Answering. EMNLP 2021 (Part A substitution)
    "grattafiori2024llama3": "2407.21783",  # The Llama 3 Herd of Models (Parts A, B)
    "gemma2024gemma2": "2408.00118",        # Gemma 2: Improving Open Language Models at a Practical Size (Parts A, B)
    "abdin2024phi4": "2412.08905",          # Phi-4 Technical Report (Part B)
    # only if used in the text:
    "lieberum2024gemmascope": "2408.05147",  # Gemma Scope (Part C exploratory)
    "gould2023successor": "2312.09230",      # Successor Heads (Part A, A8 discussion)
}

# AUTHOR_FIX additions (team names and multi-word surnames):
AUTHOR_FIX_ADD = {
    "gemma2024gemma2": [("Gemma Team", "{Gemma Team}")],
    "gurarieh2025mixing": [("Gur-Arieh", "{Gur-Arieh}")],
    "prakash2024finetuning": [("Rott Shaham", "{Rott Shaham}")],
}

# Non-arXiv entries, appended verbatim (Elhage et al. 2021 checked against the page itself; Falcon3 and the SAE release are model cards).
NON_ARXIV = r"""
@misc{elhage2021framework,
  title        = {A Mathematical Framework for Transformer Circuits},
  author       = {Elhage, Nelson and Nanda, Neel and Olsson, Catherine and Henighan, Tom and Joseph, Nicholas and Mann, Ben and Askell, Amanda and Bai, Yuntao and Chen, Anna and Conerly, Tom and DasSarma, Nova and Drain, Dawn and Ganguli, Deep and Hatfield-Dodds, Zac and Hernandez, Danny and Jones, Andy and Kernion, Jackson and Lovitt, Liane and Ndousse, Kamal and Amodei, Dario and Brown, Tom and Clark, Jack and Kaplan, Jared and McCandlish, Sam and Olah, Chris},
  year         = {2021},
  howpublished = {Transformer Circuits Thread},
  url          = {https://transformer-circuits.pub/2021/framework/index.html}
}
@misc{tii2024falcon3,
  title        = {Falcon3-7B-Instruct},
  author       = {{Technology Innovation Institute}},
  year         = {2024},
  howpublished = {Hugging Face model card},
  url          = {https://huggingface.co/tiiuae/Falcon3-7B-Instruct}
}
@misc{andyrdt2025saes,
  title        = {Sparse autoencoders for {Qwen2.5-7B-Instruct}},
  author       = {{andyrdt}},
  year         = {2025},
  howpublished = {Hugging Face repository andyrdt/saes-qwen2.5-7b-instruct, revision c37e53c4},
  url          = {https://huggingface.co/andyrdt/saes-qwen2.5-7b-instruct}
}
"""

# Citation of Anonymous (2026): decide at submission time (see design, section v):
#  - accepted and de-anonymised: cite by author names, in the third person, like any other work;
#  - still under review: keep @misc{anonymous2026fitted} with note = {Submitted to ICLR 2027; OpenReview forum <public URL>};
#  - withdrawn and not public: cite the public release of its bases only.
