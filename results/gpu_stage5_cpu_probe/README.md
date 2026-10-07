# Stage 5, the exploratory GPT-2 small attention probe, re-run on CPU

`experiments/ioi_attention.py` failed on the GPU box (device mismatch; `results/gpu_stage5/FAILED.txt`). It was re-run on
a 4-core CPU in FP32 from a clean worktree of commit 51e105e with the box's command:
`python experiments/ioi_attention.py --model gpt2 --n 200 --seed 1 --dtype float32 --out <dir> --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e`
(log: `log_ioi_attention_gpt2_cpu.txt`). Exploratory only (preregistration G, part (e)); not read by the scorer.
