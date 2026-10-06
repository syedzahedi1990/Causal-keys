"""Shared pytest setup: at most 16 CPU threads. The FP32 exactness tests compare batch rows whose inputs are equal;
on a many-core box the batched GEMMs block rows by batch position at high thread counts (a 4-core Xeon gives
bitwise-equal rows up to 48 threads and 1.6e-5 nats at 64), so the cap keeps the tests independent of the host."""
import torch

torch.set_num_threads(min(torch.get_num_threads(), 16))
