# Gate-7 preparation — performance research notes

**ENGINEERING RESEARCH ONLY.**

External guidance used for the execution-scaling plan:

- PyTorch Performance Tuning Guide: avoid unnecessary CPU/GPU synchronizations, explicitly including `cuda_tensor.item()`.
- PyTorch `torch.profiler`: profile operator CPU/CUDA time, memory, input shapes and execution traces before choosing an optimization.
- PyTorch `torch.compile`: `reduce-overhead` can use CUDA graphs to reduce Python/kernel-launch overhead, but dynamic shapes and graph breaks can trigger recompilation or prevent the intended benefit.
- NVIDIA CUDA Graph guidance: graphs mainly remove CPU/launch overhead and are most useful when the CPU cannot feed the GPU quickly enough; they are not a substitute for fixing GPU-bound or algorithmic bottlenecks.
- NVIDIA Nsight Systems: use CUDA/PyTorch traces and GPU-idle/starvation analysis to distinguish CPU launch/synchronization bottlenecks from GPU compute/memory limits.

Project interpretation:

1. first remove Python object scaling, full-reserve sorts, hot-path `.item()` synchronizations and repeated frontier construction;
2. profile the eager tensorized baseline;
3. only then test compiler/CUDA-graph modes as a separate execution variable.

This document does not claim measured speedups. Measurements must come from the user's local hardware/profile artifacts.
