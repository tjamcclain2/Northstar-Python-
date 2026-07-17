Subproject-2: GPU acceleration and memory-bandwidth optimization of a Python/NumPy pipeline

Overview (what was slow)

Initially, the inherited code's runtime was dominated by two operations that scale with n_angles × n_amplitudes: building the model response grid and comparing it against the real signal. At the target size (500 angles × 500 amplitude combinations) these touch arrays of shape (n_angles, n_amplitudes, n_times, n_detectors) — multiple gigabytes for one array with billions of elements — and the comparison step in particular builds a huge intermediate array only to sum it away. The work is highly parallel but ran mostly serially on CPU, and a naive first CuPy port was actually slower than NumPy (too many tiny GPU launches and host↔device transfers). The goal was to exploit that parallelism: vectorize the math, move it to the GPU, and fuse the dominant operation into a custom kernel.

For reference, all the outputs and performance improvement mentioned are on a (500 angles × 500 amplitude combination)

Thought Process
Measure first, optimize second; I only wanted to touch what actually dominated runtime.
1) Profiled the whole pipeline with cProfile + graphviz. generate_model_detector_responses and get_best_fit_angles_deltas dominated; everything else was negligible.
2) Line-profiled those two functions with line_profiler and its @profile to find the exact bottleneck lines: in the best-fit step, the abs(real - model) + sum; in model generation, building the full response grid and the nested for loop.
3) Diagnosed the cause as memory bandwidth, not compute. Those lines do almost no arithmetic per element but move billions of elements and materialize a full-size intermediate. So the win had to come from moving less data.
4) Vectorized and batched the model path. Removing the nested for loop first required every helper it called (source vector, change-of-basis, beam pattern, time delay, oscillatory terms) to be batch-aware. That meant, promoting their shapes from a single (3,) triple/(3,3) matrix to (n_ang,3)/(n_ang,3,3) stacks via broadcasting and ellipsis einsum. Only then could I replace the per-angle/per-amplitude dot products with a single batched einsum. Then ported to CuPy,  fixed naive-port overhead, and migrated fp64→fp32 (halves bytes moved — near-free since the bottleneck is bandwidth).
5) Fused the dominant operation into a custom CuPy ReductionKernel that computes the abs-difference and sums in one pass, reconstructing each model element on the fly so the giant response grid is never materialized. Trades a little redundant arithmetic for eliminating gigabytes of memory traffic.
6) Verified correctness after every change against the original NumPy reference (rtol ≈ 1e-4 for fp32). Because of the changed shapes, fused kernel and the fp32, I maintained a correctness harness that compares every function's GPU output against the original NumPy reference implementation.
7) Distinguished cold-start cost from steady-state cost. The first CUDA call in any process pays a one-time tax — cuSOLVER/cuRAND handle creation, NVRTC kernel JIT-compilation — that has nothing to do with the algorithm itself. A single fresh invocation is ~5× slower than a second call in the same process. Since these represent genuinely different real-world usage patterns (a one-shot script run vs. many calls from a long-lived process, e.g. a Monte Carlo loop), reporting only one number would misrepresent the other, so the pipeline takes an explicit warmup flag and both numbers are reported separately below.

I benchmarked to calculate the times of the single and weighted best-fit steps separately (via cupyx.profiler.benchmark) because after fusion they dropped to sub-millisecond, where a single inline timer is mostly noise; benchmark handles GPU sync and runs warmed-up repeats for a stable per-stage number.

Key results/ Findings
Measured on an NVIDIA GeForce RTX 4070, at 500 angles × 500 amplitude combinations.

|                 ---                 |   Time   | Speedup vs. NumPy baseline |
|                 ---                 |   ---    |           ---              |
| Original (NumPy, CPU)               | ~5.7 s   |            —               |
| GPU, cold start (single invocation) | ~0.150 s |          ~37×              |
| GPU, steady-state (repeated calls)  | ~0.023 s |          ~245×             |

- The dominant line — `abs(real - model)` + `sum`, inside the best-fit comparison — dropped from ~1.27 s to ~0.007 s (~180×) once fused into a single custom `ReductionKernel`.
- That fusion also eliminates a ~2.6 GB intermediate array per call, which is the actual source of the win: the bottleneck was memory bandwidth, not arithmetic.
- Correctness held throughout: every function's GPU output matches the NumPy reference to rtol ≈ 1e-4 across the full correctness harness.

Repo Structure

Devin_Work/
├── src/                          # the optimized implementation (this subproject)
│   └── devin_optimized.py
├── baseline/                     # original NumPy implementation, kept for comparison/correctness only
│   └── northstar_og_abid.py
├── Tests/                        # correctness harness: compares GPU output against the NumPy baseline
│   ├── abid_testing.py
│   ├── devin_optimized_testing.py
│   └── test_correctness.py
├── profile/                      # cProfile + gprof2dot call-graph visualizations, before/after
│   ├── pre_optimized_profile.dat
│   ├── pre_optimized_profile.png
│   ├── current_profile.dat
│   └── current_profile.png
├── results/                      # accumulated run output
│   └── results.txt
├── pyproject.toml                # dependency source of truth (uv-managed)
├── uv.lock
└── Readme.md

Installation/Environment

How to reproduce

