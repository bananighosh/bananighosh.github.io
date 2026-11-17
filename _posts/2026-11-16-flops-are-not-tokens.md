---
title: "FLOPS Are Not Tokens: A First-Principles Guide to GPUs and ASICs for LLM Inference"
summary: "Why peak FLOPS do not tell you which chip serves your model faster, and how memory, energy, kernels, compilers, networking, and workload shape actually decide inference performance."
tags: [inference, gpu, asic, trainium]
math: false
---

<details class="toc" markdown="1">
<summary>Table of contents</summary>

* TOC
{:toc}

</details>

---

## 0. What spec sheets do not tell you

Every accelerator spec sheet leads with peak FLOPS. A user sees tokens per second (TPS), and an operator pays per token. During decode, the phase that produces every output token, each step re-reads the model weights and every conversation's KV cache from HBM. For long conversations, the math units wait on those bytes at any batch size (Section 4). While decode stays memory-bound, doubling FLOPS at the same bandwidth leaves tokens per second unchanged.

A GPU is built for generality and carries hardware for many workloads beyond LLM inference. An ASIC such as AWS Trainium3 or OpenAI Jalapeño starts from what inference binds on and spends its die area, power, and HBM interface there.

<p>
<b>The headline numbers do not show which of those choices matter.</b> 
</p>

Performance comes out of a chain:

```txt
Model shape → workload shape → arithmetic intensity → bytes moved → energy per byte
→ compute → communication → scheduling → latency / throughput / cost per token
```

This post walks that chain once, with Trainium3 as the worked example, and names the limit at every link. By the end one can reason from first principles about where an ASIC gains, before looking at a benchmark.

<div class="pullquote" markdown="1">
<b>The recurring theme:</b> do not optimize for the chip. Identify what prevents the chip from doing useful work, and remove that.
</div>

---

## 1. The chips, and the ten rules

### The chips in this post

| Chip | Type | Availability | Design focus |
| --- | --- | --- | --- |
| <b>NVIDIA</b> B300 | General-purpose GPU | Available | Flexible GPUs on CUDA, with NVLink rack-scale domains |
| <b>AMD</b> Instinct MI355X | General-purpose GPU | Available | Flexible GPUs on ROCm |
| <b>AWS Trainium3</b> (Trn3 Gen2 UltraServer, 144 chips) | Hyperscaler ASIC for training and inference | Available on Amazon EC2 | Software-scheduled ASIC: compiler-managed SRAM, 16 dedicated collective cores per chip, and a 144-chip scale-up domain |
| <b>OpenAI Jalapeño</b> | Inference-only ASIC | OpenAI internal use | Inference ASIC built around OpenAI's own serving patterns: 15.4 TB/s of HBM bandwidth at 700 W |

### The ten rules that explain everything below

| Rule | Statement | Section |
| --- | --- | --- |
| Roofline | Attainable speed = min(peak compute, arithmetic intensity × memory bandwidth) | 3 |
| Memory-bandwidth lower bound | Each decode step reads every weight once, so step time ≥ weight bytes ÷ bandwidth | 3 |
| Weight reuse vs KV reuse | Batching shares weight reads; KV cache reads are per sequence unless sequences share a prefix | 3, 4 |
| KV size | Bytes per token = 2 × layers × KV heads × head dim × bytes per element | 4 |
| Data-movement energy | Reading a byte from DRAM costs far more energy than computing on it | 5 |
| Tile quantization | Fixed-size matrix tiles are paid for in full even when mostly empty | 6 |
| α–β model | Message time = α (startup latency) + size ÷ β (bandwidth) | 7 |
| Little's law | Requests in flight = arrival rate × time in system | 12 |
| Pipeline bottleneck | A pipeline runs at the speed of its slowest stage | 11, 12 |
| Queueing | Waiting time explodes as utilization approaches 100% | 12 |

---

## 2. What is physically different inside the chips

A GPU is thousands of programmable workers sharing caches the hardware manages. An ASIC such as Trainium uses a few very large math engines fed by a scratchpad that software, the compiler or a custom kernel, fills ahead of time. Both are programmable. They differ in where the flexibility lives and who does the optimization work.

<figure><img src="/assets/img/flops-are-not-tokens/hierarchy-v4.svg" alt="hierarchy v4" loading="lazy"></figure>

Both designs have software-managed SRAM: shared memory on a GPU, SBUF on Trainium3. The difference is what sits behind it. A GPU also has a hardware-managed L2 cache and hardware warp schedulers, so irregular access patterns and changing shapes are absorbed by cache hits and by switching to another warp.

Trainium3 replaces the hardware cache with explicit DMA: every byte moves on a schedule built ahead of time by the Neuron compiler or an NKI kernel. When the work is predictable, that schedule keeps every engine fed, and serving stacks compile one schedule per shape bucket to keep it that way. Trainium3 adds dynamic shapes and control flow to the NeuronCore-v4 ISA, which can reduce the number of buckets. GPUs also move data under software control, through the Tensor Memory Accelerator (TMA, since Hopper) and asynchronous copies, so the difference is the division of work: on a GPU, hardware caches and warp schedulers handle at run time what software leaves unplanned; on Trainium, the compiler or kernel plans the movement ahead of time.

### Across the chips: anatomy

|  | GPU | AWS Trainium3 | OpenAI Jalapeño |
| --- | --- | --- | --- |
| Compute units | Many streaming multiprocessors (NVIDIA) or compute units (AMD), each with matrix cores and SIMD lanes | 8 NeuronCore-v4, each a 128×128 systolic array plus vector, scalar, and GpSimd engines | Not disclosed |
| On-chip memory | Small shared memory per SM or CU, plus hardware-managed caches | 256 MiB software-managed SBUF (32 MiB per core) | Not disclosed |
| Control | Hardware warp (NVIDIA) or wavefront (AMD) schedulers | Per-engine instruction streams built by the compiler or NKI, synchronized by hardware semaphores; ISA extensions for dynamic shapes and control flow | Not disclosed |
| Collectives | NCCL or RCCL, by default as kernels on the SMs or CUs | Neuron collectives on dedicated CC-Cores, SBUF to SBUF | Not disclosed |
| Low-precision formats (all also run BF16 and FP16) | FP8 plus several microscaled FP8, FP6, and FP4 formats | FP8, MXFP8, MXFP4 | MXFP8, MXFP4 |

### Why build an ASIC at all

Every accelerator has the same three fixed budgets: <b>die area, power, and the HBM interface</b>. A GPU spends its budget on generality: thousands of thread contexts, register files, hardware schedulers, caches, and number formats some customer somewhere needs. That is insurance against unknown workloads. An ASIC asks what LLM inference actually binds on (bytes moved, on-chip reuse, chip-to-chip traffic), drops the insurance, and spends the budget there. ASICs like Trainium follow this approach to optimize performance per dollar of total cost of ownership (TCO) across large-scale training and inference workloads, with cost per token as the key metric for production inference.

Think of a kitchen equipped for any cuisine against one built only for ramen. The ramen kitchen does not have a better chef. It has no idle equipment, and every square metre goes to the steps that slow ramen down.

What the reallocation buys, beyond cost:

- <b>Communication that does not steal compute.</b> On GPUs, NCCL and RCCL run collectives by default as kernels on the same SMs or CUs as the math. DeepSeek-V3 gave 20 of 132 H800 SMs (15.2%) to all-to-all, and asked hardware vendors to offload communication from the SMs. Trainium3 ships dedicated collective cores (CC-Cores) on every chip.
- <b>Software-managed on-chip memory for reuse.</b> Trainium3 has 8 × 32 MiB = 256 MiB of SBUF per chip, placed by the compiler or kernel rather than by a hardware cache.
- <b>Larger scale-up domains</b> (chips joined by the scale-up interconnect, before traffic leaves for the scale-out network). The larger the domain, the more of the tensor- and expert-parallel traffic that runs every layer stays on links with far more bandwidth per chip than the scale-out network. <b>Trn3 UltraServer:</b> 144 Trainium3 chips over NeuronLink-v4 through NeuronSwitch-v1. <b>Jalapeño</b>: 128 chips per rack over a local collective network, 2,048 per Ethernet pod.
- <b>More tokens per watt.</b> Jalapeño has 15.4 TB/s of peak HBM bandwidth on a 700 W chip power budget, 22.0 GB/s per watt. Trainium follows the same path: the Trn3 UltraServer delivers 4x the energy efficiency of the Trn2 UltraServer.

### The system is the building block

A large model runs across many chips, so the unit to reason about is the system that serves it.

|  | AWS Trn3 Gen2 UltraServer |
| --- | --- |
| Chips | 144 Trainium3 (36 servers × 4) |
| FP8 dense | 362.448 PF |
| HBM capacity (nominal) | 20.7 TB |
| HBM bandwidth | 705.6 TB/s |
| Scale-up | NeuronLink-v4, 2,560 GB/s per chip (both directions) |

Trainium3 runs FP8, MXFP8, and MXFP4 at 2.517 PF, and FP4 pays off where decode is limited: bytes. MXFP4 stores 4.25 bits per weight, one shared 8-bit scale per 32 values included, so the running example's weights shrink from 70.55 GB to 37.48 GB. Every decode step reads 53.125% of the bytes, and the freed HBM holds more KV cache, which means more conversations per chip.

<b>Efficiency, same kernel workflow.</b> A new operator becomes a Triton or CUDA kernel on GPUs and an NKI kernel on Trainium, often starting from the open-source NKI Library. Anthropic's Project Rainier runs on nearly 500,000 Trainium2 chips.

<div class="callout callout-gray" markdown="1">
<span class="callout-icon">🎯</span>

<b>Interview Lens:</b> <i>How do you compare accelerators built from chips of different sizes?</i> Normalize to the system that serves the model: compute, HBM capacity, and HBM bandwidth, then divide by dollars or watts. Chip count is a packaging choice: 144 Trainium3 chips add up to 362.448 PF of FP8, 20.7 TB of HBM, and 705.6 TB/s in one Trn3 Gen2 UltraServer.

</div>

---

## 3. Prefill vs decode: why one request stresses hardware two ways

<b>Prefill</b> reads the whole prompt at once: lots of math, very parallel. <b>Decode</b> writes one token at a time: very little math per step, but every step re-reads the weights (only the active experts in a MoE model) and each sequence's KV cache.

<figure><img src="/assets/img/flops-are-not-tokens/prefill-decode-v3.png" alt="prefill decode v3" loading="lazy"></figure>

<b>Deep dive: tensor shapes.</b>

```txt
PREFILL (S prompt tokens)                 DECODE (1 new token per step)
Q:     [B, H,    S, D]                     Q:        [B, H,    1, D]
K, V:  [B, H_kv, S, D]                     K cache:  [B, H_kv, S, D]   read every step
MLP:   [B*S, d_model] x [d_model, d_ff]    V cache:  [B, H_kv, S, D]   read every step
       → fat matrix multiplies             MLP:      [B, d_model] x [d_model, d_ff]
                                                     → skinny, nearly matrix-vector
```

Changing <b>S to 1</b> is the whole story: the same weight matrix, multiplied by B rows instead of B × S rows.

<b>The roofline.</b> Arithmetic intensity is FLOPs per byte moved from memory. Every chip has a <b>ridge point</b>, peak FLOPs divided by peak bandwidth. Below the ridge the kernel is <b>memory-bound</b>: the math units wait for bytes and extra FLOPs do nothing. Above it, the kernel is <b>compute-bound</b>.

For FP8 decode, the weight reads alone give an arithmetic intensity of <b>2 × batch</b> FLOPs per byte (each weight byte does one multiply-add per sequence in the batch). For prefill it is 2 × batch × prompt length, far above any ridge.

<figure><img src="/assets/img/flops-are-not-tokens/roofline-trainium-gens.svg" alt="Roofline per system for the Trn2, Trn3 Gen1, and Trn3 Gen2 UltraServers, from published peak specifications." loading="lazy"><figcaption>Roofline per system for the Trn2, Trn3 Gen1, and Trn3 Gen2 UltraServers, from published peak specifications.</figcaption></figure>

<b>How the roofline moves from Trainium2 to Trainium3.</b> The Trn2 and Trn3 Gen1 UltraServers both hold 64 chips, so comparing them separates the chip generation from the system size. Dense 8-bit compute grows 1.936x and HBM bandwidth 1.690x, so the ridge point moves from 448.28 to 513.67 FLOP/B, and decode becomes compute-bound at batch 257 instead of 225 (weights only). Scaling Trn3 from 64 to 144 chips raises both ceilings by the same factor and leaves the ridge point at 513.67 FLOP/B. Every decode marker sits left of all three ridge points, so for decode the bandwidth increase is what raises the ceiling. More compute raises the flat roof; more bandwidth raises the slanted one.

<figure><img src="/assets/img/flops-are-not-tokens/trainium-gens-scaling.svg" alt="Peak compute, HBM bandwidth, and HBM capacity of each UltraServer relative to Trn2, from published specifications." loading="lazy"><figcaption>Peak compute, HBM bandwidth, and HBM capacity of each UltraServer relative to Trn2, from published specifications.</figcaption></figure>

<b>Latency is two numbers, throughput a third.</b> <b>TTFT</b> (time to first token) is driven by prefill plus queueing. <b>TPOT</b> (time per output token) is driven by decode. <b>Throughput</b>, in output tokens or requests completed per second, is measured at a given concurrency: more concurrent requests raise throughput until TTFT or TPOT misses its target. A chip can be great at one and mediocre at another, which is the root reason for prefill/decode disaggregation (Section 11).

### The running example

Every number from here on uses Meta Llama 3.3 70B (80 layers, 8 KV heads, head dim 128, 70.55B parameters), so 70.55 GB of FP8 weights and 160 KiB of FP8 KV cache per token, at an 8K-in / 1K-out shape. AWS Trainium figures are for Trainium3 and the 144-chip Trn3 Gen2 UltraServer.

Minimum time per token = 70.55 GB of weights ÷ HBM bandwidth. No kernel can beat it.

|  | AWS Trn3 Gen2 UltraServer |
| --- | --- |
| One copy of the weights | One Trn3 server: 4 chips, 576 GB of HBM, 19.6 TB/s |
| Minimum time per token | 3.60 ms |
| Max tokens/s for one user | 277.8 |
| Ridge point | 514 FLOP/B |
| Batch at which decode becomes compute-bound | 257 |
| Copies of the weights per UltraServer | 36 |

Each four-chip Trn3 server holds one copy of the weights, the same unit Section 4 uses. In principle, a large enough batch lifts decode past the ridge. In practice, the KV cache stops decode short of it, as the next section shows.

<div class="callout callout-gray" markdown="1">
<span class="callout-icon">🎯</span>

<b>Interview Lens:</b> <i>Why does doubling FLOPS not double tokens per second?</i> Decode sits on the sloped part of the roofline. Doubling FLOPS raises a ceiling decode never reaches. Doubling bandwidth moves the slope decode is actually on.

</div>

---

## 4. The KV cache: the capacity wall

Every token a user has sent or received leaves a record in memory, the KV cache. Batching is a carpool: one read of the weights serves every passenger. But every passenger brings luggage, its own KV cache, and the luggage does not shrink when you add riders. Add enough riders and you are mostly moving luggage.

```txt
KV bytes per token = 2 (K and V) × 80 layers × 8 KV heads × 128 head dim × 1 byte (FP8)
                   = 163,840 bytes = 160 KiB per token
4K-token sequence   → 0.671 GB
32K-token sequence  → 5.37 GB
128K-token sequence → 21.47 GB
```

With batch B, weight bytes are shared but KV bytes grow with B, so arithmetic intensity stops rising and converges to a per-sequence limit:

```txt
weights and KV traffic only:
arithmetic intensity(B)  =  (2P + attention FLOPs) × B  /  (P × bytes_per_param  +  B × KV_bytes_per_seq)
B → ∞:        →  (2P + attention FLOPs)  /  KV_bytes_per_seq
```

| Context | KV per sequence | Max decode arithmetic intensity at any batch |
| --- | --- | --- |
| 8K | 1.34 GB | 121.1 FLOP/B |
| 32K | 5.37 GB | 42.3 FLOP/B |
| 128K | 21.47 GB | 22.6 FLOP/B |
| Limit (attention only, GQA-8, FP8 KV) | Grows with context | 16 FLOP/B |

Every value sits below the Trainium3 ridge point (514 FLOP/B). <b>At 8K context and beyond, with FP8 weights and KV cache, this model's decode is memory-bound on Trainium3 at any batch size.</b>

### How many conversations fit

FP8 stores each of Llama 3.3 70B's 70.55 billion parameters in 8 bits, one byte, so the weights take 70.55 GB. The rest of the HBM holds the KV cache, minus the activation and runtime buffers the serving engine reserves, so the counts below are upper bounds. Decode activations themselves are small: one MLP intermediate for 753 sequences is 753 × 28,672 × 2 bytes, 43.2 MB.

|  | AWS Trn3 Gen2 UltraServer |
| --- | --- |
| HBM | 20.7 TB |
| HBM bandwidth | 705.6 TB/s |
| HBM left for KV cache (36 copies of the weights) | 18,196.1 GB |
| 4K-token sequences that fit | 27,108 |
| 32K-token sequences that fit | 3,384 |
| Arithmetic intensity at the 4K max batch | 198.55 FLOP/B |
| Share of the ridge point | 38.65% |

Each 576 GB of HBM, one Trn3 server, holds one copy of the weights and 753 conversations at 4K. The UltraServer holds 36 such copies and reads all of their KV cache at 705.6 TB/s.

<figure><img src="/assets/img/flops-are-not-tokens/intensity-vs-batch.svg" alt="Decode arithmetic intensity against batch on one Trn3 server, for 4K, 8K, 32K, and 128K contexts. Each curve flattens toward its ceiling as KV traffic grows with the batch, and every ceiling sits below the 513.67 FLOP/B ridge." loading="lazy"><figcaption>Decode arithmetic intensity against batch on one Trn3 server, for 4K, 8K, 32K, and 128K contexts. Each curve flattens toward its ceiling as KV traffic grows with the batch, and every ceiling sits below the 513.67 FLOP/B ridge.</figcaption></figure>

<b>The key insight.</b> Even with HBM packed full of conversations, decode runs at 38.65% of the ridge point or less. KV traffic grows with every sequence you add. That is why, for this model, memory bandwidth, HBM capacity, and KV cache design set decode speed, while extra peak FLOPS barely move it. Three things help: <b>more bandwidth per dollar or per watt</b>, <b>more HBM capacity, and a different attention design such as MLA or KV quantization.</b>

<div class="callout callout-gray" markdown="1">
<span class="callout-icon">🎯</span>

<b>Interview Lens:</b> <i>Why can no chip make long-context GQA decode compute-bound?</i> Every sequence brings its own KV bytes, so arithmetic intensity converges to a per-sequence limit far below any ridge point. Raising that limit takes fewer KV bytes per token, as with MLA or KV quantization, or more queries per KV read, as with speculative decoding.

</div>

---

## 5. Energy: why inference ASICs exist at all

Math is cheap; moving data is expensive. In a data center with a fixed power budget, the chip that spends less energy moving bytes serves more tokens.

```txt
Energy per operation (45 nm process):
  32-bit read from off-chip DRAM  640 pJ
  32-bit floating-point multiply  3.7 pJ      → moving the data costs 173x the math

Bytes read per decode token (70.55 GB of FP8 weights):
  batch 1:   70.55 GB per token, 2 FLOPs per byte
  batch 64:  70.55 GB ÷ 64 = 1.10 GB per token
```

The 45 nm figures show the size of the gap between moving data and computing on it, not modern absolute values. At batch 1, decode does two FLOPs for every byte it pulls from HBM, so almost all of its energy goes to moving weights. Batching divides the weight bytes per token, which makes <b>batching an energy optimization, not only a throughput one</b>.

Three levers cut energy per token: keep more data on chip, move fewer bits per value, and spend silicon and power on data movement rather than on flexibility decode does not use.

### Across the chips: energy strategy

|  | NVIDIA | AMD | AWS Trainium | OpenAI Jalapeño |
| --- | --- | --- | --- | --- |
| Main lever | Lower-precision formats such as NVFP4: fewer bytes per weight | Advanced process nodes and chiplets: more math per watt | Explicitly scheduled SRAM reuse (compiler or NKI); collectives that move data SBUF to SBUF | Bandwidth first: 15.4 TB/s at 700 W |

<b>Reading the table.</b> Jalapeño has 22.0 GB/s of peak HBM bandwidth per watt of chip power. Trn3 UltraServers deliver over 5x the output tokens per megawatt of Trn2 UltraServers at similar latency per user.

<div class="callout callout-gray" markdown="1">
<span class="callout-icon">🎯</span>

<b>Interview Lens:</b> <i>How does an inference ASIC lower cost per token?</i> Decode is limited by bytes moved, and moving bytes is where the energy goes. A chip that moves bytes more cheaply serves more tokens per watt, and a power-capped data center converts tokens per watt directly into throughput.

</div>

---

## 6. Tile quantization: the hidden tax on small batches

Matrix units work on fixed-size blocks. If the block has room for 128 rows and you bring 8, you still pay for 128 and get 8. This is a simplified padding model: real utilization also depends on operand orientation, instruction granularity, and how the kernel maps the GEMM onto the hardware.

<figure><img src="/assets/img/flops-are-not-tokens/tile-quantization.png" alt="tile quantization" loading="lazy"></figure>

<figure><img src="/assets/img/flops-are-not-tokens/d4_tile_quantization.png" alt="Tile quantization: decode batch 8 on a 128-row tile does 6.25% useful work; batch 129 drops back to 50.39% because one extra row costs a whole tile." loading="lazy"><figcaption>Tile quantization: decode batch 8 on a 128-row tile does 6.25% useful work; batch 129 drops back to 50.39% because one extra row costs a whole tile.</figcaption></figure>

```txt
Useful fraction = rows ÷ (ceil(rows ÷ tile) × tile)
batch 8,   tile 128 → 6.25%
batch 128, tile 128 → 100%
batch 129, tile 128 → 50.39%   (one extra row costs a whole tile)
```

Prefill fills tiles easily (batch × prompt rows). Decode at small batch wastes most of each tile, on top of being memory-bound. Utilization counters can look healthy while most of the work is padding. At small batch the bandwidth limit binds first; tile quantization starts to cost real throughput as batch grows toward the ridge.

### Across the chips: matrix unit granularity

|  | NVIDIA | AMD | AWS Trainium | OpenAI Jalapeño |
| --- | --- | --- | --- | --- |
| Matrix unit | Tensor Core MMA with M of 64 or 128 per thread block, 256 per CTA pair (Blackwell) | MFMA on 16- or 32-row blocks per wavefront | 128×128 systolic array per NeuronCore | Not disclosed |
| Small-batch handling | Many small tiles run concurrently across SMs; split-K and stream-K | 16-row MFMA tiles; split-K and stream-K in hipBLASLt | Stationary operand loaded into the array, moving operand streamed; dynamic-shape ISA support can reduce padded shape buckets | Not disclosed |

<div class="callout callout-green" markdown="1">
<span class="callout-icon">🛠️</span>

<b>Takeaway:</b> set batch caps and shape buckets at tile multiples, and use speculative decoding to turn one-row decode into multi-row verification (Section 10).

</div>

---

## 7. Communication and scale-up

Large models span many chips, and chips must talk every layer. The <b>α–β model</b> gives the cost of each message: sending n bytes takes α + n ÷ β, where α is the fixed startup latency every message pays before any bytes flow and β is the link bandwidth. Small messages are dominated by α, large messages by n ÷ β.

<figure><img src="/assets/img/flops-are-not-tokens/scale-hierarchy-v4.png" alt="scale hierarchy v4" loading="lazy"></figure>

```txt
Message time = α (startup latency) + size ÷ β (bandwidth)

Tensor-parallel decode, running example on 8 chips, 32 sequences, BF16 activations:
  one all-reduce = 32 × 8,192 × 2 bytes = 524,288 bytes   → time dominated by α
  2 all-reduces per layer × 80 layers = 160 per decode step
  → every generated token pays the fabric's startup latency 160 times
```

More link bandwidth barely helps here; lower latency does. That is why serving stacks ship one-shot all-reduce kernels for small messages, and why vendors push reductions into the fabric.

Each kind of parallelism puts a different load on the fabric:

| Parallelism | What is split | Traffic | Where it fits |
| --- | --- | --- | --- |
| Tensor (TP) | Each layer's matrices | Two all-reduces per layer, small and latency-bound | Inside the scale-up domain |
| Expert (EP) | MoE experts | Two all-to-alls per MoE layer | Inside the scale-up domain where possible |
| Pipeline (PP) | Groups of layers | Activations between stages, once per stage | Scale-up or scale-out |
| Data (DP) | Whole model replicas | None during serving | Anywhere |

Domain size sets how far expert parallelism can spread on the fast fabric. Collective startup latency, topology, expert placement, routing imbalance, and overlap with compute set how well it runs there.

### Across the chips: scale-up

| Platform | Fast domain | Link per chip | Who runs collectives | Beyond the domain |
| --- | --- | --- | --- | --- |
| NVIDIA NVL72 systems | One rack of GPUs | NVLink | NCCL on the SMs; in-switch reduction | RDMA |
| AMD MI355X | One node of GPUs | Infinity Fabric | RCCL on the CUs | RDMA over Ethernet |
| AWS Trn3 UltraServer (Gen2) | 144 chips: 36 servers × 4 Trainium3 chips, across racks | NeuronLink-v4 through NeuronSwitch-v1: 2,560 GB/s per chip (both directions): 1,024 GB/s within a server, 1,280 GB/s across servers in a rack, 256 GB/s across racks | Neuron collectives on dedicated CC-Cores, SBUF to SBUF | EFA, 57,600 Gb/s per UltraServer |
| OpenAI Jalapeño | 128 chips per rack; 2,048 per pod at 200 GB/s per chip | Ethernet, 600 GB/s per chip within a rack | Not disclosed | Broadcom Tomahawk 6 Ethernet across racks |

<b>The design bets:</b>

- <b>NVIDIA</b> joins a rack of GPUs over NVLink, and the switches perform part of each reduction themselves, so collectives spend less time on the GPUs.
- <b>AMD</b> joins the GPUs in a node over Infinity Fabric, and expert parallelism beyond the node runs over the RDMA Ethernet scale-out network.
- <b>Trainium3 UltraServer</b> keeps 144 chips in one NeuronLink-v4 domain through NeuronSwitch-v1: 2,560 GB/s per chip, 1,280 GB/s of it across servers. Expert parallelism stays on NeuronLink without touching the scale-out network, and the GPT-OSS 120B deployment below keeps it inside one instance. 16 dedicated CC-Cores per chip run the collectives, so communication does not compete with the compute engines.
- <b>Jalapeño</b> keeps 128 chips in a local domain and joins 16 racks into a 2,048-chip Ethernet pod. Its links run at 600 GB/s per chip, and its HBM delivers 15.4 TB/s at 700 W (Section 5).

### Design: GPT-OSS 120B expert parallelism on Trn3

GPT-OSS 120B is a MoE model with 36 layers, 128 experts per layer, and 4 experts active per token on a 2,880-wide residual stream: 116.83B parameters, 5.13B active. Its MoE weights are stored in MXFP4 at 4.25 bits per parameter, and the checkpoint is 60.8 GiB. vLLM Neuron ships a tested Trn3 deployment for it at 8K in / 1K out, and the deployment gives each pool the parallelism that fits its stage:

| Pool | Parallelism | Placement |
| --- | --- | --- |
| Decode | TP8 × DP16 = 128 ranks; EP128, one expert per rank | One Trn3 instance (16 Trainium3 chips, 128 NeuronCores), one logical core per physical core |
| Prefill | Six replicas, each TP4 with EP2 | One Trn3 instance, 24 of its 64 logical NeuronCores, two physical cores per logical core |
| Between pools | KV cache handed from prefill to decode | NIXL over EFA |

<b>Decode: one expert per core.</b> With 128 experts and 128 ranks, each NeuronCore holds one expert of every layer, and the expert all-to-all stays inside one instance on NeuronLink. One NKI megakernel runs all 36 decoder layers of a decode step, one kernel call per step instead of one per layer, and the KV cache is FP8. Each of the 16 data-parallel ranks takes 8 sequences, 128 concurrent requests per decode instance; the cap of 8 is 128 × TP ÷ EP.

<b>Prefill: small replicas.</b> Each prefill replica is its own TP4 server with EP2, because expert parallelism cannot share data-parallel ranks on the prefill graph.

<b>Scaling out: whole instances.</b> More traffic adds instances, not wider expert groups. For production, the tutorial recommends six prefill replicas for each decode instance and 16 replicas on each prefill instance. Three prefill instances and eight decode instances then use every NeuronCore: 48 prefill replicas and 8 decode servers on 11 instances.

<b>What the GPU design does differently.</b> A GPU rack-scale NVLink domain is one flat tier: every GPU reaches every other through the switches at the same bandwidth, so expert placement inside it needs no tiering, and the all-to-all runs by default on the same SMs as the math (Section 2). On a Trn3 UltraServer the tiers make placement a design decision, and the CC-Cores run the all-to-all while the NeuronCores compute.

<div class="callout callout-gray" markdown="1">
<span class="callout-icon">🎯</span>

<b>Interview Lens:</b> <i>Why does the GPT-OSS 120B decode server run exactly 128 ranks?</i> The model has 128 experts per layer. At EP128 each NeuronCore holds one expert of every layer, so expert weights split evenly and the all-to-all stays inside one instance. Throughput then scales by adding instances, not by widening the expert group.

</div>

<div class="pullquote" markdown="1">
Interconnect bandwidth can matter more than another few hundred TFLOPS.
</div>

---

## 8. Software: the part that often decides real performance

A chip is only as fast as the code that runs on it. An excellent chip with a weak compiler loses to weaker hardware with a mature stack.

<figure><img src="/assets/img/flops-are-not-tokens/software-stacks-v3.png" alt="software stacks v3" loading="lazy"></figure>

<b>Compilers and kernels.</b> Both platforms pair a compiler with hand-written kernels. A compiler fuses most of the graph (torch.compile on GPUs, the Neuron compiler on Trainium), and performance-critical operators are hand-tuned kernels (CUDA or Triton on GPUs, NKI on Trainium). The difference is scheduling: the Neuron compiler also plans every DMA transfer and engine instruction stream, work GPU hardware does at run time:

```txt
Model graph → compiler IR → fusion → tiling → SRAM placement → engine scheduling → accelerator instructions
```

Between NVIDIA and AMD the names differ but the concepts map closely; porting usually fails on tile shapes and wave-size assumptions, not syntax.

<b>The real jump is GPU to ASIC:</b> Trainium replaces warps and the hardware warp scheduler with compiled instruction streams: each engine runs its own stream asynchronously, and the compiler inserts semaphore synchronization between them. Occupancy gives way to overlap: the instruction streams must keep DMA transfers and the Tensor, Vector, Scalar, and GpSimd engines busy at the same time. Coalescing survives as DMA efficiency: a transfer needs at least 2 KiB of contiguous data per partition to approach peak bandwidth, and one large DMA beats many small ones. On top of that, you reason about SBUF partition layout and which operand stays stationary in the systolic array.

<b>Custom kernels exist on both sides.</b> On Trainium they are written in NKI (Neuron Kernel Interface), a bare-metal kernel language, written in Python, that exposes the NeuronCore ISA directly through <code>nki.isa</code>. AWS publishes the NKI Library, an open-source set of pre-built kernels optimized for Neuron devices, and NKI kernels drop into PyTorch and JAX models as custom operators. The difference from CUDA is what you control: no warps to schedule, but explicit tiles, SRAM placement, and DMA transfers.

### Across the chips: the kernel level

| Concept | NVIDIA | AMD | AWS Trainium | OpenAI Jalapeño |
| --- | --- | --- | --- | --- |
| Write a fused kernel in | CUDA, Triton, CUTLASS/CuTe | HIP, Triton, Composable Kernel | NKI (Python, direct ISA access) or the open-source NKI Library; Neuron compiler for the rest of the graph | Internal only |
| Execution unit | SM, warp of 32 threads | CU, wavefront of 64 | NeuronCore-v4: Tensor, Vector, Scalar, GpSimd engines. No warps | Not disclosed |
| Programmer SRAM | Shared memory | LDS | SBUF: 32 MiB per core, 128 partitions | Not disclosed |
| Accumulators | Registers; Tensor Memory on Blackwell | VGPRs / AGPRs | PSUM: 2 MiB per core | Not disclosed |
| HBM to SRAM movement | TMA, async copies | Global / buffer loads | 128 DMA engines, explicit in NKI, scheduled by the compiler | Not disclosed |
| Latency hiding | Hardware swaps warps | Hardware swaps wavefronts | Overlapped engine instruction streams: double-buffered tiles, semaphores between engines | Not disclosed |
| Profile with | Nsight Systems / Compute | rocprof, ROCm Compute Profiler | Neuron Explorer (neuron-profile) | Internal |
| Ecosystem | CUDA libraries and community kernels | ROCm, Composable Kernel, and upstream framework support | Open source: NKI Library, TorchNeuron, vLLM Neuron | Internal |

---

## 9. Optimization is bottleneck removal

Every optimization removes one specific bottleneck. Diagnose first, then pick.

| Symptom | Bottleneck | First fixes |
| --- | --- | --- |
| HBM bandwidth near peak, matrix units idle | Memory bandwidth (memory-bound) | Weight and KV quantization, fusion, larger batch |
| Matrix units saturated | Compute (compute-bound) | Lower precision, better tile shapes |
| Healthy utilization counters, low useful work | Tile quantization | Shape buckets at tile multiples, speculative decoding |
| Many tiny kernels, gaps on the timeline | Kernel launch overhead | Graph capture or static compilation, fusion |
| Chips waiting on each other | Inter-chip communication overhead | Lower tensor-parallel degree, one-shot all-reduce, overlap |
| Batch cannot grow, preemptions | HBM capacity (KV cache) | Paged KV, KV quantization, offload, more chips |
| Good kernels, bad latency | Queueing delay | Lower utilization target, PD disaggregation |

### Across the chips: same principle, different implementation

| Optimization | NVIDIA | AMD | AWS Trainium | OpenAI Jalapeño |
| --- | --- | --- | --- | --- |
| Continuous batching | vLLM, SGLang, TRT-LLM | vLLM, SGLang | vLLM Neuron; SGLang Neuron | Internal |
| Kernel fusion | Hand-written Triton/CUDA, torch.compile | Triton, Composable Kernel | Neuron compiler plus NKI kernels | Internal |
| FlashAttention-style tiling | FA3 / FA4-class kernels | Ported Triton and CK kernels | NKI flash attention kernel, SBUF-resident tiles | Internal |
| Quantization | FP8, MXFP8, MXFP6, MXFP4, NVFP4 | FP8, MXFP8, MXFP6, MXFP4 | FP8, MXFP8, MXFP4 | MXFP4 |
| Launch overhead fix | CUDA Graphs | HIP Graphs | Whole graph compiled per shape bucket and warmed up before serving | Not disclosed |
| Paged KV | PagedAttention | PagedAttention | Paged KV cache in fixed-size blocks | Not disclosed |
| Speculative decoding | EAGLE, MTP heads | EAGLE, MTP heads | EAGLE3 in vLLM Neuron | Not disclosed |
| PD disaggregation | Dynamo, vLLM, SGLang, llm-d | vLLM, SGLang, llm-d | vLLM Neuron, SGLang Neuron | Not disclosed |

<div class="callout callout-green" markdown="1">
<span class="callout-icon">🛠️</span>

<b>Takeaway:</b> the <i>principle</i> of an optimization transfers across hardware. The <i>implementation</i> often does not. Port the principle, then re-derive the implementation.

</div>

---

## 10. Five workload scenarios

<b>A. Large-batch prefill.</b> Compute-bound. Achievable dense low-precision FLOPS and long-sequence attention kernels decide. Dense FLOPS per dollar and per system set the cost.

<b>B. Small-batch, latency-first decode.</b> Memory-bound, with kernel launch overhead on top. HBM bandwidth, quantization, static graphs, and speculative decoding decide. Look at measured TPOT, not specs.

<b>C. Long context.</b> KV capacity and KV bandwidth decide (Section 4).

<b>D. Mixture of Experts.</b> Each layer adds two all-to-alls, and the busiest device sets the pace. A large scale-up domain keeps that traffic off the scale-out network: the GPT-OSS 120B deployment in Section 7 keeps it inside one Trn3 instance, and Jalapeño inside a 128-chip local domain.

<figure><img src="/assets/img/flops-are-not-tokens/moe-layer-v2.png" alt="moe layer v2" loading="lazy"></figure>

<b>E. Speculative decoding.</b> A small draft proposes K tokens and the target model verifies them in one pass, turning 1-row decode into (K+1)-row decode: higher arithmetic intensity, fuller tiles. If each draft token is accepted with probability p, one verification step emits (1 − p^(K+1)) ÷ (1 − p) tokens on average, so the gain grows with p and with how cheap the draft is. Gains shrink at high batch because batching already used the spare compute, so the best K falls as batch size rises.

### The five scenarios on a Trn3 UltraServer

| Scenario | AWS Trn3 Gen2 UltraServer |
| --- | --- |
| A. Prefill | 362.448 PF of dense FP8 |
| B. Low-latency decode | 3.60 ms minimum per token (Section 3) |
| C. Long context | 20.7 TB of HBM; 3,384 conversations at 32K |
| D. MoE | One 144-chip NeuronLink-v4 domain; collectives on dedicated CC-Cores |
| E. Speculative decoding | EAGLE3 in vLLM Neuron |

<div class="callout callout-gray" markdown="1">
<span class="callout-icon">🎯</span>

<b>Interview Lens:</b> <i>Why does MoE depend on the interconnect more than dense models do?</i> Two all-to-alls per layer, each costing startup latency plus bytes over bandwidth, and step time is the maximum over devices. Imbalance and fabric latency both add straight to TPOT.

</div>

---

## 11. Prefill-Decode disaggregation

Prefill wants compute. Decode wants bandwidth, KV capacity, and energy efficiency. Running both on the same chip makes each stall the other. Split them into two pools, and let each pool use the hardware that fits its job.

<figure><img src="/assets/img/flops-are-not-tokens/diagram4-pd-disaggregation.png" alt="diagram4 pd disaggregation" loading="lazy"></figure>

TTFT and TPOT get separate targets and separate tuning. And each pool can be sized for its own stage: FLOPS per dollar for prefill, HBM bandwidth per dollar for decode.

<b>The cost: moving the KV cache.</b>

```txt
KV transfer bytes/s = request rate × prompt tokens × KV bytes per token
Per request, running example: 8,192 prompt tokens × 160 KiB = 1.34 GB moved from prefill to decode
```

Total transfer is that figure times your request rate. Across <b>different</b> chip types, the KV layout and precision must also match or be converted, which is why both pools usually run on one platform.

<b>When it hurts:</b> short prompts or low load; small models that batch well on one chip; slow links between pools; a prompt:output mix that swings without autoscaling.

---

## 12. Capacity planning: sizing each pool from demand

Prefill and decode form a two-stage pipeline: every request passes through prefill, then decode. A pipeline serves requests only as fast as its slower stage, so adding prefill nodes while decode is the bottleneck adds cost without adding throughput. Each pool is sized from the traffic it receives, not split 50/50. Two results from queueing theory do the sizing.

<b>Little's law.</b> In any stable system, the average number of items inside equals their arrival rate times the average time each one stays: L = λ × W. In decode the items are sequences, and each stays for O × TPOT, so sequences in flight = λ × O × TPOT. Every one of them holds its KV cache in HBM the whole time, which is what sets decode memory.

<b>Utilization and waiting.</b> When requests arrive at random and a single server handles them in turn (an M/M/1 queue), average waiting time grows as ρ ÷ (1 − ρ) times the service time, where ρ is utilization, the fraction of time a node is busy. Waiting rises without bound as ρ approaches 1, so every pool runs with ρ below 1.

```txt
Inputs
  λ      requests per second           P, O   average prompt and output tokens
  μ_p    prefill tokens/s per node at the TTFT target   (measure on your hardware)
  μ_d    decode tokens/s per node at the TPOT target    (measure on your hardware)
  ρ      target utilization, kept below 1 because waiting time explodes near 1

Prefill nodes   N_p = ceil( λ·P ÷ (ρ·μ_p) )
Decode nodes    N_d = ceil( λ·O ÷ (ρ·μ_d) )

Memory check (Little's law)
  sequences in flight  = λ · O · TPOT
  KV needed            = sequences × (P + O/2) × KV bytes per token
Link check             λ · P · KV bytes per token ≤ link bandwidth
```

<figure><img src="/assets/img/flops-are-not-tokens/capacity-planning-v4.png" alt="capacity planning v4" loading="lazy"></figure>

<figure class="embed"><iframe class="embed-frame" data-embed="fleet-sizer" src="/assets/embeds/fleet-sizer.html" title="Prefill/Decode Fleet Sizer" loading="lazy"></iframe></figure>


<b>The shape of the traffic sets the fleet.</b> Two published InferenceX sequence shapes, and real multi-turn chat traffic:

| Sequence shape | Prefill tokens (P) | Decode tokens (O) | Decode : prefill tokens |
| --- | --- | --- | --- |
| 8K in / 1K out (RAG, long documents) | 8,192 | 1,024 | 1 : 8 |
| 1K in / 8K out (reasoning, agents) | 1,024 | 8,192 | 8 : 1 |
| Multi-turn chat (LMSYS-Chat-1M, per turn, 2.0 turns per conversation) | 69.5 new tokens | 214.5 | 3.09 : 1 |

N\_p scales with λ·P and N\_d with λ·O, so moving from the first shape to the second shifts the decode-to-prefill split by 64x at the same request rate, before any hardware choice. Same model, completely different fleet. Multi-turn chat sits in between, and prefix caching keeps its P at the new tokens of each turn; without it, every turn re-prefills the whole conversation.

### Sizing the decode pool by bandwidth

Decode is memory-bound (Sections 3 and 4), so to first order <b>decode throughput scales with HBM bandwidth, and decode cost with the price of that bandwidth</b>. That gives a one-line rule, per system, the unit you buy:

```txt
decode cost per token  ∝  system cost (or power) ÷ delivered HBM bandwidth
```

For a Trn3 Gen2 UltraServer the denominator is 705.6 TB/s per system. In production, measured decode throughput replaces peak bandwidth, since kernels, MoE routing, and software shift it (Section 13).

<b>Sizing the decode pool.</b> In the 1K in / 8K out shape, decode tokens outnumber prefill tokens 8 to 1, so the decode pool is most of the fleet. Three numbers size it:

- <b>HBM bandwidth per dollar</b> sets cost per output token, because bytes per second generate decode tokens.
- <b>HBM capacity</b> sets concurrent conversations: 753 per Trn3 server at 4K (Section 4).
- <b>Disaggregation</b> (Section 11) lets the pool grow apart from prefill. vLLM Neuron runs prefill and decode as separate Trainium pools and moves the KV cache between them over EFA.

<div class="callout callout-gray" markdown="1">
<span class="callout-icon">🎯</span>

<b>Interview Lens:</b> <i>You doubled prefill nodes and TTFT did not improve. Why?</i> Decode is probably the bottleneck: handed-off requests queue for decode slots. Recompute both pools from λ·P and λ·O, and check decode KV capacity with Little's law.

</div>

---

## 13. What a token costs

Most serving cost is decode, and decode is bound by HBM bandwidth, the scale-up fabric, and software, so <b>$/hour ÷ delivered tokens</b> decides fleet economics.

### ASICs: the principles hold in production

<b>AWS Trainium3 optimizes dollars.</b> Decode cost tracks the price of HBM bandwidth (Section 12), and the Trn3 Gen2 UltraServer delivers 705.6 TB/s of HBM bandwidth per system. Anthropic, Karakuri, Metagenomi, NetoAI, Ricoh, and Splash Music report up to 50% lower training and inference costs on Trainium, Decart reports 4x faster real-time video inference at half the cost of GPUs, and Amazon Bedrock runs up to 3x faster on Trainium3 than on Trainium2.

<b>OpenAI Jalapeño optimizes watts.</b> At 22.0 GB/s of HBM bandwidth per watt (Section 5), first principles predict high throughput per watt. OpenAI reports 1.5x to 1.9x higher throughput per watt than its tested GPU baselines on selected inference workloads. Watts matter most when data center power, rather than budget, is the limit.

<b>What favors ASICs.</b> In a MoE model, expert parallelism runs an all-to-all every layer, so the scale-up domain shapes cost per token (Section 7). ASIC systems build that domain large: 144 chips on one NeuronLink-v4 domain in the Trn3 Gen2 UltraServer, 128 chips in Jalapeño's local domain.

<div class="callout callout-green" markdown="1">
<span class="callout-icon">🛠️</span>

<b>Takeaway:</b> cost per token is driven by software (speculative decoding, PD disaggregation, kernel quality), $/chip-hr, HBM bandwidth, the scale-up domain (MoE), FP4 throughput (prefill), and FP8 peak. The relative importance of these factors depends on the workload and its bottleneck, and peak FLOPS is rarely the binding one. Count only tokens served within your TTFT and TPOT targets (goodput): a system that makes more tokens but misses the SLO costs more per useful token.

</div>

<div class="callout callout-gray" markdown="1">
<span class="callout-icon">🎯</span>

<b>Interview Lens:</b> <i>Two systems deliver the same FP8 PFLOPS. Which costs less per token?</i> For decode-heavy serving, the one with more HBM bandwidth per dollar, because decode is memory-bound. For MoE, also the one that keeps all-to-all inside its scale-up domain. Software then decides how much of that bandwidth becomes tokens, through kernel quality and speculative decoding.

</div>

---

## 14. Choosing hardware

### Which spec to compare first

| Your workload | Bottleneck | Compare this first |
| --- | --- | --- |
| Long prompts, short answers (RAG, summarization) | Prefill compute | Achievable FP8 or FP4 FLOPS per dollar |
| Short prompts, long answers (reasoning, agents) | Decode memory bandwidth | HBM bandwidth per dollar and per watt |
| Long context | KV capacity | HBM capacity per system |
| MoE | All-to-all latency, imbalance | Scale-up domain size and collective latency |
| Strict per-user latency | Minimum time per token | Measured TPOT at your batch |
| Research, fast-changing models | Kernel availability | Kernel and framework coverage for your models |
| Large, shifting mix of the above | Pipeline bottleneck | Separate pools sized by capacity planning |
| Existing custom kernels | Porting effort | Kernels to rewrite in the target language: CUDA, HIP, or NKI |

<div class="pullquote" markdown="1">
<b>There is no universally faster accelerator. There is only an accelerator better matched to a particular workload stage, software stack, and deployment objective.</b>
</div>

---

<div class="callout callout-purple" markdown="1">
<span class="callout-icon">🧩</span>

<b>One question to leave you with.</b> On Trainium every decision is explicit, and a good schedule keeps every engine busy. Someone has to choose which matrix stays stationary in a 128×128 systolic array, what lives in 32 MiB of SBUF per core, and when each of 128 DMA engines fires. The Neuron compiler makes those choices automatically, and NKI gives kernel engineers direct control when they want it. So what does one attention layer look like on Trainium, and where does a custom NKI kernel beat the compiler?

</div>

---

## References

Web pages accessed November 2026.

1. S. Williams, A. Waterman, D. Patterson. [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://dl.acm.org/doi/10.1145/1498765.1498785). Communications of the ACM, 2009.
2. R. Hockney. The Communication Challenge for MPP: Intel Paragon and Meiko CS-2. Parallel Computing, 1994.
3. M. Horowitz. Computing's Energy Problem (and What We Can Do About It). ISSCC, 2014.
4. J. Little. A Proof for the Queuing Formula L = λW. Operations Research, 1961.
5. Y. Leviathan, M. Kalman, Y. Matias. [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192). ICML, 2023.
6. DeepSeek-AI. [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437). 2024.
7. L. Zheng et al. [LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset](https://arxiv.org/abs/2309.11998). ICLR, 2024.
8. Open Compute Project. [OCP Microscaling Formats (MX) Specification v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf). 2023.
9. Meta. [Llama 3.3 70B Instruct model configuration](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/blob/main/config.json).
10. SemiAnalysis. [InferenceX: supported accelerators](https://inferencex.semianalysis.com/about).
11. SemiAnalysis. [OpenAI Jalapeño: Better Than Nvidia Blackwell](https://inferencex.semianalysis.com/blog/openai-jalapeno-better-than-nvidia). 2026.
12. SemiAnalysis. [AWS Trainium3 Deep Dive](https://newsletter.semianalysis.com/p/aws-trainium3-deep-dive-a-potential). 2025.
13. NVIDIA. [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/).
14. NVIDIA. [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html).
15. NVIDIA. [Matrix Multiplication Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html).
16. NVIDIA. [CUTLASS: Blackwell architecture](https://www.mintlify.com/NVIDIA/cutlass/architectures/blackwell).
17. PyTorch. [torch.compile](https://docs.pytorch.org/docs/stable/torch.compiler.html).
18. AMD. [ROCm: MI350 Series microarchitecture](https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi350.html).
19. AWS. [Amazon EC2 Trn3 instances](https://aws.amazon.com/ec2/instance-types/trn3/).
20. Amazon. [Trainium3 UltraServers now available](https://press.aboutamazon.com/2025/12/trainium3-ultraservers-now-available-enabling-customers-to-train-and-deploy-ai-models-faster-at-lower-cost). 2025.
21. Amazon. [Project Rainier](https://aboutamazon.com/news/aws/aws-project-rainier-ai-trainium-chips-compute-cluster/). 2025.
22. AWS Neuron. [Trainium3](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium3.html).
23. AWS Neuron. [Trn3 architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trn3-arch.html).
24. AWS Neuron. [Trainium3 architecture guide for NKI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium3_arch.html).
25. AWS Neuron. [Trainium architecture guide for NKI](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/_sources/nki/guides/architecture/trainium_inferentia2_arch.rst).
26. AWS Neuron. [NKI DMA bandwidth guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/deep-dives/nki-dma-bandwidth-guide.html).
27. AWS Neuron. [Neuron Kernel Interface (NKI)](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/_sources/nki/index.rst).
28. AWS Neuron. [Collective communication](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-features/collective-communication.html).
29. AWS Neuron. [Developer tools](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_sources/tools/index.rst).
30. AWS Neuron. [Speculative decoding in vLLM Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_sources/vllm-neuron/docs/design/speculation/speculative_decoding_design.md).
31. AWS Neuron. [Disaggregated inference in vLLM Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_sources/vllm-neuron/docs/design/vllm/disaggregated-inference.md).
32. AWS. [NKI Library](https://github.com/aws-neuron/nki-library). GitHub.
33. AWS. [TorchNeuron](https://github.com/aws-neuron/torch-neuronx). GitHub.
34. vLLM Project. [vLLM Neuron](https://github.com/vllm-project/vllm-neuron). GitHub.
35. vLLM Project. [GPT-OSS 120B 8K/1K disaggregated serving on Trn3](https://github.com/vllm-project/vllm-neuron/blob/main/docs/tutorials/tutorial-gpt-oss-120b-8k1k-trn3.md). GitHub.
36. OpenAI. [gpt-oss-120b and gpt-oss-20b Model Card](https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf). 2025.
37. LMSYS. [SGLang-Neuron v0.1.0: SGLang's Serving Stack on AWS Trainium](https://www.lmsys.org/blog/2026-11-12-sglang-neuron-v0.1.0). 2026.
38. OpenAI. [OpenAI and Broadcom: Jalapeño](https://openai.com/index/openai-broadcom-jalapeno-inference-chip/). 2026.
39. OpenAI. [Jalapeño first results](https://openai.com/index/jalapeno-first-results/). 2026.
40. ServeTheHome. [OpenAI Jalapeño at Hot Chips 2026](https://www.servethehome.com/openai-jalapeno-asic-at-hot-chips-2026/). 2026.
41. Tom's Hardware. [Hot Chips 2026: OpenAI's Jalapeño unpacked](https://www.tomshardware.com/tech-industry/artificial-intelligence/hot-chips-2026-openais-jalapeno-ai-asic-unpacked-accelerator-developed-using-ai-achieves-efficiency-and-throughput-gains-against-power-hungry-blackwell). 2026.
42. Tom's Hardware. [Jalapeño is for OpenAI's internal use](https://www.tomshardware.com/tech-industry/artificial-intelligence/openais-custom-jalapeno-ai-inference-asic-is-for-openais-internal-use-but-company-leaves-the-door-open-to-broader-rollout-firm-says-it-will-have-its-hands-full-with-jalapeno-for-a-good-long-time). 2026.
43. HotHardware. [OpenAI and Broadcom reveal Jalapeño](https://hothardware.com/news/openai-broadcom-reveal-jalapeno-ai-chip). 2026.
44. Tom's Hardware. [Amazon launches Trainium3](https://www.tomshardware.com/tech-industry/artificial-intelligence/amazon-launches-trainium3-ai-accelerator-competing-directly-against-blackwell-ultra-in-fp8-performance-new-trn3-gen2-ultraserver-takes-vertical-scaling-notes-from-nvidias-playbook). 2025.
