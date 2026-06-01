# Memory Systems for Large Language Models

> **A comprehensive research survey and experimental study of persistent memory architectures for LLMs** — covering theoretical foundations, architectural paradigms, production system comparisons, and a novel hybrid prototype.

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Institution](https://img.shields.io/badge/Institution-SPbU-red)](https://spbu.ru)
[![Year](https://img.shields.io/badge/Year-2026-green)]()
[![Benchmark](https://img.shields.io/badge/Benchmark-MemEval--7-orange)]()
[![Systems](https://img.shields.io/badge/Systems%20Surveyed-12-purple)]()

**Slamani Abdelhafid** · Saint Petersburg State University (SPbU), Faculty of Mathematics and Mechanics  

</div>

---

## Overview

Large Language Models achieve remarkable in-session performance yet share a fundamental architectural limitation: **they forget everything outside their context window**. This project calls this the *ephemeral intelligence problem* and systematically investigates how persistent memory architectures solve it.

The work spans theory, engineering, and experimentation:

- **Formal taxonomy** of 3 memory types × 6 atomic operations
- **Comparative survey** of 12 major memory systems (2022–2026)
- **5 architectural paradigms** analyzed with formal complexity bounds
- **MAC-C3**, an original hybrid 4-level memory prototype
- **MemEval-7**, a novel 7-dimensional benchmark framework
- **Controlled experiments** on 650 conversations with statistical verification

---

## The Ephemeral Intelligence Problem

Current LLMs (GPT-4, Claude 3, Gemini 1.5, LLaMA-3) fail in five concrete modes when memory is absent:

| Failure Mode | Description |
|---|---|
| **Session Amnesia** | No access to prior sessions without explicit re-injection |
| **Personalization Blindness** | Every user treated identically; preferences not retained |
| **Long-Horizon Failure** | Intermediate reasoning steps lost beyond context window |
| **Hallucination Amplification** | Factual gaps filled with statistically plausible but false output |
| **Token Budget Inflation** | Conversation re-injection rapidly exhausts context limits |

> **Why larger context windows are not the solution:**  
> Liu et al. (2023) showed LLMs lose 30–60% recall for tokens in the middle of long inputs (*lost-in-the-middle effect*). Processing a novel per request costs ~$1.28. And even a 1M-token window still erases at session end — cross-session amnesia persists regardless of window size.

---

## Formal Memory Taxonomy

Three memory types, six atomic operations:

```
┌───────────────────────────────────────────────────────────────┐
│  I.  Parametric Memory    — weights θ, implicit, permanent    │
│  II. Contextual Memory    — KV cache, session-only, O(n²)     │
│  III.External Memory      — DB/graphs, persistent, unlimited  │
└───────────────────────────────────────────────────────────────┘

6 Atomic Operations (Du et al., 2025):
  Consolidation · Updating · Indexing · Forgetting · Retrieval · Condensation
```

---

## Five Architectural Paradigms

| # | Paradigm | Key Idea | Limitation |
|---|---|---|---|
| **I** | Flat Retrieval | Single pool, cosine similarity | SNR(t) → 0 without forgetting |
| **II** | Hierarchical Memory | 4 levels, triple-score retrieval | Implementation complexity |
| **III** | Temporal Knowledge Graphs | Bi-temporal edges, hybrid retrieval | Expensive entity extraction |
| **IV** | Memory Compression | Gisting κ=50–100×, KV sparsification | Fidelity loss at high κ |
| **V** | Forgetting Mechanisms | Consolidation-based eviction | Requires tuning of λ, σ |

**Key proposition (4.1):** With consolidation forgetting at rate λ > 0, memory pool converges to stable size N\* = μ/λ with retrieval precision P\* = cλ/μ > 0. Without forgetting, P(t) → 0 as t → ∞.

---

## Systems Surveyed (2022–2026)

### Generation 1 — Establishing the Pattern (2022–2023)
| System | Key Result | Limitation |
|---|---|---|
| **RETRO** | 7.5B params ≈ GPT-3 (175B) via retrieval at pre-training scale | Corpus fixed at training |
| **Generative Agents** | Emergent social behavior in 25-agent virtual town | No forgetting → SNR collapse |

### Generation 2 — System Architectures (2023–2024)
| System | Key Result | Limitation |
|---|---|---|
| **MemGPT** | Outperforms GPT-4 fixed-context on multi-session tasks | Each archival search = extra LLM call |
| **ReadAgent** | Near-perfect long-doc QA at κ=50–100× compression | Document-centric, weak on high-freq updates |

### Generation 3 — Production Scale (2024–2025)
| System | Key Result | Limitation |
|---|---|---|
| **Mem0** | 52K+ GitHub stars, $24M funding; 66.9% LoCoMo | 4.5 pp gap vs full-context |
| **Zep / Graphiti** | +14.8 pp over Mem0 on LongMemEval (63.8% vs 49.0%) | Complex maintenance |
| **A-MEM** | Zettelkasten-style linked memory (NeurIPS 2025) | Not production-ready |

### Generation 4 — OS Paradigm Shift (2025–2026)
| System | Key Result | Limitation |
|---|---|---|
| **MemOS** | First Memory OS; 35.24% token savings vs RAG | Complex infra, periodic fine-tuning |
| **MemMachine** | **93.0% on LongMemEvalS** — current SOTA | Not open-source |
| **TTT-E2E** | **2.7×–35× speedup** at 128K–2M tokens, O(1) latency | Rare token loss risk |

---

## MAC-C3: Proposed Architecture

**MAC-C3** (Memory-Augmented Chatbot, Configuration 3) is a full hybrid system combining:

```
┌───────────────────────────────────────────────────────────────┐
│  1. Conversation Manager   — session mgmt, context assembly   │
│  2. Memory Writer          — LLM extraction, importance τ=0.40│
│  3. Memory Hierarchy       — 4-level with auto-promotion      │
│       L0 Episodic → L1 Working Summary                        │
│       → L2 Semantic Knowledge → L3 Reflective Insights        │
│  4. Hybrid Retrieval       — ChromaDB HNSW + BM25 → RRF       │
│  5. Forgetting Daemon      — async DBSCAN + LLM merge, daily  │
│  6. Response Generator     — GPT-4o, lost-in-middle mitigation│
└───────────────────────────────────────────────────────────────┘
```

**Key hyperparameters:**

| Parameter | Value |
|---|---|
| Base LLM | GPT-4o (T = 0.1) |
| Embedding | text-embedding-3-large → PCA 1536-dim |
| Retrieval top-k | 5 per level per turn |
| Importance threshold τ | 0.40 |
| Never-evict gate σ | ≥ 0.80 |
| DBSCAN ε | 0.15 cosine distance |
| Context injection budget | 2,000 tokens |

**4 ablation configurations tested:**

| Config | Description |
|---|---|
| C0 | No Memory (baseline) |
| C1 | Flat Retrieval |
| C2 | Hierarchical, No Forgetting |
| **C3** | **Full Hybrid MAC ← Best** |

---

## MemEval-7 Benchmark

A novel 7-dimensional framework — the **only benchmark** providing full coverage of all dimensions including token efficiency and forgetting robustness.

**Dataset:** 650 conversations (500 synthetic + 100 human-annotated + 50 adversarial)

| Dim | Name | Metric |
|---|---|---|
| **D1** | Episodic Recall | Recall@5 (F1) |
| **D2** | Semantic Consistency | NLI contradiction score |
| **D3** | Cross-Session Persistence | Binary accuracy + similarity |
| **D4** | Long-Horizon Reasoning | GPT-4o judge (0–10) |
| **D5** | Personalization Index | Expert annotation (0–10) |
| **D6** | Token Efficiency Ratio | ΔAccuracy / ΔTokens (TER) |
| **D7** | Forgetting Robustness | Precision · Recall · F1 |

**Coverage comparison vs existing benchmarks:**

| Benchmark | D1 | D2 | D3 | D4 | D5 | D6 | D7 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| LoCoMo (2024) | ✓ | ∼ | ∼ | ✗ | ✗ | ✗ | ✗ |
| LongMemEval (ICLR 2025) | ✓ | ∼ | ✓ | ✓ | ✗ | ✗ | ✗ |
| MemBench (2025) | ✓ | ✓ | ✓ | ∼ | ∼ | ∼ | ✗ |
| MemoryAgentBench (2025) | ✓ | ✗ | ∼ | ✓ | ✗ | ✗ | ✓ |
| **MemEval-7 (ours)** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

---

## Key Results

### MAC-C3 vs Baseline (C0)

| Metric | Improvement |
|---|---|
| Multi-turn accuracy | **+132%** |
| Cross-session persistence (D3) | **+788%** |
| Personalization (D5) | **+219%** |
| Hallucination rate | **−63%** |
| Accuracy-per-token | **+49%** |

### Full MemEval-7 Scores

| Metric | C0 (None) | C1 (Flat) | C2 (Hier.) | **C3 (Full)** |
|---|:---:|:---:|:---:|:---:|
| D1 Episodic Recall | 0.31 | 0.62 | 0.74 | **0.81** |
| D2 Semantic Consistency | 0.68 | 0.79 | 0.84 | **0.91** |
| D3 Cross-Session | 0.08 | 0.47 | 0.63 | **0.71** |
| D4 Long-Horizon | 0.42 | 0.55 | 0.67 | **0.73** |
| D5 Personalization | 0.21 | 0.44 | 0.59 | **0.67** |
| D6 Token Efficiency | 1.00 | 0.74 | 0.81 | **0.89** |
| D7 Forgetting F1 | N/A | N/A | N/A | **0.78** |
| **★ Average** | 0.34 | 0.57 | 0.71 | **0.79** |

All incremental gains statistically significant (Wilcoxon signed-rank, α = 0.05).

### Forgetting: 30-Day Simulation

| Configuration | Day 1 Pool | Day 30 Pool | Day 30 Recall |
|---|:---:|:---:|:---:|
| C2 (No Forgetting) | 47 | 1,408 | 0.61 ↓ |
| **C3 (Consolidation)** | **47** | **334** | **0.84 ↔** |

Equilibrium N\* = μ/λ ≈ 334 matches theoretical prediction exactly.

### Production System Comparison (LongMemEvalS)

```
GPT-4o bare   ████████████░░░░░░░░░░░░░░░  37%
MemGPT        ████████████████████░░░░░░░  48%
LangMem       ██████████████████░░░░░░░░░  44%
Mem0          ████████████████████░░░░░░░  49%
Mem0 + graph  █████████████████████░░░░░░  52%
MAC-C3 (ours) █████████████████████████████  71%  ← our system
Zep/Graphiti  █████████████████████████░░░  64%
MemMachine    █████████████████████████████  93%  ← SOTA
```

---

## Hypotheses: All Confirmed

| Hypothesis | Claim | Result |
|---|---|---|
| **H1.1** Long-term Memory Benefit | Persistent memory → statistically significant accuracy gains | ✅ +132%, p < 0.001 |
| **H1.2** Hierarchical Superiority | Hierarchical > flat on accuracy and token efficiency | ✅ +14 pp, p < 0.001 |
| **H1.3** Forgetting Necessity | Consolidation maintains precision; accumulation causes decay to zero | ✅ 0.84 stable vs 0.61 decay |
| **H1.4** Token Efficiency | Memory improves accuracy with < 80% token overhead | ✅ +49% accuracy/token |

---

## Repository Structure

```
.
├── notebook/
│   └── llmmemory     # Full experimental code
│
├── paper/
│   └── LLM_Memory_Systems.pdf        # Full research paper
│
└── README.md
```

---

## Open Problems & Research Roadmap (2026–2028)

Seven unsolved problems identified:

1. **Selective Forgetting Mastery** — most common failure mode across all benchmarked systems
2. **Temporal Reasoning at Scale** — structural KG advantage not yet replicated by vector stores
3. **Cross-Agent Memory Sharing** — no standard protocol for multi-agent memory access
4. **Privacy-Preserving Memory** — user data retention raises regulatory and ethical concerns
5. **Multimodal Memory** — extending beyond text to images, audio, structured data
6. **Memory at AGI Scale** — architectures must handle billions of stored facts
7. **Unified Benchmark Coverage** — no single standard covers all 7 evaluation dimensions

---

## Key References

- Vaswani et al. (2017) — Attention is All You Need
- Lewis et al. (2020) — Retrieval-Augmented Generation (NeurIPS)
- Park et al. (2023) — Generative Agents (UIST)
- Packer et al. (2023) — MemGPT
- Wu et al. (2025) — LongMemEval (ICLR)
- Rasmussen et al. (2025) — Zep / Graphiti
- Xu et al. (2025) — A-MEM (NeurIPS)
- Tang et al. (2025) — MemOS
- Wang et al. (2026) — MemMachine
- Sun et al. (2025) — TTT-E2E

---

## Citation

```bibtex
@misc{slamani2026memorysystems,
  author      = {Slamani, Abdelhafid},
  title       = {Memory Systems for Large Language Models},
  year        = {2026},
  institution = {Saint Petersburg State University},
  note        = {Research Survey and Experimental Study, Faculty of Mathematics and Mechanics}
}
```

---

## Author

**Slamani Abdelhafid**  
Artificial Intelligence & Data Science, Group 24.Б83-мм  
Saint Petersburg State University (SPbU)  

---

<div align="center">

*"Memory is not a performance optimization for LLMs — it is a prerequisite for genuine long-term intelligence.*  
*Persistent, adaptive, intelligent memory is what distinguishes a calculator from a colleague."*

</div>
