# HyCon: Hypergraph Contrastive Self-Supervised Learning for Multi-Modal Medical Image Segmentation

![HyCon Framework](docs/figures/hycon_pipeline.png)

> **Self-supervised learning (SSL)** has become the de-facto starting point for multi-modal medical image segmentation.  
> Yet **sequential SSL** forgets previous modalities and **joint SSL** entangles conflicting cues.  
> **HyCon** unifies the strengths of both with *hypergraph contrastive learning* and *topology-aware knowledge distillation*.

---

## ✨ Highlights
| Component | What it does | Why it matters |
|-----------|--------------|----------------|
| **Stage-1 Adversarial Pre-training** | Learns cross-modal features via CycleGAN / Pix2Pix style translation. | Reduces domain gaps early. |
| **Hypergraph Contrastive Learning Network (HCLN)** | Builds hyperedges across images and modalities, then performs student–teacher contrastive learning. | Captures *high-order* relationships that plain graphs miss. |
| **Topology Hybrid Distillation (THD)** | Distills topological patterns, context, and relations from teacher → student. | Retains useful cues while avoiding catastrophic forgetting. |
| **Two-Organ Benchmark** | Lung (CT) and Brain (MRI); 5 % labeled data. | Demonstrates robustness across anatomy and modality. |

---


