# SmartBelt — Phase 4: General Belt Surface Damage Model & Gap Analysis

> **Module:** Visual Subsystem — Dual-Branch Detection Architecture  
> **Status:** Phase 4 Complete (Architectural Design Complete | Data Ingestion Pending)  
> **Output Directory:** `smartbelt_pipeline/phase4/`  
> **Date:** 2026-09-10  

---

## 1. Codebase Data Audit: Belt Surface Images

A thorough recursive audit of the workspace was conducted to identify any collected continuous belt-surface (non-joint) imagery:

* **Inspected Directories:**
  - `dataset/candidates/`: 292 images — **100% Joint Splice frames**.
  - `dataset/cleaned/`: 292 images — **100% Joint Splice frames**.
  - `dataset/patchcore/train/good/`: 219 images — **100% Joint Splice frames**.
  - `dataset/patchcore/test/good/`: 37 images — **100% Joint Splice frames**.
  - `dataset/patchcore/test/bad/real_damage/`: 40 images — **100% Joint Rupture frames**.
  - `conveyor_original_damaged_joint_40/`: 40 images — **100% Joint Rupture frames**.
  - `dataset/patchcore/test/bad/synthetic_damage/`: 30 images — **Augmented Joint Splice frames**.
* **Audit Finding:** **Zero (0) non-joint continuous belt-surface images currently exist in the repository.**

### Scientific Integrity Stance:
In compliance with project global rules (*"Never fabricate results, sensor data, or accuracy numbers"*), we do **not** synthesize or fabricate a trained `patchcore_belt_v1.ckpt` on non-existent data. Instead, we:
1. Define the full dual-branch visual architecture.
2. Provide the visual score fusion engine (`fusion_visual_score.py`).
3. Formulate the precise data collection requirements needed to train the second branch.
4. Provide ready-to-use slide language for tomorrow's project presentation.

---

## 2. Dual-Branch Visual Architecture Overview

Conveyor failure modes partition naturally into two distinct physical zones with radically different visual textures and structural dynamics:

```
                            Camera Stream / Video Ingestion
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
       ┌─────────────────────────┐                   ┌─────────────────────────┐
       │   Branch A: Joint Model │                   │   Branch B: Belt Model  │
       │ (patchcore_joint_v1.ckpt)                   │ (patchcore_belt_v1.ckpt)│
       │   [TRAINED & VALIDATED] │                   │   [ARCHITECTURE READY]  │
       └────────────┬────────────┘                   └────────────┬────────────┘
                    │                                             │
               Joint Score                                   Belt Score
                    │                                             │
                    └──────────────────────┬──────────────────────┘
                                           ▼
                                 ┌───────────────────┐
                                 │   Visual Fusion   │
                                 │  conservative_max │
                                 └─────────┬─────────┘
                                           ▼
                                  Final Visual Score
                                  (NORMAL / WARNING / CRITICAL)
```

1. **Branch A (Joint-Specific Model):** Focuses exclusively on the mechanical splice zone (lacing pins, vulcanized seams, vulcanized finger joints), detecting cord pullout, joint separation, and fatigue cracking.
2. **Branch B (Continuous Belt Surface Model):** Focuses on the running rubber surface between joints, detecting longitudinal gouges, through-belt punctures, skirt-rubber abrasion, and edge fraying.
3. **Fusion Policy (`conservative_max`):** 
   $$\text{Final Visual Score} = \max(\text{Joint Score}, \text{Belt Score})$$
   *Rationale:* Conveyor failure is an **OR-condition**. A severe longitudinal belt rip is just as catastrophic to plant operations as a joint snap. Either subsystem detecting an anomaly immediately elevates plant risk.

---

## 3. Data Requirements to Train Branch B (`patchcore_belt_v1.ckpt`)

To train and deploy the general belt surface branch, the following dataset must be ingested into `dataset/patchcore_belt/`:

| Dataset Partition | Recommended Quantity | Description & Specifications |
| :--- | :---: | :--- |
| **Train Normal (`train/good`)** | **150 – 300 images** | High-resolution images of nominal continuous rubber surface (free of joints, taken across various conveyor lighting conditions and belt moisture states). |
| **Validation Normal (`val/good`)** | **50 images** | Unseen normal continuous surface for statistical threshold calibration ($\mu_{val} + 3\sigma_{val}$). |
| **Test Normal (`test/good`)** | **50 images** | Nominal running belt surface for false alarm validation. |
| **Test Damage (`test/bad`)** | **40+ images** | Real or controlled damaged surface samples categorized by failure mode:<br>• Longitudinal gouges / cuts<br>• Punctures from trapped material<br>• Edge fraying / delamination |

---

## 4. Recommended Presentation Slide Language *(For Tomorrow)*

Use the following slide copy to explain this design to evaluators and judges:

### Slide Title:
> **SmartBelt Visual Architecture: Modular Dual-Branch Anomaly Detection**

### Bullet Points:
* **Decoupled Risk Zones:** Joint splices and continuous belt surfaces experience entirely distinct stress dynamics. Conflating them into a single generic vision model dilutes feature sensitivity.
* **Specialized Joint Model [Operational]:**
  - PatchCore WideResNet-50 trained on 219 normal joint splices.
  - Achieved **100% detection rate (AUROC 1.0, 0 False Negatives)** on real damaged joint footage with zero false plant shutdowns.
* **Belt Surface Model [Architecture Specified]:**
  - Pre-engineered modular pipeline mirroring the joint model architecture.
  - Ready for data ingestion upon deployment of continuous surface inspection cameras.
* **Conservative Fusion Policy:** $\text{Risk} = \max(\text{Joint Anomaly}, \text{Belt Anomaly})$ ensures that structural degradation in either zone triggers immediate automated motor deceleration and emergency alerts.

### Judge Verbal Pitch:
> *"In conveyor operations, joint rupture represents the catastrophic sudden-failure mode, while belt surface wear represents progressive degradation. To ensure maximum sensitivity, SmartBelt treats them as decoupled visual branches rather than a noisy single model. Our joint-specific model is fully trained, benchmarked at 1.0 AUROC, and operational. We have established the dual-branch fusion architecture so that once continuous surface cameras are positioned in production, the second model slots in seamlessly with zero system refactoring."*

---

## 5. Deliverables Index

All files are located in [`smartbelt_pipeline/phase4/`](file:///c:/Users/korra/OneDrive/Desktop/Conveyor/smartbelt_pipeline/phase4/):

* [**`fusion_visual_score.py`**](file:///c:/Users/korra/OneDrive/Desktop/Conveyor/smartbelt_pipeline/phase4/fusion_visual_score.py) — Production visual fusion engine implementing `combined_visual_score(joint_score, belt_score)` with 6 multi-scenario architectural validation tests.
* [**`belt_damage_gap_report.md`**](file:///c:/Users/korra/OneDrive/Desktop/Conveyor/smartbelt_pipeline/phase4/belt_damage_gap_report.md) — This formal gap analysis, dataset specification, and presentation guide.
