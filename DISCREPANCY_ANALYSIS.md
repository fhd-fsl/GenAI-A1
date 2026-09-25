# Discrepancy Analysis: Assignment Requirements vs. Project Documentation

**Date:** 2026-09-25  
**Scope:** All markdown files in the project vs. `assignment_text.txt` (350 lines)

---

## Summary

| Document | Status | Critical Issues |
|----------|--------|-----------------|
| `PROJECT_PLAN.md` | ✅ Mostly aligned | Missing IEEE LaTeX report details, Google Stitch evidence, AI-use appendix |
| `README.md` | ✅ Good alignment | Missing demo video requirements, experiment tracking specifics |
| `.agents/AGENTS.md` | ✅ Good alignment | Missing SSIM loss implementation details |
| `.agents/rules/corruptions.md` | ⚠️ **Critical gaps** | Missing validation/test manifest specifics, equal probability enforcement |
| `.agents/rules/application.md` | ✅ Good alignment | Missing health-check endpoint specifics |
| `.agents/rules/research_requirements.md` | ⚠️ **Missing sections** | No mention of Optuna bounds justification, parameter importance plots, training curves |
| `.agents/skills/data-pipeline/SKILL.md` | ⚠️ **Critical gaps** | Missing balanced batching for classifier, paired augmentation for FS2K details |
| `.agents/skills/train-model/SKILL.md` | ⚠️ **Critical gaps** | Missing Task 2 specialist independent training detail, Task 3 warm-up specifics |
| `.agents/skills/optuna-study/SKILL.md` | ⚠️ **Missing details** | No mention of trial pruning for Task 3 routing collapse, GAN fewer epochs |
| `.agents/skills/onnx-export/SKILL.md` | ⚠️ **Critical gaps** | Missing Task 3 identity branch export, Task 4 style embedding in ONNX |

---

## Detailed Discrepancies by Category

---

### 1. DATASET & CORRUPTION PIPELINE (Tasks 1–3)

| Requirement (Assignment) | Project Docs | Discrepancy |
|--------------------------|--------------|-------------|
| **80/20 split with seed 42** | ✅ PROJECT_PLAN.md:36, data-pipeline:15 | None |
| **Runtime corruption (not pre-saved)** | ✅ corruptions.md:3, data-pipeline:17 | None |
| **Equal probability (25% each)** | ⚠️ corruptions.md:5 says "EQUAL probability" but no code enforcement | **No validation in code** |
| **Salt-and-pepper: p ∈ [0.02, 0.15]** | ✅ data-pipeline:21, PROJECT_PLAN.md:48 | None |
| **Gaussian blur: kernel ∈ {3,5,7}, σ ∈ [0.5, 2.5]** | ✅ data-pipeline:22, PROJECT_PLAN.md:49 | None |
| **Occlusion: 1–3 rects, 10–35% area** | ✅ data-pipeline:23, PROJECT_PLAN.md:50 | None |
| **Validation manifest: deterministic** | ✅ PROJECT_PLAN.md:51, data-pipeline:25 | None |
| **Test manifest: 3 severity levels per corruption** | ✅ PROJECT_PLAN.md:52-55, data-pipeline:37-42 | None |
| **Test S&P: p = 0.03, 0.08, 0.15** | ✅ data-pipeline:40 | None |
| **Test Blur: (3,0.7), (5,1.5), (7,2.5)** | ✅ data-pipeline:41 | None |
| **Test Occlusion: ~10%/1, ~20%/2, ~35%/3** | ✅ data-pipeline:42 | None |
| **Corruption label returned for classifier** | ✅ corruptions.md:9 | None |
| **Balanced batches for classifier (Task 2)** | ⚠️ train-model:41 mentions "Balanced batches" but data-pipeline **does not specify how** | **Missing implementation detail** |

**🔴 Critical Missing:** How balanced batching is actually implemented in the data loader.

---

### 2. TASK 1: UNIVERSAL DENOISING AUTOENCODER

| Requirement | Project Docs | Discrepancy |
|-------------|--------------|-------------|
| Conv encoder → bottleneck → conv decoder | ✅ PROJECT_PLAN.md:64-66, train-model:29 | None |
| **Genuine compressed latent (bottleneck)** | ✅ PROJECT_PLAN.md:65, AGENTS.md:8 | None |
| **Limited skip connections justified** | ✅ research_requirements.md:8 mentions justification needed | **No architecture decision documented** |
| **Loss: α·L1 + (1-α)·(1-SSIM)** | ✅ PROJECT_PLAN.md:70, train-model:32 | None |
| **Initial α = 0.8** | ✅ PROJECT_PLAN.md:71 | None |
| **Optuna: LR, batch, bottleneck, channels, dropout, α** | ✅ PROJECT_PLAN.md:76, optuna-study:27-35 | None |
| **Val objective: reconstruction + SSIM** | ✅ optuna-study:36 | None |
| **Report: search space, trials, best trial, final config** | ✅ optuna-study:83-89 | None |
| **Per-corruption-type metrics** | ✅ PROJECT_PLAN.md:81 | None |
| **Per-severity-level metrics** | ✅ PROJECT_PLAN.md:82 | None |
| **Visual grid: clean | corrupted | reconstruction | error map** | ✅ PROJECT_PLAN.md:83 | None |
| **≥12 representative + ≥4 failure cases** | ✅ PROJECT_PLAN.md:84 | None |
| **ONNX export + verification** | ✅ PROJECT_PLAN.md:87-88, onnx-export:56-60 | None |
| **App: Universal Restoration workspace** | ✅ PROJECT_PLAN.md:217, application.md:14 | None |
| **App: upload/select + apply corruption + show input/output/settings/time** | ✅ PROJECT_PLAN.md:217 | None |

**🟡 Gap:** Skip connection architecture decision not documented in any config or architecture file.

---

### 3. TASK 2: CORRUPTION CLASSIFIER & HARD-ROUTED SPECIALISTS

| Requirement | Project Docs | Discrepancy |
|-------------|--------------|-------------|
| Classifier: 4 classes (clean, S&P, blur, occlusion) | ✅ PROJECT_PLAN.md:96, train-model:39 | None |
| **Balanced training batches** | ✅ train-model:41, AGENTS.md:8 | **data-pipeline missing implementation** |
| Cross-entropy loss | ✅ PROJECT_PLAN.md:98, train-model:40 | None |
| Optuna: LR, batch, channels, dropout, weight decay | ✅ PROJECT_PLAN.md:99, optuna-study:38-45 | None |
| Metrics: accuracy, macro P/R/F1, per-class, confusion matrix | ✅ PROJECT_PLAN.md:100 | None |
| **3 specialist AEs (S&P, blur, occlusion)** | ✅ PROJECT_PLAN.md:102-106, train-model:44-49 | None |
| **Same architecture base, independent weights** | ✅ PROJECT_PLAN.md:106, train-model:48 | None |
| **Optuna shared search → independent training** | ✅ PROJECT_PLAN.md:107, optuna-study:48-56 | None |
| Clean → identity bypass | ✅ PROJECT_PLAN.md:108, train-model:49 | None |
| **Oracle-routing mode** | ✅ PROJECT_PLAN.md:111 | None |
| **Predicted-routing mode** | ✅ PROJECT_PLAN.md:112 | None |
| **Compare both modes** | ✅ PROJECT_PLAN.md:113 | None |
| **Identify classifier-error-induced failures** | ✅ PROJECT_PLAN.md:114, research_requirements.md:13 | None |
| ONNX: classifier + 3 specialists | ✅ PROJECT_PLAN.md:117-118, onnx-export:62-67 | None |
| App: Hard-Routed Restoration workspace | ✅ PROJECT_PLAN.md:218, application.md:15 | None |
| App: show 4 probs, predicted corruption, selected expert, reconstruction, time | ✅ PROJECT_PLAN.md:218 | None |

**🔴 Critical Missing:** Balanced batching implementation in data pipeline not specified.

---

### 4. TASK 3: SOFT MIXTURE-OF-EXPERTS

| Requirement | Project Docs | Discrepancy |
|-------------|--------------|-------------|
| Gating network: softmax(G(x̃)/τ) → 4 weights | ✅ PROJECT_PLAN.md:126-127, train-model:53-60 | None |
| Initialize gate from Task 2 classifier | ✅ PROJECT_PLAN.md:128, train-model:58 | None |
| Soft fusion: x̂ = w₁·x̃ + w₂·A_salt + w₃·A_blur + w₄·A_occ | ✅ PROJECT_PLAN.md:131, train-model:64 | None |
| Initialize experts from Task 2 specialists | ✅ PROJECT_PLAN.md:132, train-model:59 | None |
| **Stage 1: Freeze experts, train gate only** | ✅ PROJECT_PLAN.md:135, train-model:54 | **No epoch count specified** |
| **Stage 2: Unfreeze all, smaller LR** | ✅ PROJECT_PLAN.md:136, train-model:55 | **No LR ratio specified** |
| Joint loss: λ₁·L1 + λ₂·(1-SSIM) + λ₃·L_CE + λ₄·L_balance | ✅ PROJECT_PLAN.md:137, train-model:64 | None |
| Balance loss: Σ(w̄_k - 1/4)² | ✅ PROJECT_PLAN.md:138, train-model:65 | None |
| Initial λ: 0.8, 0.2, 0.1, 0.01 | ✅ PROJECT_PLAN.md:139, train-model:67 | None |
| **Alternative balance/entropy regularizer allowed if justified** | ✅ research_requirements.md:17 | **No research documented** |
| Optuna: joint LR, τ, CE weight, balance weight, recon weight | ✅ PROJECT_PLAN.md:142, optuna-study:58-67 | None |
| **Pruning on routing collapse (one expert > 90%)** | ✅ optuna-study:67 | **Not in PROJECT_PLAN.md** |
| Recon quality vs Task 1 & 2 | ✅ PROJECT_PLAN.md:146 | None |
| **Average expert weights per corruption × severity** | ✅ PROJECT_PLAN.md:147 | None |
| **Routing heatmap / weight-distribution diagrams** | ✅ PROJECT_PLAN.md:148, research_requirements.md:18 | None |
| **Examples: single expert dominance vs distributed** | ✅ PROJECT_PLAN.md:149 | None |
| **Check inactive expert / over-dominant expert** | ✅ PROJECT_PLAN.md:150, research_requirements.md:18 | None |
| ONNX: gate + 3 fine-tuned experts (separate files) | ✅ onnx-export:69-84 | **PROJECT_PLAN.md:153 says "complete pipeline" but onnx-export says separate** |
| **Inference pipeline orchestrated in backend** | ✅ onnx-export:77-80 | **Not in PROJECT_PLAN.md or application.md** |
| App: Soft MoE Restoration workspace | ✅ PROJECT_PLAN.md:219, application.md:16 | None |
| App: show 4 weights, visual expert contribution, reconstruction, time | ✅ PROJECT_PLAN.md:219 | None |

**🔴 Critical Conflicts:**
1. PROJECT_PLAN.md:153 says "Export complete soft MoE pipeline to ONNX" but onnx-export:69-84 explicitly says **separate files + backend orchestration**
2. Warm-up epoch count not specified anywhere
3. LR ratio for joint fine-tuning not specified
4. Routing collapse detection threshold (90%) only in optuna-study, not PROJECT_PLAN

---

### 5. TASK 4: CONDITIONAL GAN (FACE-TO-SKETCH)

| Requirement | Project Docs | Discrepancy |
|-------------|--------------|-------------|
| FS2K dataset (2,104 paired, 3 styles) | ✅ PROJECT_PLAN.md:160, data-pipeline:52 | None |
| Official train/test + 15% stratified val (seed 42) | ✅ PROJECT_PLAN.md:163, data-pipeline:55 | None |
| **Paired transforms: identical spatial augmentation** | ✅ data-pipeline:58 | **Critical - not in PROJECT_PLAN.md** |
| Resize 128×128 | ✅ PROJECT_PLAN.md:165 | None |
| Style labels loaded | ✅ PROJECT_PLAN.md:166, data-pipeline:62 | None |
| Generator: U-Net encoder-decoder | ✅ PROJECT_PLAN.md:169, train-model:71 | None |
| Skip connections in U-Net | ✅ PROJECT_PLAN.md:169 | None |
| Input: photo + style condition (learned embedding) | ✅ PROJECT_PLAN.md:170-171, train-model:72-73 | None |
| Output: sketch 128×128 | ✅ PROJECT_PLAN.md:171 | None |
| Discriminator: PatchGAN | ✅ PROJECT_PLAN.md:174, train-model:76 | None |
| Input: (photo, sketch, style_embedding) | ✅ PROJECT_PLAN.md:174-175, train-model:77 | None |
| Patch-level real/fake | ✅ PROJECT_PLAN.md:175 | None |
| G_loss = L_adv + λ_L1 · L1(y, G(x,s)) | ✅ PROJECT_PLAN.md:178, train-model:81 | None |
| D_loss = BCE(real) + BCE(fake) | ✅ PROJECT_PLAN.md:179, train-model:82 | None |
| Initial λ_L1 = 100 | ✅ PROJECT_PLAN.md:180, train-model:81 | None |
| **Log: D_real, D_fake, G_adv, G_recon, val metrics** | ✅ PROJECT_PLAN.md:181, train-model:85-87 | None |
| **Log samples at fixed intervals (same val photos)** | ✅ PROJECT_PLAN.md:182, train-model:87 | None |
| Optuna: G_lr, D_lr, batch, channels, dropout, embed_dim, λ_L1 | ✅ PROJECT_PLAN.md:185, optuna-study:70-78 | None |
| **Fewer epochs per trial → full retrain best** | ✅ optuna-study:80 | **Not in PROJECT_PLAN.md** |
| Side-by-side: photo | generated | ground truth | ✅ PROJECT_PLAN.md:189 | None |
| Per-style results | ✅ PROJECT_PLAN.md:190 | None |
| FID or GAN metrics if feasible | ✅ PROJECT_PLAN.md:191 | None |
| ONNX: Generator only (discriminator training-only) | ✅ PROJECT_PLAN.md:194, onnx-export:86-91 | None |
| Style embedding inside ONNX | ✅ onnx-export:91 | None |
| App: Face-to-Sketch Generator workspace | ✅ PROJECT_PLAN.md:220, application.md:17 | None |
| App: upload/webcam → select style 1/2/3 → generate → side-by-side → download | ✅ PROJECT_PLAN.md:220 | None |

**🔴 Critical Missing in PROJECT_PLAN.md:**
- Paired augmentation rule (only in data-pipeline skill)
- Fewer epochs for Optuna trials (only in optuna-study skill)

---

### 6. APPLICATION ARCHITECTURE

| Requirement | Project Docs | Discrepancy |
|-------------|--------------|-------------|
| **Google Stitch design first** | ✅ PROJECT_PLAN.md:202, application.md:13 | **No evidence of Stitch design captured** |
| 4 workspaces with exact names | ✅ PROJECT_PLAN.md:217-220, application.md:13-17 | None |
| FastAPI backend | ✅ PROJECT_PLAN.md:205, application.md:4 | None |
| Health-check endpoint | ✅ PROJECT_PLAN.md:206 | **application.md:8 says `/health` but no details** |
| `/api/universal-restore` | ✅ PROJECT_PLAN.md:207 | None |
| `/api/hard-route` (probs + expert + result) | ✅ PROJECT_PLAN.md:208 | None |
| `/api/soft-mixture` (4 weights + result) | ✅ PROJECT_PLAN.md:209 | None |
| `/api/face-to-sketch` (style selection) | ✅ PROJECT_PLAN.md:210 | None |
| ONNX sessions loaded ONCE at startup | ✅ application.md:7 | None |
| All endpoints return JSON + timing | ✅ application.md:8 | None |
| File upload validation (image, max 10MB) | ✅ application.md:9 | None |
| Preprocessing server-side | ✅ application.md:10 | None |
| React + Tailwind frontend | ✅ PROJECT_PLAN.md:215, application.md:12 | None |
| Shared components: ImageUploader, ResultDisplay, MetricsCard | ✅ application.md:18 | None |
| Webcam capture for Task 4 | ✅ PROJECT_PLAN.md:220, application.md:19 | None |
| Responsive layout | ✅ PROJECT_PLAN.md:221 | None |
| Docker: backend (Python 3.12, onnxruntime, FastAPI) | ✅ PROJECT_PLAN.md:228, application.md:22 | None |
| Docker: frontend (Node build → nginx) | ✅ PROJECT_PLAN.md:229, application.md:23 | None |
| docker-compose.yml at root | ✅ PROJECT_PLAN.md:230, application.md:24 | None |
| ONNX models mounted/downloaded | ✅ PROJECT_PLAN.md:231, application.md:25 | None |
| README with complete instructions | ✅ PROJECT_PLAN.md:232 | None |
| Verify: clone → get models → docker compose up → browser | ✅ PROJECT_PLAN.md:233 | None |

**🟡 Gaps:**
- Google Stitch design evidence not captured anywhere
- Health-check endpoint response format not specified
- API response schemas not defined

---

### 7. EXPERIMENT TRACKING & REPORTING

| Requirement | Project Docs | Discrepancy |
|-------------|--------------|-------------|
| MLflow or W&B (mandatory) | ✅ AGENTS.md:9, train-model:89 | None |
| Log: hyperparameters | ✅ train-model:91 | None |
| Log: train/val losses per epoch | ✅ train-model:92 | None |
| Log: best metrics | ✅ train-model:93 | None |
| Log: model checkpoint path | ✅ train-model:94 | None |
| Log: visual samples (periodic) | ✅ train-model:95 | None |
| **IEEE LaTeX report format** | ⚠️ PROJECT_PLAN.md:240-259 lists sections but **no LaTeX template** | **Missing LaTeX structure** |
| Report: all 4 tasks separately | ✅ PROJECT_PLAN.md:240-259 | None |
| Architecture diagrams | ✅ PROJECT_PLAN.md:244, 253 | None |
| Training/validation curves | ✅ PROJECT_PLAN.md:250 | None |
| Optuna results | ✅ PROJECT_PLAN.md:247, 251 | None |
| Confusion matrices (T2) | ✅ PROJECT_PLAN.md:251 | None |
| Routing heatmaps (T3) | ✅ PROJECT_PLAN.md:252 | None |
| Generated image grids (T4) | ✅ PROJECT_PLAN.md:253 | None |
| Error maps | ✅ PROJECT_PLAN.md:254 | None |
| Application screenshots | ✅ PROJECT_PLAN.md:255 | None |
| Failure cases | ✅ PROJECT_PLAN.md:255 | None |
| **AI-use appendix** | ✅ PROJECT_PLAN.md:259 | **Not in research_requirements.md** |
| **Interpret every table/diagram/image in text** | ✅ research_requirements.md:22 | None |

**🔴 Critical Missing:**
- No LaTeX report template or structure defined
- AI-use appendix requirement only in PROJECT_PLAN.md, not in research_requirements.md

---

### 8. DEMO VIDEO

| Requirement | Project Docs | Discrepancy |
|-------------|--------------|-------------|
| 5-7 minutes | ✅ PROJECT_PLAN.md:261 | None |
| YouTube upload + link in report | ✅ PROJECT_PLAN.md:261-262 | None |
| App startup (docker compose) | ✅ PROJECT_PLAN.md:262 | None |
| Task 1: upload → corrupt → restore | ✅ PROJECT_PLAN.md:263 | None |
| Task 2: upload → classify → route → restore | ✅ PROJECT_PLAN.md:264 | None |
| Task 3: upload → soft routing weights → restore | ✅ PROJECT_PLAN.md:265 | None |
| Task 4: upload/webcam → select style → generate → download | ✅ PROJECT_PLAN.md:266 | None |
| Show W&B/MLflow records | ✅ PROJECT_PLAN.md:267 | None |
| **Must not upload to Google Classroom** | ❌ **Not documented anywhere** | **Missing constraint** |

---

### 9. GIT & REPOSITORY

| Requirement | Project Docs | Discrepancy |
|-------------|--------------|-------------|
| GitHub repo with source, configs, deps, data prep, training, eval, Optuna, ONNX, app, Docker, README | ✅ PROJECT_PLAN.md:48-51 | None |
| **No datasets or large model files on GitHub** | ✅ AGENTS.md:48-50 | None |
| **Git LFS or download links for trained models** | ✅ AGENTS.md:50 | None |
| .gitignore for data/, checkpoints/, optuna_studies/, __pycache__/ | ✅ AGENTS.md:49 | None |

**✅ All covered**

---

### 10. RESEARCH & ANALYSIS REQUIREMENTS

| Requirement | Project Docs | Discrepancy |
|-------------|--------------|-------------|
| **Independent research mandatory** | ✅ research_requirements.md:3 | None |
| Task 1: Skip connections justified | ✅ research_requirements.md:8 | **No analysis documented** |
| Task 1: α selected via Optuna (not blind) | ✅ research_requirements.md:9 | None |
| Task 2: Classifier arch & regularization justified | ✅ research_requirements.md:12 | **No analysis documented** |
| Task 2: Error analysis (classifier → restoration failures) | ✅ research_requirements.md:13 | None |
| Task 2: Specialist arch justified (same as Task 1?) | ✅ research_requirements.md:14 | **Not addressed** |
| Task 3: Balance/entropy regularizer research & justification | ✅ research_requirements.md:17 | **Not addressed** |
| Task 3: Routing behavior analysis (τ, mixing, inactive/dominant) | ✅ research_requirements.md:18 | None |
| General: Optuna bounds justified by literature/empirical | ✅ research_requirements.md:21 | **Not addressed** |
| General: Optuna parameter importance plots analyzed | ✅ research_requirements.md:22, optuna-study:88 | **Not in PROJECT_PLAN.md report sections** |

**🔴 Critical Missing:** No research analysis documented for any of the required justification points.

---

## Cross-Document Conflicts

| Conflict | Location A | Location B | Resolution Needed |
|----------|------------|------------|-------------------|
| **Task 3 ONNX: single pipeline vs. separate files** | PROJECT_PLAN.md:153 "Export complete soft MoE pipeline to ONNX" | onnx-export.md:69-84 "Cannot reuse Task 2 ONNX... Export gate + experts separately... Inference pipeline orchestrated in backend" | **Must decide: single ONNX model or backend-orchestrated multi-model** |
| **Balanced batching: mentioned but not implemented** | train-model.md:41 "Balanced batches" | data-pipeline.md: no implementation details | **Add balanced sampler to data-pipeline skill** |
| **Paired augmentation: only in data-pipeline skill** | data-pipeline.md:58 "CRITICAL: identical spatial augmentation" | PROJECT_PLAN.md: no mention | **Add to PROJECT_PLAN.md M5.1** |
| **Task 3 warm-up epochs: not specified** | PROJECT_PLAN.md:135 "Short warm-up stage" | train-model.md:54 "Warm-up: Freeze experts, train gate only" | **Define epoch count or convergence criterion** |
| **Task 3 joint LR ratio: not specified** | PROJECT_PLAN.md:136 "smaller LR" | train-model.md:55 "use smaller LR" | **Define ratio (e.g., 1/10th)** |
| **Optuna trial pruning: only in optuna-study** | optuna-study.md:67 "Prune if routing collapse" | PROJECT_PLAN.md:143 "Pruning on routing collapse" | **Add pruning threshold to PROJECT_PLAN.md** |
| **GAN fewer epochs: only in optuna-study** | optuna-study.md:80 "Use fewer epochs per trial" | PROJECT_PLAN.md:185-186 no mention | **Add to PROJECT_PLAN.md M5.5** |
| **AI-use appendix: only in PROJECT_PLAN** | PROJECT_PLAN.md:259 | research_requirements.md: no mention | **Add to research_requirements.md** |

---

## Priority Fix List

### 🔴 P0 - Critical (Blockers for Implementation)

1. **Task 3 ONNX export strategy conflict** - Decide single vs. multi-model ONNX and update all docs
2. **Balanced batching implementation** - Add `BalancedBatchSampler` to data-pipeline skill and PROJECT_PLAN
3. **Paired augmentation for FS2K** - Add to PROJECT_PLAN.md M5.1
3. **Google Stitch design evidence** - Define where screenshots/exports will be stored
4. **LaTeX report template** - Create `reports/template.tex` with IEEE structure

### 🟠 P1 - High (Missing Details)

5. **Task 3 warm-up epoch count & convergence criterion**
6. **Task 3 joint fine-tuning LR ratio** (e.g., 0.1× warm-up LR)
7. **Task 3 routing collapse threshold** (90% from optuna-study → PROJECT_PLAN)
8. **GAN Optuna: fewer epochs per trial** (e.g., 20 vs 100 full)
9. **Health-check endpoint response schema**
10. **API response schemas** for all 4 endpoints

### 🟡 P2 - Medium (Research Documentation)

11. **Skip connection justification** for Task 1 (document in research_requirements)
12. **Classifier architecture justification** (Task 2)
13. **Specialist architecture justification** (same as Task 1 or different?)
14. **Balance regularizer research** (Task 3 - reference Shazeer MoE paper)
15. **Optuna bounds justification** for all tasks
16. **AI-use appendix** added to research_requirements.md

### 🟢 P3 - Low (Polish)

17. **Demo video: "not Google Classroom" constraint** added to PROJECT_PLAN
18. **Parameter importance plots** in report sections
19. **Training curves** logging specification

---

## Recommended Updates to Files

### 1. `PROJECT_PLAN.md` - Add:
- Paired augmentation rule in M5.1
- Task 3 warm-up epochs (e.g., "5 epochs or until gate loss plateaus")
- Task 3 joint LR ratio (e.g., "1/10th of warm-up LR")
- Routing collapse pruning threshold (90%)
- GAN Optuna trial epochs (e.g., "20 epochs per trial, 100 for final")
- Demo video constraint (not Google Classroom)
- LaTeX report template reference

### 2. `.agents/rules/corruptions.md` - Add:
- Explicit equal probability enforcement (25% each)
- Validation/test manifest JSON schema with all required fields

### 3. `.agents/skills/data-pipeline/SKILL.md` - Add:
- Balanced batch sampler implementation
- FS2K paired augmentation code pattern

### 4. `.agents/skills/train-model/SKILL.md` - Add:
- Task 3 warm-up epoch count
- Task 3 joint LR ratio
- Specialist independent training detail

### 5. `.agents/skills/optuna-study/SKILL.md` - Add:
- Routing collapse pruning callback code
- GAN trial epoch count

### 6. `.agents/skills/onnx-export/SKILL.md` - Add:
- Task 3 identity branch handling (pass-through in backend)
- Task 4 style embedding export verification

### 7. `.agents/rules/research_requirements.md` - Add:
- AI-use appendix requirement
- Optuna bounds justification requirement
- Parameter importance analysis requirement

### 8. Create new: `reports/template.tex` - IEEE LaTeX template

---

## Verification Checklist

After fixes, verify:
- [ ] All 4 tasks have complete Optuna search spaces matching assignment
- [ ] All corruption specs match assignment exactly
- [ ] Task 3 ONNX strategy is consistent across all docs
- [ ] Balanced batching is implementable from data-pipeline skill
- [ ] FS2K paired augmentation is documented in both skills and plan
- [ ] LaTeX report template exists with all required sections
- [ ] Google Stitch design capture process documented
- [ ] Demo video requirements complete
- [ ] AI-use appendix in research requirements
- [ ] All cross-doc conflicts resolved

---

*Generated by analyzing all project markdown files against assignment_text.txt*