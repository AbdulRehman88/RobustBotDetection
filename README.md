# RobustBotDetection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24152213.svg)](https://doi.org/10.5281/zenodo.24152213)

Reproducible implementation of **Reliability-Guided Expert Routing for Multimodal Twitter Bot Detection under Informative Modality Missingness**.

This repository implements a leakage-controlled multimodal bot-detection pipeline with:

- frozen BERTweet semantic-profile representations;
- training-fitted account-metadata features;
- topology-only heterogeneous graph features;
- text, metadata, graph, and text–metadata experts;
- validation-guided availability-conditioned expert routing;
- informative-missingness and pattern–label bias auditing;
- pattern–label-weighted debiased learning;
- validation-only probability calibration;
- random and label-dependent missingness stress tests;
- representation-corruption sensitivity;
- paired, seed-aware statistical inference; and
- separately processed TwiBot-20 zero-shot, threshold-adapted, and replication evaluation with identifier-overlap auditing.

## Release scope

The repository publishes the complete analysis code, experiment configuration, environment specifications, provenance hashes, aggregate evidence, and figure-generation scripts.

**Raw TwiBot-20 and TwiBot-22 data are not redistributed.** These datasets and any Twitter/X-derived content remain governed by their original access conditions and platform terms. See [DATA_ACCESS.md](DATA_ACCESS.md).

## Scientific design

The primary benchmark is TwiBot-22 and the external benchmark is TwiBot-20. Official train, validation, and test assignments are preserved.

The implementation enforces the following separation:

1. preprocessing is fitted on training data only;
2. hyperparameters, checkpoints, thresholds, routing, and calibration are selected using validation data only;
3. the source test split is evaluated after all selectable components are locked;
4. strict zero-shot TwiBot-20 evaluation retains source preprocessing, weights, and thresholds;
5. target-validation threshold adaptation and within-target replication are reported as separate protocols.

The primary semantic text contains profile description and location. Names and usernames are excluded from the primary representation and evaluated separately as an identity-sensitivity control. No tweet-content, posting-rhythm, or temporal-coordination claim is made.

## Repository structure

```text
RobustBotDetection/
├── configs/
│   └── experiment.yaml
├── evidence/
│   ├── critical_artifact_inventory.csv
│   ├── external_master_results.csv
│   ├── final_evidence_audit.json
│   ├── final_key_evidence.csv
│   ├── final_package_hashes.csv
│   └── manifest_status_inventory.csv
├── manifests/
│   ├── environment.lock.txt
│   └── input_manifest.json
├── scripts/
│   ├── 01_audit_dataset.py
│   ├── ...
├── src/robust_bot_detection/
├── tests/
├── DATA_ACCESS.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── requirements.txt
└── README.md

Generated arrays, checkpoints, logs, raw datasets, and output directories are intentionally excluded from version control.
System used for the archived experiments
- Windows 11
- Python 3.11.9
- NVIDIA GeForce RTX 3090 Ti, 24 GB
- CUDA 12.6
- PyTorch 2.14.1+cu126
- PyTorch Geometric 2.8.1
- Transformers 5.21.0
- XGBoost 3.2.0
Exact installed packages are recorded in manifests/environment.lock.txt.
Installation
Create and activate a clean Python 3.11 environment:
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r .\requirements.txt
python -m pip install -e .

Confirm GPU availability:
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

The implementation was validated with CUDA. Some feature-building and statistical-analysis stages can run on CPU, but neural encoding and training scripts expect a CUDA-capable GPU.
Data placement
Dataset locations are configured locally and must not be committed. Copy configs/experiment.yaml only for scientific settings; use the ignored file configs/config.local.yaml for machine-specific paths.

The original TwiBot-22 files used by this release are:
user.json
label.csv
split.csv
edge.csv

Their byte sizes and SHA-256 hashes are recorded in manifests/input_manifest.json.

The TwiBot-20 Format-22 release is expected to contain:
Twibot-20/
├── node.json
├── label.csv
├── split.csv
└── edge.csv

See [DATA_ACCESS.md](DATA_ACCESS.md) for authorized acquisition and verification instructions.
Reproduction workflow
Run scripts from the repository root in numerical order. Each script writes a manifest or auditable result under outputs/.
1. Primary-data audit and canonicalization
python .\scripts\01_audit_dataset.py
python .\scripts\02_audit_modality_fields.py
python .\scripts\03_build_canonical_data.py
python .\scripts\04_audit_graph_protocols.py

2. Metadata and text features
python .\scripts\05_build_metadata_features.py
python .\scripts\06_build_tfidf_features.py
python .\scripts\07_run_classical_baselines.py
python .\scripts\08_tokenize_bertweet.py
python .\scripts\09_encode_frozen_bertweet.py
python .\scripts\10_run_bertweet_baselines.py

3. Graph controls and multimodal baselines
python .\scripts\11_build_graph_inputs.py
python .\scripts\12_train_graph_baselines.py --model rgcn --protocol full_transductive
python .\scripts\12_train_graph_baselines.py --model rgcn --protocol split_contained
python .\scripts\13_extract_rgcn_embeddings.py --protocol full_transductive
python .\scripts\13_extract_rgcn_embeddings.py --protocol split_contained
python .\scripts\14_train_multimodal_fusion.py --fusion gated --protocol full_transductive --modality-dropout off
python .\scripts\14_train_multimodal_fusion.py --fusion gated --protocol full_transductive --modality-dropout on
python .\scripts\15_finetune_bertweet.py
python .\scripts\16_run_xgboost_metadata.py
python .\scripts\17_build_heterogeneous_graph_features.py
python .\scripts\18_run_heterogeneous_graph_baseline.py
python .\scripts\19_run_metadata_heterogeneous_graph.py

Scripts 12–15 are comparator and sensitivity experiments. The primary expert-routing framework uses the frozen BERTweet, XGBoost metadata, XGBoost topology, and frozen BERTweet-plus-metadata experts.
4. Identity, routing, calibration, and robustness
python .\scripts\20_build_identity_tfidf.py
python .\scripts\21_run_identity_sensitivity.py
python .\scripts\22_build_reliability_router.py
python .\scripts\23_calibrate_experts.py
python .\scripts\24_run_robustness_stress_tests.py
python .\scripts\25_run_statistical_inference.py
python .\scripts\26_run_sequence_length_sensitivity.py
python .\scripts\27_run_missingness_bias_audit.py
python .\scripts\28_train_missingness_debiased_model.py

5. TwiBot-20 external evaluation with identifier-overlap auditing
python .\scripts\29_audit_twibot20_external.py
python .\scripts\30_build_twibot20_external_features.py
python .\scripts\31_tokenize_twibot20_external.py
python .\scripts\32_encode_twibot20_external_bertweet.py
python .\scripts\33_evaluate_twibot20_external_transfer.py
python .\scripts\34_evaluate_twibot20_external_metadata.py
python .\scripts\35_run_twibot20_length_sensitivity.py
python .\scripts\36_run_twibot20_external_inference.py
python .\scripts\37_run_twibot20_external_graph_controls.py
python .\scripts\38_run_twibot20_graph_inference.py

6. Evidence finalization and manuscript result figures
python .\scripts\39_finalize_experiment_evidence.py
python .\scripts\40_create_manuscript_figures.py

7. Identifier-overlap sensitivity and revised external figure
python .\scripts\42_run_twibot20_overlap_sensitivity.py
python .\scripts\43_regenerate_twibot20_disjoint_figure.py

Evidence package
The evidence/ directory contains compact aggregate evidence from the completed run:
- inventory of critical artifacts;
- PASS-manifest inventory;
- internal and external key results;
- external master results;
- final evidence-audit report;
- SHA-256 hashes for the release evidence package;
- identifier-overlap auditing and overlap-excluded TwiBot-20 aggregate results; and
- the revised identifier-disjoint external-validation figure.
The core evidence audit verified 27 critical artifacts, 47 PASS manifests, and 92 key evidence rows. The subsequent identifier-overlap sensitivity is preserved under evidence/twibot20_overlap_sensitivity/, while the revised external-validation figure is stored under evidence/figures/. These files support result verification without redistributing user-level identifiers, predictions, embeddings, checkpoints, or source datasets.
Reproducibility checks
Before release, the following checks should pass:
python -m compileall .\scripts .\src
python -m pytest
python .\scripts\39_finalize_experiment_evidence.py
python .\scripts\42_run_twibot20_overlap_sensitivity.py
python .\scripts\43_regenerate_twibot20_disjoint_figure.py

Exact numerical reproduction requires the same authorized source files, software versions, official splits, model assets, and CUDA-compatible environment.
Interpretation constraints
This repository does not support claims that:
- graph evidence always improves metadata-based detection;
- the framework is universally noise resistant;
- missingness is always random;
- calibrated probabilities imply superior discrimination;
- target-validation adaptation is equivalent to strict zero-shot transfer; or
- third-party data may be freely redistributed.
Natural-distribution performance, pattern-label-standardized performance, calibration, representation corruption, missingness stress, and cross-dataset transfer are reported as distinct validity dimensions.
Citation
Citation metadata are provided in [CITATION.cff](CITATION.cff). The exact v1.1.0 release is archived at https://doi.org/10.5281/zenodo.24199691. The concept DOI https://doi.org/10.5281/zenodo.24152213 resolves to the latest archived version.
License
Original code and documentation are released under the MIT License.

The MIT license does not apply to TwiBot datasets, Twitter/X content, pretrained models, or other third-party resources. Users are responsible for complying with the corresponding licenses, access requirements, and platform policies.
Responsible use
The software is intended for research on social bot detection, missing-data robustness, and evaluation methodology. Predictions should not be used as the sole basis for account suspension, enforcement, legal action, or consequential decisions about individuals.
