# Unsupervised-multimodal-anomaly-detection-with-missing-sources-for-liquid-rocket-engine.

UU-Net: A Dual-Stage Autoencoder for Multi-Sensor Anomaly Detection under Missing Modalities

Deep Learning ,PyTorch ,Autoencoders ,Multimodal Fusion, Time Series, GAF.

# Executive Summary

Real engineering systems are monitored by dozens of sensors, and in practice some of those sensors regularly drop out — due to failure, communication loss, or maintenance. This project builds an unsupervised anomaly detection pipeline (UU-Net) that stays robust even when a random subset of sensor channels is missing at inference time.

The pipeline converts raw multivariate sensor time series into 2-D images using the Gramian Angular Field (GAF) transform, then reconstructs them through a two-stage cascaded autoencoder that fuses information first within a sensor modality and then across modalities. Anomalies are flagged as time windows the model struggles to reconstruct.

The project was built and evaluated on the NASA C-MAPSS turbofan engine degradation dataset, a standard benchmark for prognostics and health management (PHM), as a proxy multi-sensor system for studying missing-source robustness.

# Problem Statement

Detect anomalous (degrading/failing) operating states in a multi-sensor system in an unsupervised setting, using only healthy-state data for training, while remaining robust to the random unavailability of individual sensor channels at both training and inference time.

# Why We Built This

Most anomaly detection systems assume every sensor is always available — an assumption that rarely holds in the field. We wanted to:


- Study how a multimodal autoencoder can be trained only on unsupervised, healthy-operation data (no anomaly labels required)
- Simulate and explicitly handle randomly missing sensor sources rather than ignoring the problem
- Represent 1-D sensor signals in a form a CNN can process, using the GAF image transform
- Practice building a full deep-learning pipeline: data loading → representation learning → dual-stage reconstruction → loss design → evaluation

# Dataset Overview

- Dataset: NASA C-MAPSS Turbofan Engine Degradation Simulation (subset FD001)
- Format: Whitespace-separated .txt files, no header — 26 raw columns (unit ID, cycle number, 3 operational settings, 21 sensor readings s1–s21)
- Granularity: Per-cycle readings for 100 engine units, from healthy operation through to failure
- Files used: train_FD001.txt, test_FD001.txt, RUL_FD001.txt

# Sensor preprocessing


- 7 near-constant sensors dropped (s1, s5, s6, s10, s16, s18, s19) — near-zero variance under this operating regime, would dilute model attention
- 14 remaining sensors grouped into 3 physical modalities:
| Modality | Sensors | Description |
|---|---|---|
| Temperature | s2, s3, s4, s11 | Fan/core inlet & outlet temperatures |
| Pressure | s7, s8, s12, s13, s15 | Total/static pressure at various engine stages |
| Mechanical | s9, s14, s17, s20, s21 | Shaft speeds, bleed enthalpy, efficiency ratios |



- Per-sensor Min-Max normalisation to [0, 1], since raw units differ vastly (°R, psia, rpm)
- Only the first 50 cycles per engine are used for training, treated as the healthy/normal regime (validated against the dataset's minimum engine lifetime of ~120 cycles)


Tools Used


Python — data loading, preprocessing, model implementation
PyTorch — UU-Net architecture, training loop, autograd
pyts (GramianAngularField) — 1-D → 2-D time series transformation
NumPy / pandas — array handling, missing-mask simulation, sensor parsing
Matplotlib / Seaborn — sensor degradation plots, lifetime distributions, GAF visualisations


Method

1. Time-series → image representation (GAF)

Each 32-cycle sliding window per sensor is rescaled to [-1, 1], angularly encoded (ϕ = arccos(x)), and converted into a 32×32 Gramian Angular Summation Field (GASF) image:

G(i, j) = cos(ϕi + ϕj)

This turns a noisy 1-D signal into a 2-D image where periodic behaviour appears as diagonal banding and degradation appears as a corner-to-corner colour gradient — patterns a CNN can learn directly.

2. Simulating missing sensors

Every training batch samples a fresh binary missing mask over the 14 sensors (p(missing) = 0.2, ~2.8 sensors masked on average), forcing the model to learn to operate without full sensor coverage rather than merely memorizing all-channel patterns.

3. UU-Net architecture (two cascaded autoencoders)

StageWhat happensPer-source encodingEach of the 14 sensors has its own dedicated CNN encoder → 64-d latent vectorIntramodality fusionMissing sensors' latents are imputed as the mean of available latents (z̄ = 1/K Σ zᵢ)First reconstruction (Module 1)Per-sensor decoders reconstruct each GAF image from the fused latentIntermodality fusion (Module 2, Skip-AE)A second encoder-decoder network processes all 14 reconstructed channels jointly, learning cross-sensor correlations

The encoder/decoder backbones use strided convolutions / transposed convolutions (kernel 4, stride 2) with BatchNorm and LeakyReLU, progressively downsampling 32×32 → 4×4 and back.

4. Loss & optimisation


Reconstruction loss (Lr1): Smooth L1 (Huber) between each per-sensor reconstruction and its input GAF image
Reconstruction loss (Lr2): Smooth L1 between the second-stage joint reconstruction and the concatenated 14-channel input
Optimiser: Adam (lr = 1e-3, β₁ = 0.9, β₂ = 0.999)
The full UU-Net formulation also defines a discrepancy loss term (S = Lr1 + Lr2 − λ·Dz) that widens the gap between the two latent spaces for anomalous inputs, amplifying the anomaly score beyond reconstruction error alone


5. Anomaly scoring

At inference, the combined reconstruction error across both stages serves as the anomaly score; a threshold is set from the reconstruction-error distribution on healthy training data using Kernel Density Estimation (KDE).

Key Insights from Visualisation


Individual sensor degradation is subtle and noisy early on but shows visible drift (e.g. sensors s11, s20) approaching failure — justifying the healthy-window training strategy
Engine lifetimes across the fleet are right-skewed, averaging ~206 cycles, confirming that a fixed run-to-failure assumption would be inaccurate
The 1-D→2-D GAF transform visibly preserves temporal structure: the main diagonal reflects raw signal values, while off-diagonal regions capture temporal correlations that highlight anomalous, irregular patterns


Repository Structure

├── data/                 # C-MAPSS raw data (train/test/RUL files, per FD subset)
├── dashboard/            # Visualisation / results dashboard
├── notebook/             # Main implementation notebook (UU-Net pipeline)
└── README.md

Team

Machine Learning and Deep Learning Lab (AM620L) — Department of Applied Mathematics
Defence Institute of Advanced Technology (DIAT), Pune


Tuhinsubra Sen
Sarthak Bodkhe
Niyati Patil
Dev Solanki


Notes & Limitations


The discrepancy loss term from the original UU-Net formulation is simplified in this implementation (loss = Lr1 + Lr2) to avoid latent-dimension mismatch during execution; the structural design otherwise follows the two-stage cascaded reconstruction described above.
This project uses the C-MAPSS turbofan dataset as a general-purpose multi-sensor benchmark for studying missing-source robustness in unsupervised anomaly detection.
