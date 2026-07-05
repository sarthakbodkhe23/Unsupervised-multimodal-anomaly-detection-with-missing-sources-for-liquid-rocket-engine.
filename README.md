# Unsupervised-multimodal-anomaly-detection-with-missing-sources-for-liquid-rocket-engine.

UU-Net: A Dual-Stage Autoencoder for Multi-Sensor Anomaly Detection under Missing Modalities

Deep Learning ,PyTorch ,Autoencoders ,Multimodal Fusion, Time Series, GAF.

# Executive Summary

Real engineering systems are monitored by dozens of sensors, and in practice some of those sensors regularly drop out — due to failure, communication loss, or maintenance. This project builds an unsupervised anomaly detection pipeline (UU-Net) that stays robust even when a random subset of sensor channels is missing at inference time.

The pipeline converts raw multivariate sensor time series into 2-D images using the Gramian Angular Field (GAF) transform, then reconstructs them through a two-stage cascaded autoencoder that fuses information first within a sensor modality and then across modalities. Anomalies are flagged as time windows the model struggles to reconstruct.

The project was built and evaluated on the NASA C-MAPSS turbofan engine degradation dataset, a standard benchmark for prognostics and health management (PHM), as a proxy multi-sensor system for studying missing-source robustness.

#Why We Built This

Most anomaly detection systems assume every sensor is always available — an assumption that rarely holds in the field. We wanted to:


-Study how a multimodal autoencoder can be trained only on unsupervised, healthy-operation data (no anomaly labels required)
-Simulate and explicitly handle randomly missing sensor sources rather than ignoring the problem
-Represent 1-D sensor signals in a form a CNN can process, using the GAF image transform
-Practice building a full deep-learning pipeline: data loading → representation learning → dual-stage reconstruction → loss design → evaluation
