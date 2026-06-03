# DeepShield

DeepShield is a comprehensive evaluation and execution framework for state-of-the-art Multi-Modal Deepfake Detection models. It serves as a master repository and testing ground for various sophisticated architectures designed to detect forged audio-visual content.

## Project Structure

- **`CMAR/`**: Integration and execution scripts for the Cross-Modal Attention Representation models.
- **`ROBUSTAV/`**: Integration for the Robust Audio-Visual deepfake detection frameworks.
- **`model_repos/`**: Centralized storage for model checkpoints and pre-trained weights.
- **`scripts/`**: Automation scripts for training, testing, and dataset ingestion.

## Kaggle Integration

DeepShield is explicitly designed to run seamlessly in Kaggle environments to leverage free GPU accelerators.
- See `KAGGLE_BEGINNER_GUIDE.md` for instructions on setting up Kaggle notebooks.
- See `KAGGLE_RUNBOOK.md` for standard operating procedures during training runs.
- `requirements-kaggle.txt` contains the specific pip dependencies needed for the Kaggle environment.

## Goal

To provide a standardized pipeline for benchmarking cutting-edge multi-modal deepfake detectors on extensive datasets, ensuring robust and reproducible results across various hardware setups.
