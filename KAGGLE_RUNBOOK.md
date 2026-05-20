# DeepShield Kaggle Runbook

## Kaggle inputs

Upload or attach these datasets to the notebook:

- `deepshield-code`
  - contains `deepshield/`, `scripts/`, `requirements-kaggle.txt`
- `deepshield-assets`
  - contains `model_repos/` and `artifacts/checkpoints/`
- `FaceForensics++ c23`
  - primary for `image` and `video`
- `Celeb-DF v2`
  - cross-dataset clean test for `image`
- `ASVspoof 2019 LA`
  - primary for `audio`
- `ASVspoof 2021 DF`
  - cross-dataset clean test for `audio`
- `ASVspoof 2021 keys`
  - only needed if `trial_metadata.txt` is not bundled inside the DF audio dataset
- `FakeAVCeleb`
  - primary for `multimodal`
- `LAV-DF`
  - cross-dataset clean test for `video` and `multimodal`

## Notebook bootstrap

```bash
CODE=/kaggle/input/deepshield-code
ASSETS=/kaggle/input/<assets-dataset>
FFPP=/kaggle/input/<ffpp-dataset>
CELEBDF=/kaggle/input/<celebdf-dataset>
ASV2019=/kaggle/input/<asvspoof2019-la-dataset>
ASV2021=/kaggle/input/<asvspoof2021-df-dataset>
ASV2021_KEYS=/kaggle/input/<asvspoof2021-keys-dataset>
FAKEAV=/kaggle/input/<fakeavceleb-dataset>
LAVDF=/kaggle/input/<lavdf-dataset>/LAV-DF
RUN=/kaggle/working/deepshield

pip install -r $CODE/requirements-kaggle.txt
```

## 1. Prepare the workspace

```bash
python $CODE/scripts/prepare_kaggle_workspace.py \
  --assets-root $ASSETS \
  --ffpp-root $FFPP \
  --celebdf-root $CELEBDF \
  --asvspoof2019-la-root $ASV2019 \
  --asvspoof2021-df-root $ASV2021 \
  --asvspoof2021-keys-root $ASV2021_KEYS \
  --fakeavceleb-root $FAKEAV \
  --lavdf-root $LAVDF \
  --output-dir $RUN \
  --device cuda
```

This writes:

- `$RUN/kaggle_workspace_summary.json`
- `$RUN/manifests/image_primary_ffpp_c23.csv`
- `$RUN/manifests/image_cross_celebdf_v2.csv`
- `$RUN/manifests/audio_primary_asvspoof2019_la.csv`
- `$RUN/manifests/audio_cross_asvspoof2021_df.csv`
- `$RUN/manifests/video_primary_ffpp_c23.csv`
- `$RUN/manifests/video_cross_lavdf.csv`
- `$RUN/manifests/multimodal_primary_fakeavceleb.csv`
- `$RUN/manifests/multimodal_cross_lavdf.csv`

The summary JSON also stores the correct `data_root` for every manifest.

## 2. Phase 0 clean baselines

### Recommended: run the whole clean matrix

```bash
python $CODE/scripts/run_phase0_clean_matrix.py \
  --assets-root $ASSETS \
  --workspace-summary $RUN/kaggle_workspace_summary.json \
  --output-dir $RUN/results/phase0_clean \
  --device cuda
```

This runs, in order:

- `image_primary_ffpp_c23`
- `image_cross_celebdf_v2`
- `audio_primary_asvspoof2019_la`
- `audio_cross_asvspoof2021_df`
- `video_primary_ffpp_c23`
- `video_cross_lavdf`
- `multimodal_primary_fakeavceleb`
- `multimodal_cross_lavdf`

Each subdirectory under `$RUN/results/phase0_clean/` gets:

- `clean_runs.jsonl`
- `clean_summary.json`
- `clean_summary.csv`

### Optional: run a single clean benchmark

```bash
python $CODE/scripts/run_clean_baselines.py \
  --assets-root $ASSETS \
  --manifest-path $RUN/manifests/image_primary_ffpp_c23.csv \
  --data-root $FFPP \
  --output-dir $RUN/results/clean_image_primary_ffpp \
  --device cuda
```

## 3. Phase 1 attacks on primary datasets

### Image attacks on FF++

```bash
python $CODE/scripts/run_phase1_image.py \
  --assets-root $ASSETS \
  --manifest-path $RUN/manifests/image_primary_ffpp_c23.csv \
  --data-root $FFPP \
  --output-dir $RUN/results/phase1_image \
  --device cuda
```

### Audio attacks on ASVspoof 2019 LA

```bash
python $CODE/scripts/run_phase1_audio.py \
  --assets-root $ASSETS \
  --manifest-path $RUN/manifests/audio_primary_asvspoof2019_la.csv \
  --data-root $ASV2019 \
  --output-dir $RUN/results/phase1_audio \
  --device cuda
```

### Video attacks on FF++ and multimodal attacks on FakeAVCeleb

```bash
python $CODE/scripts/run_phase1_video_multimodal.py \
  --assets-root $ASSETS \
  --video-manifest-path $RUN/manifests/video_primary_ffpp_c23.csv \
  --multimodal-manifest-path $RUN/manifests/multimodal_primary_fakeavceleb.csv \
  --video-data-root $FFPP \
  --multimodal-data-root $FAKEAV \
  --output-dir $RUN/results/phase1_video_multimodal \
  --device cuda
```

## 4. Post-compression realism checks

### Image post-processing on FF++

```bash
python $CODE/scripts/run_phase1_postcompression.py \
  --assets-root $ASSETS \
  --manifest-path $RUN/manifests/image_primary_ffpp_c23.csv \
  --data-root $FFPP \
  --output-dir $RUN/results/postcompression_image \
  --device cuda
```

### Audio post-processing on ASVspoof 2019 LA

```bash
python $CODE/scripts/run_phase1_postcompression.py \
  --assets-root $ASSETS \
  --manifest-path $RUN/manifests/audio_primary_asvspoof2019_la.csv \
  --data-root $ASV2019 \
  --output-dir $RUN/results/postcompression_audio \
  --device cuda
```

## Notes

- `prepare_kaggle_workspace.py` uses the intended dataset mapping instead of routing every modality through `LAV-DF`.
- `ASVspoof 2021 DF` needs `DF/CM/trial_metadata.txt`. Pass `--asvspoof2021-keys-root` when that file lives in a separate Kaggle dataset.
- `LAV-DF` should be mounted at the dataset root that contains `train/`, `dev/`, `test/`, `metadata.json`, and `metadata.min.json`.
- The attack runners enforce:
  - `SSIM >= 0.95` for image/video acceptance
  - `PESQ >= 3.5` for audio acceptance
- `run_phase1_video_multimodal.py` now accepts separate roots because `video` and `multimodal` primary datasets are different.
- If `FakeAVCeleb` metadata does not expose an explicit split column, the manifest builder falls back to a deterministic hash split so the split assignment stays stable across runs.
