# Kaggle Beginner Guide

## What to upload to Kaggle

Use **two Kaggle datasets** for your own project files:

### 1. `deepshield-assets`

Make **one zip file** named something like `deepshield-assets.zip`.

That zip should contain exactly:

```text
model_repos/
artifacts/
  checkpoints/
```

Important:

- `model_repos/` must be at the top level of the zip
- `artifacts/checkpoints/` must also be at the top level of the zip
- do **not** zip the whole repo
- do **not** include `datasets information/`
- do **not** include `scripts/`
- do **not** include `deepshield/`

### 2. `deepshield-code`

Upload these as normal files or folders, not as a large zip:

```text
deepshield/
scripts/
requirements-kaggle.txt
KAGGLE_RUNBOOK.md
KAGGLE_BEGINNER_GUIDE.md
Project-Manager-planning.html
final-idea.html
```

This dataset is small. Keeping it unzipped makes it much easier to run scripts directly from Kaggle.

## What not to upload as your own dataset

Do not upload `datasets information/` to Kaggle for execution.

That folder is only for reference. The real media datasets should be attached separately from Kaggle:

- `FaceForensics++ c23`
- `Celeb-DF v2`
- `ASVspoof 2019 LA`
- `ASVspoof 2021 DF`
- `ASVspoof 2021 keys` if needed
- `FakeAVCeleb`
- `LAV-DF`

## What to run locally

Only these things:

1. Create `deepshield-assets.zip`
2. Upload `deepshield-assets.zip` as a Kaggle dataset
3. Upload the code files as a second Kaggle dataset

You do **not** need to run the experiment scripts locally.

## What to run on Kaggle

Run everything below on Kaggle:

1. `scripts/kaggle_prepare_assets.py`
2. `scripts/prepare_kaggle_workspace.py`
3. `scripts/run_phase0_clean_matrix.py`
4. `scripts/run_phase1_image.py`
5. `scripts/run_phase1_audio.py`
6. `scripts/run_phase1_video_multimodal.py`
7. `scripts/run_phase1_postcompression.py`

## Exact notebook cells

### Cell 1: set dataset paths

Edit only the dataset names inside angle brackets.

```python
CODE = "/kaggle/input/<your-deepshield-code-dataset>"
ASSET_ZIP = "/kaggle/input/<your-deepshield-assets-dataset>/deepshield-assets.zip"

FFPP = "/kaggle/input/<ffpp-dataset>"
CELEBDF = "/kaggle/input/<celebdf-dataset>"
ASV2019 = "/kaggle/input/<asvspoof2019-la-dataset>"
ASV2021 = "/kaggle/input/<asvspoof2021-df-dataset>"
ASV2021_KEYS = "/kaggle/input/<asvspoof2021-keys-dataset>"
FAKEAV = "/kaggle/input/<fakeavceleb-dataset>"
LAVDF = "/kaggle/input/<lavdf-dataset>/LAV-DF"

ASSETS = "/kaggle/working/assets"
RUN = "/kaggle/working/deepshield"
```

### Cell 2: install dependencies

```bash
pip install -r $CODE/requirements-kaggle.txt
```

### Cell 3: unzip the asset bundle

```bash
python $CODE/scripts/kaggle_prepare_assets.py \
  --bundle-path $ASSET_ZIP \
  --output-dir $ASSETS
```

### Cell 4: build manifests and workspace summary

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

### Cell 5: Phase 0 clean baselines

```bash
python $CODE/scripts/run_phase0_clean_matrix.py \
  --assets-root $ASSETS \
  --workspace-summary $RUN/kaggle_workspace_summary.json \
  --output-dir $RUN/results/phase0_clean \
  --device cuda
```

### Cell 6: Phase 1 image

```bash
python $CODE/scripts/run_phase1_image.py \
  --assets-root $ASSETS \
  --manifest-path $RUN/manifests/image_primary_ffpp_c23.csv \
  --data-root $FFPP \
  --output-dir $RUN/results/phase1_image \
  --device cuda
```

### Cell 7: Phase 1 audio

```bash
python $CODE/scripts/run_phase1_audio.py \
  --assets-root $ASSETS \
  --manifest-path $RUN/manifests/audio_primary_asvspoof2019_la.csv \
  --data-root $ASV2019 \
  --output-dir $RUN/results/phase1_audio \
  --device cuda
```

### Cell 8: Phase 1 video + multimodal

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

### Cell 9: post-compression checks

Image:

```bash
python $CODE/scripts/run_phase1_postcompression.py \
  --assets-root $ASSETS \
  --manifest-path $RUN/manifests/image_primary_ffpp_c23.csv \
  --data-root $FFPP \
  --output-dir $RUN/results/postcompression_image \
  --device cuda
```

Audio:

```bash
python $CODE/scripts/run_phase1_postcompression.py \
  --assets-root $ASSETS \
  --manifest-path $RUN/manifests/audio_primary_asvspoof2019_la.csv \
  --data-root $ASV2019 \
  --output-dir $RUN/results/postcompression_audio \
  --device cuda
```

## The only paths you normally change

In practice, you usually change only:

- `CODE`
- `ASSET_ZIP`
- `FFPP`
- `CELEBDF`
- `ASV2019`
- `ASV2021`
- `ASV2021_KEYS`
- `FAKEAV`
- `LAVDF`

Everything else can stay the same.
