# ZoomGain

**Mining Global-to-Zoom Perception Gaps for Fine-Grained Visual Supervision**

This repository contains the ZoomGain training code and local vLLM evaluation tools.

- `EasyR1-Qwen35/`: bundled training framework.
- `zoomgain/` and `scripts/`: training rewards, data utilities, and launch scripts.
- `eval_ZoomGainBench/`: ZoomGainBench evaluation.
- `eval/`: evaluation on other perception benchmarks.

## Training

Install the training dependencies from `requirements.txt`. Extract the RL dataset's
`images.zip` next to `train.parquet`, so the image files are under `images/`.
Set `BASE_MODEL` to a local Qwen3.5 checkpoint and run the launcher for its size.

```bash
BASE_MODEL=/path/to/Qwen3.5-4B \
TRAIN_PARQUET=/path/to/ZoomGainDataRL/train.parquet \
bash scripts/train_zoomgain_4b.sh
```

Use `scripts/train_zoomgain_9b.sh` for the 9B checkpoint. Set `PREFLIGHT_ONLY=1`
to validate the Parquet and image paths before launching training.

## ZoomGainBench

Install `eval_ZoomGainBench/requirements.txt` and a vLLM version compatible with
your model. Extract the benchmark's `images.zip` next to `annotations.jsonl`,
keeping the `images/` subdirectory.

```bash
MODEL_PATH=/path/to/Qwen3.5-4B \
ANNOTATIONS=/path/to/ZoomGainBench/annotations.jsonl \
OUTPUT_DIR=/path/to/results \
GPUS=0 bash eval_ZoomGainBench/eval.sh
```

The evaluator reads both languages and the evidence bounding box from each
annotation. Set `LANGUAGE=en` or `LANGUAGE=cn` for a single language, or append
`--dry-run` to validate the inputs without starting vLLM.
