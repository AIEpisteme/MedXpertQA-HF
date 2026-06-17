# Production Runbook

Date: 2026-06-17

## Prerequisites

Run from `C:\Users\Thomas_Yiu\MedXpertQA\eval`.

```powershell
pip install -r requirements-hf.txt
```

For gated models:

```powershell
$env:HF_TOKEN = "<token from Hugging Face>"
```

Do not commit or paste the token into source files.

## Smoke Run

```powershell
python main.py --provider hf --model EpistemeAI/Reasoning-Medical-27B --dataset medxpertqa_sampled --task text --method zero_shot --prompting-type cot --output-dir hf-dev --max-samples 1 --num-threads 1 --hf-device-map auto --hf-torch-dtype auto --hf-max-new-tokens 512
```

## Longer Run

```powershell
python main.py --provider hf --model EpistemeAI/Reasoning-Medical-27B --dataset medxpertqa --task text --method zero_shot --prompting-type cot --output-dir hf-prod --num-threads 1 --hf-device-map auto --hf-torch-dtype auto --hf-max-new-tokens 4096
```

## Helper Script

With Bash available:

```bash
bash scripts/run_hf.sh EpistemeAI/Reasoning-Medical-27B medxpertqa_sampled text hf-dev zero_shot cot 1 512
```

Optional environment overrides:

- `HF_DEVICE_MAP=auto`
- `HF_TORCH_DTYPE=auto`
- `HF_IMAGE_POLICY=error`

## Output Location

Outputs are written under:

```text
eval/outputs/<output-dir>/<model-id>/<dataset>/<method>/<prompting-type>/
```

For `EpistemeAI/Reasoning-Medical-27B`, the slash in the model ID creates nested directories under `EpistemeAI/Reasoning-Medical-27B`.

## Diagnostics

- Missing `transformers` or `torch`: reinstall `requirements-hf.txt`.
- CUDA or memory failure: reduce `--hf-max-new-tokens`, use a smaller model, change `--hf-device-map`, or use hardware with more VRAM/RAM.
- Gated model access failure: confirm `HF_TOKEN` has access to the model.
- Image-task failure: run `--task text` for text-only models or implement a multimodal HF adapter.
- Custom-code model failure: review the model repository, then retry with `--hf-trust-remote-code` only if acceptable.

## Rollback

Use the unchanged API provider path:

```powershell
python main.py --provider api --model gpt-4o-mini --dataset medxpertqa --task mm --method zero_shot --prompting-type cot --output-dir dev --num-threads 1
```

Delete or ignore partial HF output directories before scoring if a run was interrupted.
