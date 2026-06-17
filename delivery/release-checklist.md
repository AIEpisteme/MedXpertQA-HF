# Release Checklist

Date: 2026-06-17

## Pre-Release Gates

- [ ] Install API dependencies with `pip install -r requirements.txt`.
- [ ] Install HF dependencies with `pip install -r requirements-hf.txt`.
- [ ] Confirm GPU/CPU memory is sufficient for the selected model.
- [ ] Set `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` only if the selected model requires it.
- [ ] Run `python -m py_compile main.py setup.py model/hf_agent.py` from `eval/`.
- [ ] Run `python -m unittest discover -s tests` from `eval/`.
- [ ] Run a one-sample `medxpertqa_sampled` text smoke test.

## Rollout

1. Start with `--max-samples 1` on `medxpertqa_sampled --task text`.
2. Inspect the generated JSONL record for response shape and answer parsing.
3. Increase `--max-samples` gradually.
4. Move from sampled to full `medxpertqa --task text` only after resource use is stable.

## Rollback Triggers

- Model load fails due to dependency, CUDA, memory, or access errors.
- Output files contain malformed responses or missing required fields.
- Logs or artifacts contain sensitive tokens.
- Image/MM tasks are accidentally run with the text-only HF provider.

## Rollback Steps

- Stop the active run.
- Keep existing API workflow by using `--provider api` or `bash scripts/run.sh`.
- Remove incomplete HF output directories under `eval/outputs/<output_dir>/...` if they should not be scored.
- Re-run the last known-good API command or smaller HF smoke command.

## Post-Release Verification

- Confirm output JSONL count matches requested sample count.
- Confirm `correct` values are populated.
- Confirm logs show no secret values.
- Confirm operator notes include model ID, revision if used, dtype, device map, and sample count.
