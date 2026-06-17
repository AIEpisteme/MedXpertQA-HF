# Test Plan

Date: 2026-06-17

## Automated Checks

Run from `C:\Users\Thomas_Yiu\MedXpertQA\eval`:

```powershell
python -m py_compile main.py setup.py model/hf_agent.py
python -m unittest discover -s tests
```

These checks validate import-time safety, CLI/provider syntax, and adapter message handling without downloading a model.

Validation result on 2026-06-17:

- Passed: `python -m py_compile main.py setup.py model/hf_agent.py`
- Passed: `python -m unittest discover -s tests`
- Passed: `python main.py --help`

## HF Runtime Smoke Check

After installing HF dependencies and preparing hardware:

```powershell
python main.py --provider hf --model EpistemeAI/Reasoning-Medical-27B --dataset medxpertqa_sampled --task text --method zero_shot --prompting-type cot --output-dir hf-dev --max-samples 1 --num-threads 1 --hf-device-map auto --hf-torch-dtype auto --hf-max-new-tokens 512
```

Expected result:

- A JSONL output file is written under `eval/outputs/hf-dev/EpistemeAI/Reasoning-Medical-27B/medxpertqa_sampled/zero_shot/cot/`.
- Each record includes `response`, `prediction`, `correct`, and sanitized `messages`.

## Security-Focused Validation

- Confirm no token appears in stdout, logs, or JSONL output.
- Confirm MM/image input fails by default for text-only HF inference.
- Confirm enabling `--hf-trust-remote-code` is explicit in the command line.

## Validation Gaps

- The full 27B model was not downloaded or executed during local implementation validation.
- The current Python environment has `transformers` and `accelerate`, but not `torch`; install `eval/requirements-hf.txt` before running HF inference.
- Full benchmark scoring still requires operator-provided GPU resources and model access.
