# Requirements

Date: 2026-06-17

## Actors

- Evaluator: runs MedXpertQA inference from `eval/`.
- Operator: prepares Python dependencies, Hugging Face credentials, and model hardware.

## Functional Requirements

- The CLI must support `--provider api` for existing API models and `--provider hf` for Hugging Face causal LMs.
- The HF provider must accept arbitrary model IDs such as `EpistemeAI/Reasoning-Medical-27B`.
- The HF provider must reuse the existing MedXpertQA JSONL input/output flow and `LLMAgent.get_response(messages)` contract.
- The HF provider must apply tokenizer chat templates when available and fall back to a role-tagged plain prompt.
- The HF provider must expose generation/model-loading controls for device map, dtype, max new tokens, max input tokens, revision, cache dir, local-files-only mode, attention implementation, seed, and remote-code trust.
- The helper script must run text-task inference in the foreground so model-load failures are visible.

## Security and Reliability Requirements

- Hugging Face tokens must come from environment variables and must not be printed or persisted by the code.
- `trust_remote_code` must be opt-in because model repositories can execute custom Python code.
- Text-only HF inference must reject image inputs by default to avoid invalid MM benchmark results.
- Local HF inference must run with one worker thread by default to avoid unsafe concurrent generation on one model instance.
- Error messages must identify configuration/runtime problems without exposing secrets.

## Acceptance Criteria

- `python -m py_compile main.py setup.py model/hf_agent.py` passes from `eval/`.
- `python -m unittest discover -s tests` passes from `eval/`.
- `README.md` documents HF install and example inference commands.
- `delivery/production-runbook.md` documents run, verification, and rollback procedures.
