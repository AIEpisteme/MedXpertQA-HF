# Plan

Date: 2026-06-17

## Request

Create a MedXpertQA checkout based on `https://github.com/TsinghuaC3I/MedXpertQA/tree/main` that can run Hugging Face models such as `EpistemeAI/Reasoning-Medical-27B`.

## Scope

- Import the upstream MedXpertQA repository into `C:\Users\Thomas_Yiu\MedXpertQA`.
- Add a Hugging Face provider path for text-only causal language models.
- Preserve existing API-backed inference behavior.
- Add commands and documentation for installing dependencies and running a text split.
- Add lightweight tests that do not require downloading a large model.

## Non-Goals

- Do not implement multimodal Hugging Face image inference in this change.
- Do not download or execute `EpistemeAI/Reasoning-Medical-27B` during local validation.
- Do not modify MedXpertQA scoring logic beyond provider support.

## Milestones

1. Inspect upstream runner and input/output contracts.
2. Implement `HuggingFaceAgent` behind `--provider hf`.
3. Add HF requirements, helper script, README usage, and delivery docs.
4. Validate Python syntax and adapter unit tests.
5. Record any validation gaps and runtime prerequisites.

## Risks and Mitigations

- Large model resource usage: document GPU/VRAM expectations and provide `--max-samples` smoke runs.
- Text-only model on MM task: default to rejecting images unless explicitly overridden.
- Secret leakage: read Hugging Face tokens from environment variables only.
- Remote custom code: keep `trust_remote_code` disabled unless the operator explicitly enables it.

## Completion Criteria

- `eval/main.py --provider hf` can construct the HF agent with configurable model-loading options.
- Existing `--provider api` model validation remains in place.
- README and runbook include exact commands for the example model.
- Local syntax and unit validation pass without requiring network model downloads.
