# Design

Date: 2026-06-17

## Components

- `eval/main.py`: owns CLI parsing, provider-specific option collection, and task execution.
- `eval/setup.py`: validates provider/model selection and returns an `LLMAgent`.
- `eval/model/hf_agent.py`: implements the Hugging Face text-generation adapter.
- `eval/scripts/run_hf.sh`: provides a foreground run path for local HF models.
- `eval/requirements-hf.txt`: installs optional HF dependencies separately from API-only dependencies.

## Data Flow

1. `main.py` parses `--provider hf` and HF options.
2. `setup.py` creates `HuggingFaceAgent(model_id, ...)` without requiring the model ID in `model_info.json`.
3. Existing prompt builders create OpenAI-style chat messages.
4. `HuggingFaceAgent` converts text message parts into plain chat messages.
5. The tokenizer renders the prompt with `apply_chat_template(..., add_generation_prompt=True)` when available.
6. `AutoModelForCausalLM.generate()` creates output text.
7. Existing MedXpertQA answer cleansing, correctness, and JSONL output logic remains unchanged.

## Trust Boundaries

- Dataset samples are untrusted input and are converted to text without command execution.
- Hugging Face model repositories are external dependencies. Custom model code runs only when `--hf-trust-remote-code` is provided.
- Hugging Face tokens are secrets and remain in process environment only.
- Generated model output is untrusted and is passed only to existing answer parsing logic.

## Failure Behavior

- Missing HF dependencies raise a clear runtime error pointing to `eval/requirements-hf.txt`.
- Image inputs fail by default with an explanation that the generic HF agent is text-only.
- Multi-threaded HF requests are forced to one thread in `main.py`.
- Unsupported dtype or image policy values raise validation errors before generation.

## Rollout and Rollback

- Rollout: install `requirements-hf.txt`, run a small `medxpertqa_sampled` text smoke test, then scale sample count/model size.
- Rollback: use `--provider api` and existing `scripts/run.sh`, or revert the HF-specific files and wiring.
