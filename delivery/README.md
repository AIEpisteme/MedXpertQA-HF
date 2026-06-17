# MedXpertQA Hugging Face Delivery Notes

Date: 2026-06-17

Focus: add a Hugging Face causal-language-model provider to the upstream MedXpertQA evaluation runner while preserving existing API-backed workflows.

Artifacts:

- `plan.md` tracks scope, milestones, risks, and completion criteria.
- `requirements.md` defines testable behavior and security expectations.
- `design.md` describes the provider wiring, model-loading boundary, and failure behavior.
- `test-plan.md` maps requirements to validation evidence.
- `release-checklist.md` captures setup, rollout, rollback, and post-change checks.
- `production-runbook.md` gives exact run, diagnostic, and recovery commands.

Handoff rule: keep these files synchronized with changes to `eval/main.py`, `eval/setup.py`, `eval/model/hf_agent.py`, `eval/requirements-hf.txt`, and `eval/scripts/run_hf.sh`.
