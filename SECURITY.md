# Security policy

## Reporting a vulnerability

Please report potential security or privacy issues privately to the repository owner rather than opening a public issue. Include a minimal reproduction and avoid including secrets, task payloads, account data, provider responses, or benchmark artifacts.

## Operational rules

- Never commit `.env` files, API keys, tokens, benchmark account data, run artifacts, journals, or model prompts/responses.
- Keep provider keys in the process environment or an external secret manager.
- Treat benchmark data and evaluation artifacts as separate from this source distribution.
- Verify the frozen registry, retrieval provenance, and admission receipts before injecting guidance into a model prompt.
