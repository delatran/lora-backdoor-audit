# Project Delta

The parent `AGENTS.md` is authoritative. This file adds only the following
project-specific constraints:

- Keep private trigger literals and generated backdoored adapters under ignored
  `private/` or `artifacts/private/` paths.
- Synthetic fixture outputs must contain `fixture_smoke_only: true` and must not
  be copied into thesis result tables.
- Never relax the strict capability-and-memory preflight or the validation-only
  threshold lock to make a readiness gate pass.
- Public code, documentation, schemas, and comments use neutral English.
