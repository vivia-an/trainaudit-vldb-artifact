# Security

- Do not commit API keys. Use `DEEPSEEK_API_KEY` / `OPENAI_API_KEY`.
- YAML configs in this package use `${ENV}` placeholders only.
- Do not commit machine-specific absolute paths (home directories, private drives).
- If publishing from a repository that previously contained plaintext keys in git history, **revoke those keys** and prefer `pack_release.sh` output or a fresh repository without that history.
- Smoke tests and the SQL verifier need no network access.
- During double-blind review, do not report security issues via public issues that deanonymize authors. Use the conference artifact / PC channel if required.
