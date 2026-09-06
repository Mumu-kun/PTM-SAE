# Project Guidelines & Coding Standards (ptm-sae-engine)

## 1. Visual Code Organization & Locality (No Artificial Subfunction Splitting)
- **High Locality**: Avoid fracturing a cohesive, linear procedure into dozens of artificial 3-line subfunctions. Keep the pipeline in one readable flow.
- **Flatten Nested Indentation**: Use early guard clauses (`if not condition: continue` or early `return`) to keep the primary execution path flat at indentation level 1.
- **Table-Driven Logic**: Replace cascading `if/elif/else` ladders with compact, aligned lookup tables (tuples or dictionaries) declared near the top of the logic.
- **Natural Paragraph Cadence**: Group related lines into visual paragraphs separated by blank lines and simple, plain comments (e.g. `# 1. Parse header`). Do **not** use heavy or overdesigned ASCII art banners (`=== [1] ===`).

## 2. Git & Commit Policy
- **Verify Before Committing**: Never make speculative or intermediate commits. Run the full test suite (`uv run pytest tests/ -v`) and ensure all checks pass before creating or amending a commit.
- **Atomic Commits**: Keep the Git history clean and free of dead/temporary code or churn.

## 3. Toolchain & Environment
- Always run Python commands via `uv run` inside the project virtual environment.
- On Windows, respect the sandbox/junction mount by using elevated execution permissions when required.
