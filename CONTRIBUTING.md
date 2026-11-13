# Contributing

## Environment

Install Python 3.10 or newer and [uv](https://docs.astral.sh/uv/). Enable the repository hooks:

```sh
uv sync --locked --extra datasets --extra pipecat --extra server --group dev
uvx pre-commit install
```

## Standards

- Use English (US) in code, comments, documentation, commit messages, and pull request text.
- Use only public or synthetic data in tests and examples.
- Do not include non-public datasets, non-public identifiers, recordings of real people, or secrets.
- Do not mention internal codenames, private repository names, private model repositories, or internal phase labels.
- Keep comments short and limited to explaining why.
- Use Conventional Commit messages in English. Pull requests are squash-merged.
- Run `python3 scripts/check_public.py` before submitting; it checks tracked files using locally configured terms without adding those terms to the repository.

## How to contribute

Run the same checks as CI:

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q
uv build
```

Real-model tests download weights and are excluded by default; run them with
`uv run pytest -m real_model`.

Before opening a pull request, also run `uvx pre-commit run --all-files` and confirm that examples use only public or synthetic data.
