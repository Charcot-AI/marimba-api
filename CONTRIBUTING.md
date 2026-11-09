# Contributing

## Environment

Install Python 3.10 or newer and [uv](https://docs.astral.sh/uv/). Enable the repository hooks:

```sh
uv venv
uvx pre-commit install
```

The Python package and project configuration will be added in a later pull request.

## Standards

- Use English (US) in code, comments, documentation, commit messages, and pull request text.
- Use only public or synthetic data in tests and examples.
- Do not include non-public datasets, non-public identifiers, recordings of real people, or secrets.
- Do not mention internal codenames, private repository names, private model repositories, or internal phase labels.
- Keep comments short and limited to explaining why.
- Use Conventional Commit messages in English. Pull requests are squash-merged.
- Run `python3 scripts/check_public.py` before submitting; it checks tracked files using locally configured terms without adding those terms to the repository.

## How to contribute

When the Python package is added, prepare the project environment and run:

```sh
uv run --with ruff ruff check .
uv run --with ruff ruff format --check .
uv run --with pytest pytest
```

Before opening a pull request, also run `uvx pre-commit run --all-files` and confirm that examples use only public or synthetic data.
