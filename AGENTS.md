# Repository Guidelines

## Project Structure & Module Organization

The dependency-free Rust limiter core lives in `src/lib.rs`, while gain-envelope primitives are implemented in `src/envelope.rs`. Optional PyO3/NumPy bindings are isolated in `src/python.rs` behind the `python` Cargo feature. Rust integration tests belong in `tests/*.rs`; Python binding tests belong in `tests/test_*.py`. Keep the public Python type declarations in `sf_limiter.pyi` synchronized with binding changes. Package metadata is split between `Cargo.toml` and `pyproject.toml`, and dependency locks are committed in `Cargo.lock` and `uv.lock`. Treat `target/`, `.venv/`, and `dist/` as generated output.

## Build, Test, and Development Commands

- `cargo test`: build the default Rust crate and run all Rust tests.
- `cargo fmt -- --check`: verify standard Rust formatting; run `cargo fmt` to apply it.
- `cargo clippy --all-targets -- -D warnings`: lint the Rust core and tests.
- `uv sync`: create the Python 3.11 environment, build the extension, and install development dependencies.
- `uv run pytest -q`: run the Python binding suite.
- `uv run ruff check .` and `uv run ruff format --check .`: lint and check formatting for Python files.

Run `uv sync` before Python tests whenever Rust bindings or package metadata change.

## Coding Style & Naming Conventions

Use rustfmt defaults (four-space indentation) and keep the crate free of unsafe Rust, as enforced by `Cargo.toml`. Follow Rust conventions: `snake_case` functions/modules, `CamelCase` types, and descriptive error variants. Python uses four spaces, `snake_case`, type annotations, and Ruff formatting. Keep the audio core independent of Python-specific types; binding conversion and validation belong in `src/python.rs`.

## Testing Guidelines

Add regression tests with every behavior change. Name Rust tests descriptively with `#[test]`; name Python tests `test_<behavior>`. Cover mono and multichannel shapes, invalid inputs, finite output, and the hard-ceiling guarantee. There is no numeric coverage threshold, so prioritize boundary conditions and assertions that samples never exceed the configured limit.

## Commit & Pull Request Guidelines

Follow the existing history: short, lowercase, imperative summaries such as `add python type stub`. Keep each commit focused. Pull requests should explain the user-visible behavior, list the Rust and Python commands run, and call out API or packaging changes. Link relevant issues; include before/after output for numerical behavior changes. Update `README.md` and `sf_limiter.pyi` when public APIs change.
