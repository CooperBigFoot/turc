# CLAUDE.md — Development Guidelines

This document defines the **mandatory practices** for Python development in this repository. These are not suggestions — they are rules. Deviations must be justified with strong technical reasons and agreed upon in review.

---

## Python Package Management with `uv`

* **Use `uv` exclusively** for Python package management.
* Do **not** use `pip`, `pip-tools`, `poetry`, or `conda` directly.
* Commands you should know:

  * Install: `uv add <package>`
  * Remove: `uv remove <package>`
  * Sync lockfile: `uv sync`
* Running:

  * Python scripts: `uv run <script>.py`
  * Tools: `uv run pytest`, `uv run ruff`
  * REPL: `uv run python`

---

## Ad-hoc Analyses and One-Time Scripts

* **Use shell heredoc syntax** for one-time data analyses and exploratory work.
* Do **not** create throwaway `.py` files or use alternative shell tools (awk, sed, etc.) for data manipulation.
* Python is more readable, maintainable, and powerful for these tasks.

**Preferred pattern:**

```bash
uv run python3 << 'EOF'
import pandas as pd

# Your analysis code here
df = pd.read_csv('data.csv')
print(df.describe())
EOF
```

**Why this matters:**

* **No file clutter** — no orphaned `temp.py` or `test_script.py` files
* **Self-documenting** — the command and its context live together in shell history or docs
* **Efficient** — Claude can generate complete, working analyses inline
* **Reproducible** — easy to copy-paste entire commands

This approach is **mandatory** for:

* Quick data inspections
* One-time transformations
* Exploratory analyses
* Data quality checks

For **reusable** logic that runs regularly, create proper Python scripts or modules.

---

## CARAVAN Dataset Access

**Location:** `/Users/nicolaslazaro/Desktop/CARAVAN_CLEAN_prod/train`
**Interface:** `from transfer_learning_publication.data import CaravanDataSource`
**Regions:** `camels`, `camelsaus`, `camelsbr`, `camelsch`, `camelscl`, `camelsde`, `camelsgb`, `grdc`, `hysets`, `lamah`

### Initialization

```python
data_path = "/Users/nicolaslazaro/Desktop/CARAVAN_CLEAN_prod/train"
ds = CaravanDataSource(data_path, region="camelsch")  # Single region
ds = CaravanDataSource(data_path, region=["camels", "hysets"])  # Multiple
ds = CaravanDataSource(data_path)  # All regions
```

### Discovery (Lightweight - No Data Loading)

```python
ds.list_regions()                    # Available regions
ds.list_gauge_ids()                  # All gauge IDs
ds.list_timeseries_variables()       # Timeseries columns
ds.list_static_attributes()          # Static attribute columns
ds.get_date_ranges(gauge_ids=[...])  # Date ranges per gauge (LazyFrame)
```

### Lazy Data Access (CRITICAL: Do Not Collect Without Filters)

```python
# Timeseries - ALWAYS filter before collecting
lazy_ts = ds.get_timeseries(
    gauge_ids=["camelsch_2009"],
    columns=["streamflow", "total_precipitation_sum"],
    date_range=("2010-01-01", "2010-12-31")
)

# Static attributes
lazy_attrs = ds.get_static_attributes(
    gauge_ids=["camelsch_2009"],
    columns=["area", "elevation_mean"]
)

# Convert to ML containers
ts_collection = ds.to_time_series_collection(lazy_ts, target_column="streamflow")
attr_collection = ds.to_static_attribute_collection(lazy_attrs)
```

### Safety Rules

1. **NEVER** call `.collect()` without applying `gauge_ids`, `columns`, or `date_range` filters first
2. Start with `region="camelsch"` (smallest region, 288 gauges) and 1-2 gauges
3. Use `.collect_schema()` to inspect LazyFrame structure without loading data
4. Only collect after confirming data size is manageable

---

## Python Coding Style

### Type Hints (mandatory)

* Always annotate function parameters and return types.
* Use built-in generics (`list`, `dict`, `tuple`, `set`) — **never** import `List`, `Dict`, etc. from `typing`.
* Use `|` for unions (Python 3.10+).
* Annotate variables where type is not obvious.

```python
def process_data(items: list[str]) -> dict[str, int]:
    ...

value: str | None = None
```

### Error Handling

* **Never** use bare `except`.
* Always raise meaningful errors with context.
* Prefer explicit error classes over generic `Exception`.

### Logging

* Use `logging` — never `print` — for runtime diagnostics.

### Formatting & Linting

* Use `ruff` for both linting and formatting:

  * Format: `uv run ruff format`
  * Lint + fix: `uv run ruff check --fix`

---

## Testing Philosophy

Good tests do not just check code; they shape its design. Tests are **contracts**: they describe what must stay true even if the implementation changes.

### Golden Rules

1. **Test behavior, not implementation**

   * Assert on outputs and public APIs.
   Do not inspect private attributes like `_steps` unless no public API exists. If needed, add a public `.spec()` for testability.

2. **Each test should fail for one reason**

   * Keep assertions focused. Split broad tests into smaller ones.

3. **Prefer fast, deterministic tests**

   * No `sleep()`; control time with libraries like `freezegun` or dependency injection.
   * Control randomness by seeding or injecting RNGs.

4. **Use fakes over mocks**

   * Fake implementations are easier to read and maintain than heavy mocking.
   * Mock only at external boundaries (HTTP, file I/O, external services).

5. **Structure tests for readability**

   * Setup (Arrange) → Action → Assertion.
   * Use fixtures for repeated setup, but don’t hide complexity in `conftest.py`.

---

## Test Coverage

* Use `pytest-cov` to measure coverage.
* Run with coverage on every commit:

  * `uv run pytest --cov=src/transfer_learning_publication --cov-report=term-missing tests/`
* Coverage should be **used to find gaps**, not chased to 100%. A brittle 100% is worse than 85% meaningful coverage.

---

## Testing Conventions

### File & Class Organization

* **One test file per module**: `test_<module>.py`
* **One test class per function/class under test**: `Test<ThingUnderTest>`
* Test methods: descriptive, snake\_case, explain the behavior.
  Example: `test_fails_with_empty_dataframe`, not `test1`.

### Categories of Tests

1. **Basic functionality**: happy paths with simple inputs.
2. **Error handling**: invalid inputs should raise the right exception with the right message.
3. **Edge cases**: empty data, all-null columns, large inputs, weird types.
4. **Data preservation**: non-transformed columns, schema, and order remain intact.
5. **Integration paths**: small number of tests where the real pipeline runs end-to-end.

### Assert Patterns

* Prefer **direct comparisons** (`equals`, `==`) for DataFrames.
* For complex structures, use `.to_list()` or `.spec()` for clarity.
* Check types and schema explicitly when relevant.

```python
# Good
assert result["column"].to_list() == [1.0, 2.0, 3.0]
assert result.schema["col"] == pl.Float64
```

### Error Testing

Always assert both **exception type** and **message fragment**:

```python
with pytest.raises(ValueError, match="no steps"):
    builder.build()
```

### Fixtures

* Use fixtures sparingly and descriptively (`simple_df`, `df_with_missing_values`).
* Avoid fixture over-engineering; clarity > DRY.

---

## Advanced Testing Guidelines

* **Canonical Specs**: If an object has complex internal state, expose a `.spec()` or `.describe()` method that returns a plain dict/list summary. Test against that instead of private fields.
* **Lazy evaluation**: For Polars LazyFrame functions, assert that results remain lazy (don’t call `.collect()` unless necessary).

---

## Anti-Patterns (Avoid These)

Asserting on private attributes (`._steps`, `._fitted_steps`).
Overly specific error message checks (brittle wording).
Giant integration tests covering all cases — push most variation down into unit tests.
100s of trivial tests (getter/setter, boilerplate) — test behaviors that matter.
Hiding critical setup in nested fixtures.

In short: **tests should describe contracts, not internals.**
If your test breaks after a refactor that doesn’t change behavior, the test was wrong.
