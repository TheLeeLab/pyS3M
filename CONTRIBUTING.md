We welcome patches from everyone. Please submit them as pull requests on github.
Here are few guidelines to take into account.

* If you are submitting a patch that fixes a specific bug, please link the github issue if there is one. If there is no reported bug, please give enough information to reproduce it.

* If you are thinking about a new feature, it is a good idea to open an issue about it first (before coding) to discuss it, see whether it is a good idea, and what is the best way to implement it.

* If you are submitting a code change that you think provides an improvement, we would expect to see a test case that would demonstrate that your patch improves the situation (i.e. performance/accuracy etc).

## Running the tests

`src/` (excluding the GUI) is at 100% line coverage — patches touching `src/` are expected to keep it there.

```bash
pip install -e ".[dev]"
pytest unit_tests/ --ignore=unit_tests/claude
```

To check coverage on a specific file:

```bash
pytest unit_tests/ --ignore=unit_tests/claude --cov=src --cov-report=term-missing
```

`unit_tests/claude/` holds scratch/exploratory scripts and isn't part of the maintained suite.