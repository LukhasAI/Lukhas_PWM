# LUKHAS AGI - Testing Strategy

This document outlines the testing strategy for the LUKHAS AGI repository.

## General Principles

*   All new code should be accompanied by tests.
*   Tests should be independent and not rely on the state of other tests.
*   Tests should be fast and run in a consistent environment.
*   Tests should be organized in a way that reflects the structure of the codebase.

## Running Tests

To run all tests, use the following command from the root of the repository:

```bash
python3 -m unittest discover tests
```

## Import Strategy

Due to the complex structure of the repository, there have been issues with imports in the past. The following strategy should be used to ensure that tests can find the modules they need to import:

*   **Avoid `sys.path` modifications.** Modifying `sys.path` in tests is not a reliable long-term solution and should be avoided.
*   **Use absolute imports.** Whenever possible, use absolute imports from the root of the repository. For example, to import the `BioOrchestrator` class, use `from orchestration.orchestrator import BioOrchestrator`.
*   **Create `__init__.py` files.** To make a directory a package, create an empty `__init__.py` file in it. This will allow you to use absolute imports from that directory.
*   **Consider a symbolic path utility.** If the import issues persist, consider creating a symbolic path utility to resolve paths. This utility could be placed in a `common/` or `utils/` directory.

## Symbolic Imports

All symbolic imports (functions and classes with `ΛTAG`s) should be testable. This means that they should be imported into a test file and called with appropriate arguments. The test should verify that the function or class behaves as expected.
