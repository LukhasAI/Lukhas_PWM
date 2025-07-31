# README for core/common/

## Overview

The `core/common/` directory is a foundational part of the LUKHAS AI system. It centralizes base constants, enumerations (enums), shared utility functions, and common data types that are frequently imported and reused across multiple modules within the `core` and potentially other higher-level systems.

The primary goal of this directory is to promote consistency, reduce redundancy, and improve the maintainability of the codebase by providing a single source of truth for these widely used elements.

## Symbolic Notes & Conventions

-   **`# ΛCONSTANT`**: Applied to all global constants and hyperparameters defined here. These are typically values that configure system-wide behaviors or provide fixed reference points.
-   **`# ΛSEED`**: Used for constants that act as seeds for processes like pseudo-random number generation, ensuring reproducibility or controlled initialization.
-   **`# ΛNOTE`**: Provides explanatory context for constants, enums, or utilities, clarifying their purpose or symbolic meaning.
-   **`# ΛCAUTION`**: Flags constants or configurations that are sensitive and could have significant impact if changed without thorough understanding (e.g., default timeouts, retry limits, critical file paths).
-   **`# ΛSHARED`**: This tag explicitly marks constants, enums, or utilities within this module that are designed and expected to be imported and used by many other modules across the LUKHAS system. This highlights their broad scope and importance.
-   **Headers/Footers**: All Python files must include the standardized header and footer blocks.
-   **Logging**: Basic `structlog` initialization is present. Shared utilities defined here should also use `structlog` for any necessary logging, tagged with `# ΛTRACE`.

## Key Modules and Functionality

-   **`__init__.py`**:
    -   Serves as the main file for defining and exposing common elements.
    -   Initializes a `structlog` logger for the `core.common` module.
    -   **Constants**: Defines system-wide constants such as default timeouts (`DEFAULT_NETWORK_TIMEOUT_SECONDS`), retry counts (`MAX_TRANSIENT_ERROR_RETRIES`), configuration paths (`SYSTEM_CONFIG_PATH`), and global seeds (`GLOBAL_PRNG_SEED`). These are tagged with `# ΛCONSTANT` and `# ΛSHARED`.
    -   **Enumerations**: Provides shared enums like `SystemStatus` (for operational status codes) and `DataType` (for common data types). These are also considered `# ΛSHARED`.
    -   **Shared Utilities**: This file can also host simple, widely applicable utility functions. (Example `format_error_message` is commented out but illustrates the pattern). Such utilities would be tagged with `# ΛUTIL` and `# ΛSHARED`.

## Constant Re-use and GLYPH_MAP Candidates

-   **Constant Re-use**: Constants defined in `core.common` (especially those marked `# ΛSHARED`) are intended for wide re-use. Their presence here signifies their general applicability. Tracking specific re-use locations can be done via code search/analysis tools or noted in `JULES06_OVERLAP_TRACKER.md` if particularly complex interdependencies are discovered.
-   **GLYPH_MAP Candidates**: While `core.common` primarily deals with programmatic constants, if any string constants defined here are directly tied to UI elements or symbolic representations that might be part of a `GLYPH_MAP` system (e.g., status strings from `SystemStatus` if directly displayed), this connection should be noted.

## How to Use

Elements from `core.common` are designed for easy import and use throughout the system:

```python
from core.common import DEFAULT_NETWORK_TIMEOUT_SECONDS, SystemStatus, DataType
# from core.common import format_error_message # If utility functions are present

# Using a constant
if request_time > DEFAULT_NETWORK_TIMEOUT_SECONDS:
    # Handle timeout

# Using an enum
current_status = SystemStatus.OK
if entity_type == DataType.JSON:
    # Process JSON data
```

## Contribution Guidelines

-   Only add constants, enums, or utilities that are genuinely common and widely applicable. Avoid adding component-specific items here.
-   Ensure all new additions are appropriately tagged using the ΛTAG conventions, especially `# ΛCONSTANT` and `# ΛSHARED`.
-   Provide clear `# ΛNOTE` and `# ΛCAUTION` tags where necessary.
-   If adding shared utility functions, ensure they are well-documented, have minimal dependencies, and include `structlog` logging.
-   Changes to this module, especially to existing shared constants or enums, must be made with extreme care due to their potentially broad impact on the system.
-   Update this README if significant new categories of common elements are added or if the organization changes.

## Overlap with Other Directories

-   **`core/utils/` vs `core/common/`**: `core/common/` is for highly stable, system-wide definitions (often simple constants/enums). `core/utils/` might contain more complex helper functions or utilities that, while general, might not be as universally imported as items in `core/common/`. Shared utilities in `core.common` should be very lightweight.
-   **`core/advanced/constants/` vs `core/common/`**: `core/common/` holds constants for general system operation. `core/advanced/constants/` is for constants and hyperparameters specifically related to the advanced AGI functionalities.
-   Any identified overlaps or potential for consolidation with `identity/`, `diagnostic_engine/`, `lukhas_analyze/`, or `orchestration/` should be documented in `JULES06_OVERLAP_TRACKER.md`.
