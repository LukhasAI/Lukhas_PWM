# Entity Activation Tool

This utility scans each major subsystem to locate classes and public functions. It
then generates activation modules that can be used to register these entities with
a hub or service registry.

Usage:
```bash
python tools/entity_activation/entity_activator.py
```

Running the script creates `*_entity_activation.py` files in the repository root
and an `entity_activation_report.json` summarizing the number of entities found.
