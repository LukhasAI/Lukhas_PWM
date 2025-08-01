

# VeriFold — Symbolic Identity & Memory Integrity Framework

VeriFold is a post-quantum resilient, ethically auditable, and human-readable symbolic identity and memory system. It protects personal narratives, collapse memories, and consent trails across devices, chains, and lifespans.

---

## 🧠 Modules Overview

| Module | Description |
|--------|-------------|
| `crypto_router.py` | Post-quantum cryptographic router (SPHINCS+, Kyber, Falcon, fs-PIBE) |
| `ethics_verifier.py` | Validates whether memory replays are ethically compliant |
| `consent_scope_validator.py` | Ensures symbolic actions align with user consent and tier level |
| `symbolic_audit_mode.py` | Records and audits memory replays with GDPR-compliant logging |
| `glyph_stego_encoder.py` | Generates dual-layer GLYMPHs (visible + steganographic QR) |
| `zkNarrativeProof_adapter.py` | Produces verifiable zk-proofs of symbolic memory replays |

---

## 🚀 Core Capabilities

- ✅ Tiered symbolic identity with mnemonic + emoji seed
- ✅ Post-quantum memory collapse signatures (SPHINCS+ / Falcon)
- ✅ Emotional entropy scoring for symbolic replays
- ✅ ZK-proof generation for narrative events
- ✅ Audit-ready consent checkpoints and compliance export

---

## 🧪 Running Tests

Tests are located in `/tests/`. Run all tests with:

```bash
pytest tests/
```

Each test covers key ethical, cryptographic, and narrative pipelines.

---

## 🛠 Contributing

```bash
# Install deps
pip install -r requirements.txt
```

- Clone or fork this repo
- Use the module scaffolding patterns
- Follow symbolic ID tiering when generating new replayers, glyphs, or audit handlers

---

## 🔭 Roadmap (v2+)

- zkSNARKs with trusted setup optionality
- Secure GPT-assisted narrative replayer
- Emotional NFT minting
- Visual GLYMPH generator
- Full Web3 export adapter + Filecoin storage