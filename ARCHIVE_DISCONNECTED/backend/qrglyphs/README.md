


# 📦 Qrglyphs Module (Lukhas AGI Ecosystem)

Welcome to the Qrglyphs module — a symbolic, beautiful, and secure system for encrypted identity sharing and verification, part of the Lukhas-ID Portal.

---
## ✨ Overview

Qrglyphs are **dynamic, animated symbolic artifacts** that securely carry encrypted information, links, or identity verifications.  
They combine art, cryptography, cloud storage, and QR technology into one living object.

This module currently supports:
- 📄 Uploading a payload (document, file, symbolic data)
- 🔒 Encrypting the payload (AES-Fernet)
- ☁️ Uploading encrypted payload to a mock IPFS gateway
- 🧩 Generating a QR code containing the access data
- 🌟 (Coming soon) Overlaying the QR onto animated Lottie visuals

---
## 📜 LEGAL NOTE

This project uses third-party animations sourced from publicly available Pinterest download links for **DEMONSTRATION PURPOSES ONLY**.  
Lukhas-ID and Lukhas-AGI **do not claim ownership** of these artworks.  
Production versions must use **original, licensed, or commissioned artworks**.

---
## 🛠️ Folder Structure

```plaintext
/qrglyphs/
├── __init__.py
├── qrglymph_public.py      # Public Qrglyph generator
├── assets/
│   ├── fractal_star_demo.json
│   └── previews/
│       └── fractal_star.gif
├── tests/
│   └── test_create_qrglyph.py
└── README.md
```

---
## 🚀 Usage (Quick Start)

1. Install requirements:
```bash
pip install cryptography segno
```

2. Generate a Qrglyph manually:
```bash
python qrglyphs/qrglymph_public.py --payload path/to/your/file.txt --output path/to/output/
```

3. Run automated test:
```bash
python qrglyphs/tests/test_create_qrglyph.py
```

---
## 🧪 Testing

- The test script automatically:
  - Creates a dummy payload
  - Runs full encryption + mock upload
  - Verifies QR code generation
  - Prints status messages

---
## 🛤️ Roadmap (Future Features)

- Real IPFS uploads via Web3.Storage / Pinata
- Personalizable dynamic animations (upload your own)
- Binary transformation filters
- Li-Fi symbolic transmission (light pulse scanning)
- Dynamic steganographic signature embedding
- Smart contracts for payment unlocks
- Face ID biometric linking (Lukhas-ID integration)

---
## 🏷️ Metadata

- 📦 Module: `qrglyphs`
- 🔧 Version: `v0.1.0`
- 🖋️ Author: Gonzalo Dominguez
- 📅 Updated: 2025-04-29