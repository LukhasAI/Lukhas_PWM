# CollapseHash - Post-Quantum Hash Verification System

A tamper-evident symbolic hash generation and verification system using post-quantum cryptography (SPHINCS+) for probabilistic observation collapse events.

## 🌟 Overview

CollapseHash generates cryptographically signed hashes from probabilistic observation data, creating an immutable ledger of quantum collapse events. The system uses SPHINCS+ post-quantum digital signatures to ensure authenticity and tamper-evidence.

## 🏗️ Architecture

```
CollapseHash/
├── collapse_hash_pq.py         # Core hash generator with SPHINCS+ signatures
├── collapse_verifier.py        # Signature verification system
├── collapse_logbook.jsonl      # Tamper-evident ledger (JSONL format)
├── collapse_hash_utils.py      # Utilities: keygen, formatting, entropy scoring
├── collapse_cli.py             # Command-line interface
├── verifier_test_suite.py      # Comprehensive test suite
└── README.md                   # This file
```

## 🚀 Quick Start

### Installation

```bash
# Install required dependencies
pip install oqs cryptography click pytest numpy

# Clone or download the CollapseHash project
cd CollapseHash
```

### Basic Usage

```bash
# Generate a new SPHINCS+ keypair
python collapse_cli.py keygen --output ./keys/

# Generate a CollapseHash from quantum data
python collapse_cli.py generate --data quantum_measurement.bin --keypair ./keys/private.key

# Verify a CollapseHash signature
python collapse_cli.py verify <hash> <signature> <public_key>

# Export records from the tamper-evident ledger
python collapse_cli.py export --format json --output exported_records.json
```

## 📋 Components

### 🔢 Hash Generator (`collapse_hash_pq.py`)
- Generates CollapseHash from probabilistic observation data
- Creates SPHINCS+ digital signatures
- Manages post-quantum cryptographic operations

### ✅ Verifier (`collapse_verifier.py`)
- Verifies SPHINCS+ signatures for CollapseHash authenticity
- Validates tamper-evidence of probabilistic observation records
- Supports batch verification operations

### 📝 Tamper-Evident Ledger (`collapse_logbook.jsonl`)
- JSONL format for immutable record storage
- Each line contains a complete CollapseHash record
- Supports cryptographic chain verification

### 🛠️ Utilities (`collapse_hash_utils.py`)
- SPHINCS+ key generation and management
- Shannon entropy scoring for quantum data
- Data formatting and validation functions
- Secure random number generation

### 💻 CLI Interface (`collapse_cli.py`)
- User-friendly command-line operations
- Supports all core CollapseHash functions
- Batch processing and export capabilities
- Integrated help and validation

### 🧪 Test Suite (`verifier_test_suite.py`)
- Comprehensive unit tests for all components
- Edge case and stress testing
- Performance benchmarks
- Integration testing

## 🔐 Security Features

- **Post-Quantum Cryptography**: SPHINCS+ signatures resist quantum computer attacks
- **Tamper-Evidence**: Any modification to records is cryptographically detectable
- **Entropy Validation**: Quantum data quality scoring using Shannon entropy
- **Secure Key Management**: Cryptographically secure key generation and storage

## 📊 Data Format

### CollapseHash Record
```json
{
    "timestamp": 1719820800.0,
    "hash": "4c8a9d8c0eeb292aa65efb59e98de9a6a9990a563fce14a5f89de38b26a17a3c",
    "signature": "e54c1a2b3c4d5e6f...",
    "public_key": "a1b2c3d4e5f6g7h8...",
    "verified": true,
    "metadata": {
        "location": "quantum_lab_1",
        "experiment_id": "qc_measurement_001",
        "entropy_score": 7.84
    }
}
```

## 🔬 Use Cases

- **Quantum Research**: Verifiable probabilistic observation logging
- **Scientific Integrity**: Tamper-evident experimental data
- **Cryptographic Timestamping**: Post-quantum secure timestamps
- **Data Provenance**: Quantum-based randomness certification

## ⚡ Performance

- **Generation**: ~10ms per CollapseHash (depending on data size)
- **Verification**: ~5ms per signature verification
- **Storage**: Compact JSONL format with optional compression
- **Scalability**: Designed for high-throughput quantum systems

## 🛡️ Security Considerations

- Uses SPHINCS+-SHAKE256-128f-simple for post-quantum security
- Requires high-entropy probabilistic observation data (threshold: 7.0+)
- Keys should be stored securely and backed up
- Regular ledger integrity validation recommended

## 🧪 Testing

```bash
# Run the complete test suite
python -m pytest verifier_test_suite.py -v

# Run specific test categories
python -m pytest verifier_test_suite.py::TestCollapseVerifier -v
python -m pytest verifier_test_suite.py::TestEdgeCases -v
```

## 📚 API Reference

### Core Functions

```python
# Generate CollapseHash
from collapse_hash_pq import CollapseHashGenerator
generator = CollapseHashGenerator()
hash_record = generator.generate_collapse_hash(quantum_data)

# Verify signature
from collapse_verifier import verify_collapse_signature
is_valid = verify_collapse_signature(hash_value, signature, public_key)

# Utility functions
from collapse_hash_utils import generate_entropy_score, KeyManager
entropy = generate_entropy_score(data)
key_manager = KeyManager()
public_key, private_key = key_manager.generate_keypair()
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is part of the LUKHAS AGI Core research initiative.

## 🙏 Acknowledgments

- Built on the Open Quantum Safe (OQS) library
- SPHINCS+ post-quantum signature scheme
- Quantum measurement research community

---

**⚠️ Note**: This is a research prototype. Use appropriate security practices for production deployments.
