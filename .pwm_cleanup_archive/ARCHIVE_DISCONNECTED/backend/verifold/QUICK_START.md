# 🚀 CollapseHash Project - Quick Start Guide

Welcome to **CollapseHash**, a post-quantum hash verification system for probabilistic observation events! This guide will get you up and running quickly.

## 📋 Project Overview

CollapseHash is a comprehensive quantum-safe verification system that includes:

- **Core Verification Engine**: Post-quantum hash generation and verification
- **Web Interface**: Browser-based verification with QR code support  
- **Command Line Tools**: CLI for batch operations and automation
- **Symbolic Intelligence**: Natural language narration and emotion-aware logging
- **Hardware Integration**: TPM/HSM support for enhanced security
- **Compliance Tools**: Audit trails and tamper-evident logging

## 🏗️ Project Structure

```
CollapseHash/
├── 📁 Core Components
│   ├── collapse_hash_pq.py         # Main hash generator (SPHINCS+)
│   ├── collapse_verifier.py        # Signature verification engine
│   ├── collapse_hash_utils.py      # Utility functions (entropy, keys)
│   └── collapse_chain.py           # Verifiable hash sequences
│
├── 📁 User Interfaces
│   ├── collapse_cli.py             # Command line interface
│   ├── collapse_gui.py             # Desktop GUI (optional)
│   ├── web_qr_verifier.py          # Web verification interface
│   └── web_dashboard.py            # Analytics dashboard
│
├── 📁 Advanced Features
│   ├── entropy_fusion.py           # Emotion-aware entropy fusion
│   ├── journal_mode.py             # Natural language narrator
│   ├── narrative_utils.py          # Text generation utilities
│   └── collapse_replay_engine.py   # Event replay system
│
├── 📁 Hardware & Security
│   ├── hardware_entropy_seed.py    # TPM/HSM entropy seeding
│   └── yubi_seeder.py              # YubiKey integration
│
├── 📁 Web Frontend
│   ├── qr_encoder.py               # QR code generation
│   └── web_templates/              # HTML templates
│       ├── index.html              # Main verification page
│       ├── result.html             # Verification results
│       └── dashboard.html          # Analytics dashboard
│
├── 📁 Testing & Validation
│   ├── verifier_test_suite.py      # Automated tests
│   ├── ledger_auditor.py           # Audit and compliance
│   ├── test_vectors.json           # Test data
│   └── collapse_logbook.jsonl      # Tamper-evident ledger
│
├── 📁 Configuration & Setup
│   ├── config.json                 # Global configuration
│   ├── setup_project.py            # Automated setup script
│   ├── requirements.txt            # Python dependencies
│   ├── Makefile                    # Common tasks
│   └── README.md                   # Full documentation
│
└── 📁 Documentation
    ├── CollapseHash.md              # Technical specification
    └── QUICK_START.md               # This file!
```

## ⚡ Quick Start (5 Minutes)

### 1. **Setup Environment** (2 minutes)
```bash
# Clone or navigate to project directory
cd CollapseHash/

# Run automated setup
python3 setup_project.py

# OR use Makefile
make setup
```

### 2. **Verify Installation** (30 seconds)
```bash
# Check project status
make status

# Run basic tests
make test
```

### 3. **Generate Your First Hash** (1 minute)
```bash
# Using CLI
python3 collapse_cli.py generate --data "Hello Quantum World!"

# Using Python directly
python3 -c "
import collapse_hash_pq
gen = collapse_hash_pq.CollapseHashGenerator()
result = gen.generate_collapse_hash(b'test_data')
print(f'Hash: {result[\"hash\"]}')
print(f'Signature: {result[\"signature\"][:64]}...')
"
```

### 4. **Start Web Interface** (30 seconds)
```bash
# Start verification web interface
make run

# OR manually
python3 web_qr_verifier.py
```
Then open: http://localhost:5000

### 5. **Verify a Hash** (1 minute)
```bash
# Using CLI
python3 collapse_cli.py verify --hash "your_hash_here" --signature "sig_here" --key "key_here"

# Using web interface
# Navigate to http://localhost:5000 and paste your verification data
```

## 🎯 Common Use Cases

### **Basic Hash Verification**
```bash
# Generate a hash
python3 collapse_cli.py generate --output verification_data.json

# Verify the hash  
python3 collapse_cli.py verify --file verification_data.json
```

### **Web-Based Verification**
```bash
# Start web server
make run

# Open browser to http://localhost:5000
# Paste hash, signature, and public key
# Click "Verify Signature"
```

### **QR Code Generation**
```python
from qr_encoder import CollapseQREncoder

encoder = CollapseQREncoder()
qr_image = encoder.encode_hash_to_qr(
    collapse_hash="your_hash", 
    signature="your_signature",
    public_key="your_key"
)
qr_image.save("verification_qr.png")
```

### **Batch Processing**
```bash
# Process multiple hashes
python3 collapse_cli.py batch-verify --input hashes.csv --output results.json
```

## 🔧 Development Workflows

### **Code Quality**
```bash
# Run all quality checks
make qa

# Individual checks
make lint          # Code linting
make format        # Code formatting
make type-check    # Type checking
make test          # Run tests
```

### **Development Server**
```bash
# Start in development mode
make run-dev

# Start dashboard
make run-dashboard
```

### **Testing**
```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Run specific test file
python3 -m pytest verifier_test_suite.py -v
```

## 📊 Monitoring & Analytics

### **Web Dashboard**
```bash
# Start analytics dashboard
python3 web_dashboard.py
# Open: http://localhost:8501
```

### **Audit Logs**
```bash
# Check verification logs
python3 ledger_auditor.py --check-integrity

# Export audit data
make export-data
```

### **Performance Monitoring**
```bash
# Run benchmarks
make benchmark

# Check system status
make status
```

## 🔐 Security Features

### **Hardware Security**
```python
# Enable TPM entropy seeding
from hardware_entropy_seed import HardwareEntropySeeder
seeder = HardwareEntropySeeder(use_tpm=True)
entropy = seeder.get_entropy(32)
```

### **YubiKey Integration**
```python
# Use YubiKey for enhanced security
from yubi_seeder import YubiKeySeeder
yubi = YubiKeySeeder()
if yubi.is_available():
    seed = yubi.get_entropy_seed()
```

## 🎨 Advanced Features

### **Emotion-Aware Logging**
```python
from entropy_fusion import EmotionAwareEntropy
from journal_mode import NaturalLanguageNarrator

# Create emotion-aware hash
entropy = EmotionAwareEntropy()
narrator = NaturalLanguageNarrator()

hash_result = entropy.fuse_with_emotion(quantum_data, emotion="excited")
story = narrator.narrate_collapse_event(hash_result)
```

### **Symbolic Intelligence**
```python
from collapse_replay_engine import CollapseReplayEngine

# Replay and analyze hash sequences
replay = CollapseReplayEngine()
replay.load_sequence("collapse_logbook.jsonl")
insights = replay.analyze_patterns()
```

## 🚨 Troubleshooting

### **Common Issues**

1. **Import Error: "oqs" module not found**
   ```bash
   pip install liboqs-python
   ```

2. **Web interface won't start**
   ```bash
   pip install flask jinja2
   python3 web_qr_verifier.py --port 8080
   ```

3. **QR codes not generating**
   ```bash
   pip install qrcode[pil] pillow
   ```

4. **Permission denied errors**
   ```bash
   chmod +x setup_project.py
   python3 setup_project.py
   ```

### **Getting Help**

- Check `README.md` for detailed documentation
- Run `make help` for available commands
- Check `config.json` for configuration options
- Look at test files for usage examples

## 🎯 Next Steps

1. **Explore the Web Interface**: Start with `make run` and try verifying hashes
2. **Read the Documentation**: Check `README.md` and `CollapseHash.md`
3. **Run the Tests**: Use `make test` to ensure everything works
4. **Try Advanced Features**: Experiment with symbolic intelligence and hardware integration
5. **Customize Configuration**: Edit `config.json` for your specific needs

## 📚 Learn More

- **Technical Specification**: See `CollapseHash.md`
- **Full Documentation**: See `README.md`
- **API Reference**: Check docstrings in individual modules
- **Test Examples**: Look at `verifier_test_suite.py`

---

**🚀 Ready to verify quantum events with post-quantum security!**

For questions or issues, check the documentation or create an issue in the project repository.

**LUKHAS AGI Core** | *Quantum-Safe Hash Verification Technology*
