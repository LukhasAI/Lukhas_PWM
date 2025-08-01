# 🧠 LUKHAS VeriFold Scanner

A Progressive Web App (PWA) for scanning and verifying QR codes with Lukhas ID authentication and VeriFold symbolic memory verification.

## ✨ Features

- **📱 Mobile-Optimized Scanner**: Real-time QR code scanning using device camera
- **🔐 Lukhas ID Verification**: Authenticate users against the Lukhas ID registry
- **🔮 Symbolic Memory Verification**: Verify VeriFold symbolic memory hashes
- **🎨 Lukhas Branding**: Beautiful UI matching the Lukhas design system
- **⚡ Progressive Web App**: Installable on mobile devices
- **🌐 Backend Integration**: RESTful API for verification services

## 🚀 Quick Start

### Option 1: Simple Static Server (Basic functionality)
```bash
cd scanner_pwa
python3 -m http.server 8080
# Open http://localhost:8080
```

### Option 2: Full Backend Integration (Recommended)
```bash
cd scanner_pwa
./start_scanner.sh
# Open http://localhost:5000
```

## 📂 Project Structure

```
scanner_pwa/
├── index.html              # Main PWA interface
├── style.css               # Lukhas-branded styling
├── manifest.json           # PWA manifest
├── js/
│   └── qr_scanner.js       # Enhanced scanner logic
├── scanner_backend.py      # Backend verification logic
├── scanner_api.py          # Flask API server
├── requirements.txt        # Python dependencies
├── start_scanner.sh        # Startup script
└── README.md              # This file
```

## 🔧 Backend Integration

The scanner integrates with your existing Lukhas ecosystem:

### Lukhas ID Verification
- Reads from `~/lukhas/lukhas/identity/identity/lukhas_registry.jsonl`
- Verifies user tiers and symbolic signatures
- Displays verification status with user details

### VeriFold Integration
- Connects to your VeriFold verification system
- Validates symbolic memory hashes
- Supports narrative verification

### API Endpoints

- `GET /api/status` - Health check
- `POST /api/verify` - General QR code verification
- `GET /api/lukhas-id/<id>` - Direct Lukhas ID verification
- `POST /api/verifold/verify` - Symbolic memory verification

## 🎯 Supported QR Code Formats

### Lukhas ID
```json
{
    "lukhas_id": "USER_T5_001"
}
```

### Symbolic Memory
```json
{
    "verifold_hash": "abc123...",
    "narrative": "Memory fold description"
}
```

### Plain Text
- Direct Lukhas ID: `USER_T5_001`
- URLs: `https://example.com/verify/...`
- Any text content

## 🔮 Next Steps

- **🔗 Deep Integration**: Connect to more Lukhas modules
- **📊 Analytics**: Add scanning statistics and reporting
- **🔔 Notifications**: Real-time verification alerts
- **🌐 Multi-Device**: Sync across Lukhas ecosystem
- **🎨 Customization**: Per-tier UI customization

## 🛠 Development

### Dependencies
```bash
pip install flask flask-cors requests
```

### Testing
```bash
# Test Lukhas ID verification
curl -X GET http://localhost:5000/api/lukhas-id/USER_T5_001

# Test general verification
curl -X POST http://localhost:5000/api/verify \
  -H "Content-Type: application/json" \
  -d '{"payload": "{\"lukhas_id\": \"USER_T5_001\"}"}'
```

## 🌟 Integration with Lukhas Ecosystem

This scanner is designed to work seamlessly with:
- **Lukhas ID System**: User authentication and tier management
- **VeriFold**: Symbolic memory verification
- **Lukhas Web**: Shared styling and branding
- **LUKHAS Brain**: Memory integration
- **Symbolic Audit**: Compliance and verification

---

*Part of the LUKHAS AGI Ecosystem - Symbolic Computing for Human-AI Collaboration*
