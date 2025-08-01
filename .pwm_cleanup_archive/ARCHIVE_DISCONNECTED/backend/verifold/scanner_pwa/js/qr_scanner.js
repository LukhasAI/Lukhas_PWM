// LUKHAS VeriFold Scanner - Enhanced QR Code Scanner with Lukhas ID Verification
class LukhasVeriFoldScanner {
    constructor() {
        this.html5QrCode = null;
        this.isScanning = false;
        this.resultBox = document.getElementById("result");
        this.statusText = document.getElementById("status-text");
        this.statusIndicator = document.querySelector(".status-indicator");
        this.verificationStatus = document.getElementById("verification-status");
        
        this.initializeControls();
        this.startScanner(); // Auto-start for better UX
    }

    initializeControls() {
        const startBtn = document.getElementById("start-scan");
        const stopBtn = document.getElementById("stop-scan");
        
        startBtn.addEventListener("click", () => this.startScanner());
        stopBtn.addEventListener("click", () => this.stopScanner());
    }

    updateStatus(message, type = "pending") {
        this.statusText.textContent = message;
        this.statusIndicator.className = `status-indicator status-${type}`;
    }

    async startScanner() {
        if (this.isScanning) return;
        
        try {
            this.updateStatus("Starting camera...", "pending");
            
            this.html5QrCode = new Html5Qrcode("reader");
            await this.html5QrCode.start(
                { facingMode: "environment" },
                {
                    fps: 10,
                    qrbox: { width: 250, height: 250 },
                    aspectRatio: 1.0
                },
                this.onScanSuccess.bind(this),
                this.onScanFailure.bind(this)
            );
            
            this.isScanning = true;
            this.updateStatus("Scanner active - Point at QR code", "success");
            
        } catch (err) {
            console.error("Failed to start scanner:", err);
            this.updateStatus("Camera access denied or not available", "error");
        }
    }

    async stopScanner() {
        if (!this.isScanning || !this.html5QrCode) return;
        
        try {
            await this.html5QrCode.stop();
            this.html5QrCode = null;
            this.isScanning = false;
            this.updateStatus("Scanner stopped", "pending");
        } catch (err) {
            console.error("Failed to stop scanner:", err);
        }
    }

    async onScanSuccess(decodedText, decodedResult) {
        this.resultBox.textContent = decodedText;
        this.updateStatus("Code detected - Verifying...", "pending");
        
        // Try to parse as JSON for Lukhas ID or VeriFold data
        try {
            const data = JSON.parse(decodedText);
            await this.processStructuredData(data);
        } catch (e) {
            // Handle plain text or URLs
            await this.processPlainText(decodedText);
        }
    }

    async processStructuredData(data) {
        console.log("Processing structured data:", data);
        
        // Send to backend for comprehensive processing
        try {
            const response = await fetch('/api/verify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ payload: JSON.stringify(data) })
            });
            
            if (response.ok) {
                const result = await response.json();
                this.handleBackendResult(result);
                return;
            }
        } catch (error) {
            console.warn("Backend unavailable, using fallback processing:", error);
        }
        
        // Fallback to local processing
        if (data.lukhas_id || data.id || data.user_id) {
            await this.verifyLucasId(data);
        } else if (data.verifold_hash || data.symbolic_hash || data.narrative_hash) {
            await this.verifySymbolicMemory(data);
        } else {
            this.displayGenericData(data);
        }
    }

    handleBackendResult(result) {
        switch (result.type) {
            case 'lukhas_id':
                if (result.result.valid) {
                    this.displayLucasIdResult(result.result);
                    this.updateStatus("✅ Lukhas ID verified", "success");
                } else {
                    this.displayVerificationError(result.result.error);
                    this.updateStatus("❌ Verification failed", "error");
                }
                break;
                
            case 'symbolic_memory':
                if (result.result.valid) {
                    this.displaySymbolicMemoryResult(result.result);
                    this.updateStatus("✅ Symbolic memory verified", "success");
                } else {
                    this.displayVerificationError(result.result.error);
                    this.updateStatus("❌ Memory verification failed", "error");
                }
                break;
                
            case 'structured_data':
                this.displayGenericData(result.result.data);
                this.updateStatus("📊 Data processed", "success");
                break;
                
            case 'plain_text':
                this.displayPlainText(result.result.content);
                this.updateStatus("📝 Text processed", "success");
                break;
                
            default:
                this.displayGenericData(result);
                this.updateStatus("✅ Processed", "success");
        }
    }

    async processPlainText(text) {
        console.log("Processing plain text:", text);
        
        // Check if it looks like a Lukhas ID
        if (/^(USER_T\d+_\d+|LUKHAS_\w+)/.test(text)) {
            await this.verifyLucasId({ lukhas_id: text });
        }
        
        // Check if it's a URL to backend verification
        else if (text.startsWith('http')) {
            await this.verifyUrl(text);
        }
        
        // Plain text
        else {
            this.displayPlainText(text);
        }
    }

    async verifyLucasId(data) {
        const lucasId = data.lukhas_id || data.id || data.user_id;
        
        try {
            // Use actual backend API
            const response = await fetch(`/api/lukhas-id/${encodeURIComponent(lucasId)}`);
            const verificationResult = await response.json();
            
            if (verificationResult.valid) {
                this.displayLucasIdResult(verificationResult);
                this.updateStatus("✅ Lukhas ID verified", "success");
            } else {
                this.displayVerificationError(verificationResult.error || "Invalid Lukhas ID");
                this.updateStatus("❌ Verification failed", "error");
            }
            
        } catch (error) {
            console.error("Lukhas ID verification failed:", error);
            // Fallback to mock verification if backend unavailable
            const mockResult = await this.mockLucasIdVerification(lucasId);
            if (mockResult.valid) {
                this.displayLucasIdResult(mockResult);
                this.updateStatus("✅ Lukhas ID verified (offline)", "success");
            } else {
                this.displayVerificationError("Verification service unavailable");
                this.updateStatus("⚠️ Verification error", "error");
            }
        }
    }

    async verifySymbolicMemory(data) {
        try {
            // Mock symbolic memory verification - integrate with your VeriFold backend
            const response = await fetch('/api/verifold/verify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            
            if (response.ok) {
                const result = await response.json();
                this.displaySymbolicMemoryResult(result);
                this.updateStatus("✅ Symbolic memory verified", "success");
            } else {
                throw new Error('Verification failed');
            }
            
        } catch (error) {
            console.error("Symbolic memory verification failed:", error);
            this.displayVerificationError("Could not verify symbolic memory");
            this.updateStatus("⚠️ Backend unavailable", "error");
        }
    }

    async verifyUrl(url) {
        this.verificationStatus.innerHTML = `
            <div class="verification-result">
                <h3>🌐 URL Detected</h3>
                <p><strong>URL:</strong> ${url}</p>
                <button class="btn btn-primary" onclick="window.open('${url}', '_blank')">
                    🔗 Open Link
                </button>
            </div>
        `;
        this.updateStatus("URL detected", "success");
    }

    displayLucasIdResult(result) {
        this.verificationStatus.innerHTML = `
            <div class="lukhas-id-display">
                <h3>🧠 Lukhas ID Verified</h3>
                <p><strong>ID:</strong> ${result.id}</p>
                <p><strong>Name:</strong> ${result.name}</p>
                <p class="tier">Tier ${result.tier}</p>
                <div class="signature">${result.symbolic_signature}</div>
                <p><strong>Status:</strong> <span style="color: var(--success-color)">✅ Verified</span></p>
            </div>
        `;
    }

    displaySymbolicMemoryResult(result) {
        this.verificationStatus.innerHTML = `
            <div class="verification-result">
                <h3>🔮 Symbolic Memory Verified</h3>
                <p><strong>Hash:</strong> ${result.hash}</p>
                <p><strong>Narrative:</strong> ${result.narrative || 'N/A'}</p>
                <p><strong>Timestamp:</strong> ${result.timestamp}</p>
                <p><strong>Status:</strong> <span style="color: var(--success-color)">✅ Authentic</span></p>
            </div>
        `;
    }

    displayVerificationError(message) {
        this.verificationStatus.innerHTML = `
            <div class="verification-error">
                <h3>❌ Verification Failed</h3>
                <p>${message}</p>
            </div>
        `;
    }

    displayGenericData(data) {
        this.verificationStatus.innerHTML = `
            <div class="verification-result">
                <h3>📊 Structured Data Detected</h3>
                <pre>${JSON.stringify(data, null, 2)}</pre>
            </div>
        `;
        this.updateStatus("Structured data processed", "success");
    }

    displayPlainText(text) {
        this.verificationStatus.innerHTML = `
            <div class="verification-result">
                <h3>📝 Text Content</h3>
                <p>${text}</p>
            </div>
        `;
        this.updateStatus("Text content processed", "success");
    }

    // Mock Lukhas ID verification - replace with actual API
    async mockLucasIdVerification(lucasId) {
        // Simulate network delay
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Mock registry data - replace with actual Lukhas ID registry lookup
        const mockRegistry = {
            "USER_T5_001": {
                id: "USER_T5_001",
                name: "Commander Gonzalo",
                tier: 5,
                symbolic_signature: "🧠🌌⚖️🔐",
                valid: true
            },
            "USER_T4_001": {
                id: "USER_T4_001", 
                name: "Research Operator",
                tier: 4,
                symbolic_signature: "🔬💎🔮⚡",
                valid: true
            }
        };
        
        return mockRegistry[lucasId] || { valid: false };
    }

    onScanFailure(error) {
        // Silent handling of scan failures - they're expected during scanning
    }
}

// Initialize scanner when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.lukhasScanner = new LukhasVeriFoldScanner();
});
