// 📂 FILE: crypto.js
// 🔐 PURPOSE: LucasID Symbolic Cryptographic Portal

import { useState, useEffect } from "react";

export default function CryptoPortal() {
  const [status, setStatus] = useState("🔄 Loading symbolic crypto status...");
  const [cryptoData, setCryptoData] = useState(null);

  useEffect(() => {
    fetch("/api/crypto/status")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to fetch crypto status.");
        return res.json();
      })
      .then(setCryptoData)
      .catch((err) => setStatus(`❌ ${err.message}`));
  }, []);

  if (!cryptoData)
    return (
      <div style={{ padding: "2em", fontFamily: "sans-serif" }}>
        <p>{status}</p>
      </div>
    );

  return (
    <div style={{ padding: "2em", fontFamily: "sans-serif" }}>
      <h2>🔐 LucasID Crypto Portal</h2>
      <p><strong>Encryption Tier:</strong> {cryptoData.encryption_tier}</p>
      <p><strong>Collapse Hash:</strong> {cryptoData.collapse_hash}</p>
      <p><strong>Vault Status:</strong> {cryptoData.vault_status}</p>

      {cryptoData.can_download ? (
        <a href="/api/crypto/export" style={{ marginTop: "2em", display: "inline-block", padding: "0.75em 1.5em", background: "#0070f3", color: "#fff", borderRadius: "5px" }}>
          🔽 Download Encrypted Vault
        </a>
      ) : (
        <p style={{ color: "orange" }}>🔒 Advanced vault export requires Tier 3+ access.</p>
      )}
    </div>
  );
}
