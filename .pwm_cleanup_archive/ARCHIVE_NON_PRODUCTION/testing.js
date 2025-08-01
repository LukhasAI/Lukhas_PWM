

// 📂 FILE: dashboard/testing.js
// 🧪 PURPOSE: Run and display symbolic unit test results

import { useEffect, useState } from "react";

export default function TestingDashboard() {
  const [testOutput, setTestOutput] = useState("");
  const [running, setRunning] = useState(false);

  const runTests = async () => {
    setRunning(true);
    setTestOutput("🧪 Running symbolic system tests...");

    try {
      const res = await fetch("/api/run-tests");
      const data = await res.text();
      setTestOutput(data);
    } catch (err) {
      setTestOutput("❌ Failed to run tests.");
    } finally {
      setRunning(false);
    }
  };

  return (
    <div style={{ padding: "2em", fontFamily: "sans-serif" }}>
      <h2>🧪 System Testing</h2>
      <button onClick={runTests} disabled={running}>
        {running ? "Running..." : "Run All Symbolic Tests"}
      </button>
      <pre style={{ marginTop: "1em", background: "#f4f4f4", padding: "1em", borderRadius: "6px" }}>
        {testOutput}
      </pre>
    </div>
  );
}