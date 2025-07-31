from datetime import datetime, timezone

class Memoria:
    memory_log = []

    def store(self, input_data, decision):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input": input_data,
            "decision": decision
        }
        self.memory_log.append(log_entry)
        print(f"[MEMORIA] -> Logged trace: {log_entry}")

    def trace(self):
        print("\n🧠 OXNITUS TRACE LOG:")
        for i, entry in enumerate(self.memory_log, 1):
            print(f"\n#{i}")
            print(f"Timestamp: {entry['timestamp']}")
            print(f"Intent:    {entry['input']}")
            print(f"Decision:  {entry['decision']['justification']}")