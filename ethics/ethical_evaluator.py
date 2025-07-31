from datetime import datetime, timezone

class EthicalEvaluator:
    def evaluate(self, stimulus):
        print(f"[EVALUATING] -> {stimulus}")
        return {
            "original": stimulus,
            "ethical": True,
            "justification": "Intent poses no ethical conflict. Default approval granted."
        }


class CollapseEngine:
    def collapse(self, decision):
        print(f"[COLLAPSE] -> Executing: {decision['original']}")
        return f"EXECUTED: {decision['original']}"


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
        print("\n\U0001f9e0 OXNITUS TRACE LOG:")
        for i, entry in enumerate(self.memory_log, 1):
            print(f"\n#{i}")
            print(f"Timestamp: {entry['timestamp']}")
            print(f"Intent:    {entry['input']}")
            print(f"Decision:  {entry['decision']['justification']}")
