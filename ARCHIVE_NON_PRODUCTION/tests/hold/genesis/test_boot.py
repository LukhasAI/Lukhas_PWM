# ΛORIGIN_AGENT: CODEX-01
# ΛTASK_ID: C-10
# ΛCOMMIT_WINDOW: postO3-infra-phase2
# ΛPROVED_BY: Human Overseer (Gonzalo)
# ΛUDIT: Basic genesis self-test
import core.symbolic_boot as symbolic_boot


def test_boot_sequence_runs():
    symbolic_boot.main()
