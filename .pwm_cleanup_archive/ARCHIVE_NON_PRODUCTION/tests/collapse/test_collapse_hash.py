from core.symbolic.collapse.trace import CollapseTrace


def test_collapse_hash_generated():
    tracer = CollapseTrace()
    tracer.log_collapse(["a"], "b", "test", {"r": 1})
    event = tracer.collapse_log[-1]
    assert "collapse_hash" in event
    assert len(event["collapse_hash"]) > 0
