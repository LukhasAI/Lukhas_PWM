from embodiment.body_state import ProprioceptiveState


def test_proprioceptive_update():
    state = ProprioceptiveState(joint_positions={"arm": 0.0})
    state.update_joint("arm", 1.0)
    assert state.joint_positions["arm"] == 1.0
