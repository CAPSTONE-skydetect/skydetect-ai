from ai_server.services.manual_roi_tracker import _lk_params, _target_lk_params


def test_small_target_is_not_erased_by_a_three_level_pyramid():
    params = _target_lk_params((13., 7.))
    assert params["maxLevel"] == 0
    assert params["winSize"] == (11, 11)
    assert _lk_params()["maxLevel"] == 3
    assert _lk_params()["winSize"] == (21, 21)


def test_large_target_keeps_multiscale_motion_search():
    assert _target_lk_params((128., 64.))["maxLevel"] == 3
    assert _target_lk_params((128., 64.))["winSize"] == (21, 21)


def test_resolution_scaling_follows_the_actual_target_size():
    a = _target_lk_params((12., 8.), 1.)
    b = _target_lk_params((24., 16.), 2.)
    assert b["winSize"][0] > a["winSize"][0]
    assert b["maxLevel"] > a["maxLevel"]
