import numpy as np

from ai_server.services.motion_lock_audit import MotionLockAudit


def observe(audit, candidate, **overrides):
    options = dict(klt_center=(0., 0.), background_center=(0., 0.), candidate=candidate,
                   motion_score=.8, appearance_score=.8, minimum_appearance=.58,
                   affine=np.array([[1., 0., 0.], [0., 1., 0.]]), size=(12., 7.))
    options.update(overrides)
    return audit.observe(**options)


def test_requires_two_consistent_frames_before_overriding_background_lock():
    audit = MotionLockAudit()
    assert observe(audit, (2., 0.)) is None
    assert observe(audit, (4., 0.)) == (4., 0.)


def test_does_not_override_already_independently_moving_klt():
    audit = MotionLockAudit()
    assert observe(audit, (4., 0.), klt_center=(2., 0.)) is None
    assert observe(audit, (6., 0.), klt_center=(3., 0.)) is None


def test_no_motion_or_weak_appearance_cannot_force_a_target_switch():
    audit = MotionLockAudit()
    assert observe(audit, (0., 0.)) is None
    assert observe(audit, (4., 0.), appearance_score=.2) is None
    assert observe(audit, (6., 0.), appearance_score=.2) is None
    assert audit.streak == 0


def test_missing_evidence_breaks_confirmation_streak():
    audit = MotionLockAudit()
    assert observe(audit, (2., 0.)) is None
    assert observe(audit, None) is None
    assert observe(audit, (4., 0.)) is None


def test_confirmation_transports_candidate_with_camera_motion():
    audit = MotionLockAudit()
    affine = np.array([[1., 0., 10.], [0., 1., 0.]])
    assert observe(audit, (12., 0.), klt_center=(10., 0.), background_center=(10., 0.), affine=affine) is None
    assert observe(audit, (24., 0.), klt_center=(20., 0.), background_center=(20., 0.), affine=affine) == (24., 0.)


def test_existing_motion_acceptance_still_requires_appearance_and_temporal_evidence():
    audit = MotionLockAudit(minimum_motion=.38)
    assert observe(audit, (2., 0.), motion_score=.4) is None
    assert observe(audit, (4., 0.), motion_score=.4) == (4., 0.)
    assert observe(audit, (6., 0.), motion_score=.4, appearance_score=.2) is None
