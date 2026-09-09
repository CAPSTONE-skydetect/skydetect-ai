"""Seeded renewal guidance, explicitly not fitted to animal movement records."""
import numpy as np

from .parameters import SAMPLING


class BehaviorSchedule:
    def __init__(self, rng, duration_s):
        self.segments = []
        elapsed, turn = 0., 0.
        low, high = SAMPLING["foraging_duration_s"]
        while elapsed < duration_s:
            # Truncated gamma dwell time replaces a shared four-second sine period.
            duration = float(np.clip(rng.gamma(2., .7), low, high))
            turn = float(rng.choice([-1.,1.])*np.radians(rng.uniform(*SAMPLING["foraging_turn_deg"])))
            self.segments.append(dict(start_s=elapsed, end_s=elapsed+duration, turn_rad=turn))
            elapsed += duration
        self.thermal_peak = float(rng.uniform(*SAMPLING["thermal_peak_m_s"]))
        self.thermal_radius_scale = float(rng.uniform(*SAMPLING["thermal_radius_scale"]))

    def foraging_goal(self, time_s, position, direction):
        segment = next((s for s in self.segments if time_s < s["end_s"]), self.segments[-1])
        angle = segment["turn_rad"]
        xy = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]) @ direction[:2]
        return position + 80*np.r_[xy,direction[2]]

    def thermal_updraft(self, distance_m, orbit_radius_m):
        # A stationary Gaussian thermal core, not a ring glued to the orbit.
        return self.thermal_peak*np.exp(-(distance_m/(orbit_radius_m*self.thermal_radius_scale))**2)

    def metadata(self):
        return dict(model="renewal_guidance_v1", calibrated=False, durations=self.segments,
                    thermal_peak_m_s=self.thermal_peak, thermal_radius_scale=self.thermal_radius_scale,
                    note="Guidance priors, not GPS-learned species behavior.")
