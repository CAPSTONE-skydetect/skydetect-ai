"""Fixed pinhole projection: world z is up, camera z is forward."""

from dataclasses import asdict, dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class Camera:
    width: int = 1920
    height: int = 1080
    horizontal_fov_deg: float = 65.0
    position: tuple = (0.0, -140.0, 15.0)
    look_at: tuple = (0.0, 140.0, 90.0)
    near_m: float = 1.0

    def __post_init__(self):
        if len(self.position) != 3 or len(self.look_at) != 3:
            raise ValueError("Camera position and look_at must have three coordinates")
        if int(self.width) != self.width or int(self.height) != self.height:
            raise ValueError("Image dimensions must be integers")
        values = [self.width, self.height, self.horizontal_fov_deg, self.near_m,
                  *self.position, *self.look_at]
        if not np.all(np.isfinite(values)) or min(self.width, self.height, self.near_m) <= 0:
            raise ValueError("Camera parameters must be finite and dimensions positive")
        if not 1 < self.horizontal_fov_deg < 175:
            raise ValueError("horizontal_fov_deg must be between 1 and 175")
        if np.linalg.norm(np.subtract(self.look_at, self.position)) < 1e-8:
            raise ValueError("Camera look_at must differ from position")

    @property
    def focal_px(self):
        return self.width / (2 * np.tan(np.radians(self.horizontal_fov_deg) / 2))

    @cached_property
    def rotation(self):
        forward = np.subtract(self.look_at, self.position)
        forward = forward / np.linalg.norm(forward)
        reference_up = np.array([0., 0., 1.])
        if abs(forward @ reference_up) > .99:
            reference_up = np.array([0., 1., 0.])
        right = np.cross(forward, reference_up)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        return np.stack([right, down, forward])

    def project(self, position, object_width, object_height):
        if not np.all(np.isfinite([object_width, object_height])) or min(object_width, object_height) <= 0:
            raise ValueError("Object dimensions must be positive")
        x, y, depth = self.rotation @ (np.asarray(position) - self.position)
        if not np.all(np.isfinite([x, y, depth])) or depth <= self.near_m:
            return None
        return {"cx": float(.5 + self.focal_px * x / depth / self.width),
                "cy": float(.5 + self.focal_px * y / depth / self.height),
                "w": float(self.focal_px * object_width / depth / self.width),
                "h": float(self.focal_px * object_height / depth / self.height)}

    def visible(self, bbox, min_size_px=.25):
        if bbox is None:
            return False
        return (0 < bbox["cx"] < 1 and 0 < bbox["cy"] < 1
                and min_size_px <= bbox["w"] * self.width < self.width
                and min_size_px <= bbox["h"] * self.height < self.height
                and max(bbox["w"] * self.width, bbox["h"] * self.height) >= 1.5)

    def to_dict(self):
        return asdict(self)
