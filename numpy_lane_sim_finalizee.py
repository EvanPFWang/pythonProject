import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

R_BASE = 1000 / (2 * np.pi)
LANE_WIDTH = 10.0
LANES = 5
DT = 1e-2
V_MAX = 70.0
THETA_VIEW = np.pi / 4
RADII = R_BASE + np.arange(LANES) * LANE_WIDTH
TRACK_LENGTHS = 2 * np.pi * RADII

SWERVE  = 0b001
BRAKE   = 0b010
CRASHED = 0b100


def wrap_theta(theta: np.ndarray) -> np.ndarray:
    return (theta + np.pi) % (2 * np.pi) - np.pi

def theta_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return wrap_theta(b - a)

def meters_per_sec_to_theta(v: np.ndarray, lane_idx: np.ndarray) -> np.ndarray:
    return v * DT * 2 * np.pi / TRACK_LENGTHS[lane_idx]

@dataclass
class Car:
    cid: int
    lane: int
    theta: float
    vel: float
    status: int = 0
    theta_total: float = 0.
    radius: float = field(init=False)

    def __post_init__(self):
        self.radius = RADII[self.lane] if self.lane >= 0 else np.nan

#-------------------------------------------------
#collision truth‑table (4‑bit, 16 combos)
#-------------------------------------------------
_COLLISION_TABLE = np.zeros(16, dtype=bool)
for key in range(16):
    rs = key & 0b0001
    rb = key & 0b0010
    fb = key & 0b0100
    fc = key & 0b1000
    _COLLISION_TABLE[key] = bool(fc) or (fb and not rb)


class LaneSim:
    """NumPy‑accelerated multi‑lane traffic model with 8‑mode peek matrix."""

    def __init__(self, cars: List[Car]):
        self.cars = cars
        self.N = len(cars)
        self._lane   = np.array([c.lane  for c in cars], dtype=np.int16)
        self._theta  = np.array([c.theta for c in cars], dtype=np.float32)
        self._vel    = np.array([c.vel   for c in cars], dtype=np.float32)
        self._status = np.array([c.status for c in cars], dtype=np.uint8)
        self._theta_total = np.zeros(self.N, dtype=np.float32)
        self._idx = np.arange(self.N)

    def step(self):
        active = self._lane >= 0
        lane_idx = self._lane[active]

        fr = self._peek_nearest(mode=1)   #front‑narrow, same lane
        new_speed, lane_delta = self._drive(fr)
        self._apply_decision(new_speed, lane_delta)

        dtheta = meters_per_sec_to_theta(self._vel[active], lane_idx)
        self._theta[active] = wrap_theta(self._theta[active] + dtheta)
        self._theta_total[active] += dtheta

        self._crash_resolution()
        self._post_lane_change_collision(fr)

        for i, c in enumerate(self.cars):
            c.lane, c.theta, c.vel, c.status = int(self._lane[i]), float(self._theta[i]), float(self._vel[i]), int(self._status[i])

    #------------- peek helpers -------------
    def _compute_distance_matrices(self):
        dθ = theta_distance(self._theta[:, None], self._theta[None, :])
        same_lane = self._lane[:, None] == self._lane[None, :]
        return dθ, same_lane

    def _visibility_mask(self, mode: int, dθ, same_lane):
        front_n = (0 < dθ) & (dθ <  np.pi/16)
        back_n  = (-np.pi/16 < dθ) & (dθ < 0)
        front_w = (0 < dθ) & (dθ <  THETA_VIEW)
        back_w  = (-THETA_VIEW < dθ) & (dθ < 0)

        if mode == 1:
            vis = same_lane & front_n
        elif mode == 2:
            vis = same_lane & back_n
        elif mode == 3:
            vis = same_lane & (front_w | back_w)
        elif mode == 4:
            vis = same_lane & back_w
        elif mode == 5:
            vis = same_lane & front_w
        elif mode == 6:
            vis = front_w | back_w
        elif mode == 7:
            vis = back_w
        elif mode == 8:
            vis = front_w
        else:
            raise ValueError("mode must be 1‑8")
        return vis

    def _peek_matrix(self, mode: int, k: int = 1) -> np.ndarray:
        """Return array shape (N, k) of neighbour IDs for the chosen mode."""
        dθ, same_lane = self._compute_distance_matrices()
        vis = self._visibility_mask(mode, dθ, same_lane)
        dist = np.abs(dθ)
        dist[~vis] = np.inf
        #partial argpartition for top‑k
        idx_part = np.argpartition(dist, kth=k, axis=1)[:, :k]
        #ensure sorted by actual distance
        rows = np.arange(self.N)[:, None]
        sorted_k = np.take_along_axis(idx_part, np.argsort(dist[rows, idx_part], axis=1), axis=1)
        sorted_k[dist[rows, sorted_k] == np.inf] = -1
        return sorted_k.astype(np.int16)

    def _peek_nearest(self, mode: int = 1) -> np.ndarray:
        return self._peek_matrix(mode, k=1)[:, 0]

#decisions
    def _drive(self, fr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        theta_max = meters_per_sec_to_theta(np.full(self.N, V_MAX, float), self._lane)
        theta2n = np.zeros(self.N, float)
        valid = fr >= 0
        theta2n[valid] = theta_distance(self._theta[valid], self._theta[fr[valid]])
        theta_min = theta_max * 0.05
        new_speed = np.where(theta2n <= theta_min, self._vel, V_MAX * np.tanh(theta2n / theta_max))
        # simple greedy lane choice
        left  = np.where(self._lane > 0, theta_max[self._lane - 1], -1)
        mid   = theta_max
        right = np.where(self._lane < LANES - 1, theta_max[self._lane + 1], -1)
        lane_delta = np.vstack([left, mid, right]).argmax(axis=0) - 1
        return new_speed, lane_delta.astype(np.int8)

    def _apply_decision(self, new_speed, lane_delta):
        swap = lane_delta != 0
        self._vel[:] = new_speed
        self._status &= ~SWERVE
        self._status |= swap.astype(np.uint8) * SWERVE
        self._lane += lane_delta
        np.clip(self._lane, 0, LANES - 1, out=self._lane)

    #crash passes
    def _crash_resolution(self):
        for lane in range(LANES):
            idx = np.where(self._lane == lane)[0]
            if idx.size < 2:
                continue
            order = np.argsort(self._theta[idx])
            idx_sorted = idx[order]
            gap = theta_distance(self._theta[idx_sorted[:-1]], self._theta[idx_sorted[1:]])
            collide = np.where(gap < 2 * np.pi * 2 / TRACK_LENGTHS[lane])[0]
            if collide.size:
                hit = idx_sorted[collide]
                vic = idx_sorted[collide + 1]
                self._status[np.concatenate([hit, vic])] |= CRASHED
                self._lane[np.concatenate([hit, vic])] = -1

    def _post_lane_change_collision(self, fr):
        rear_ids = self._idx
        front_ids = fr
        mask = front_ids >= 0
        rear_ids = rear_ids[mask]
        front_ids = front_ids[mask]
        key = (((self._status[front_ids] & CRASHED) >> 2) * 8 |
               ((self._status[front_ids] & BRAKE)   >> 1) * 4 |
               ((self._status[rear_ids]  & BRAKE)   >> 1) * 2 |
               (self._status[rear_ids] & SWERVE))
        need = _COLLISION_TABLE[key]
        if need.any():
            crash = np.concatenate([rear_ids[need], front_ids[need]])
            self._status[crash] |= CRASHED
            self._lane[crash] = -1



def build_default_grid(N: int = 20, vavg: float = 45.) -> List[Car]:
    half = N // 2
    cars: List[Car] = []
    for i in range(half):
        theta = i * 2 * np.pi / N
        cars.append(Car(cid=i,        lane=1, theta=theta, vel=vavg))
        cars.append(Car(cid=i + half, lane=3, theta=theta, vel=vavg))
    return cars

if __name__ == "__main__":
    np.seterr(all="raise")
    sim = LaneSim(build_default_grid())
    for _ in range(int(25 / DT)):
        sim.step()
    print(f"Finished – {np.sum(sim._lane < 0)} crashes / {sim.N} cars")
