import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from dataclasses import dataclass, field
from typing import List, Tuple

#core model

R_BASE = 1000 / (2 * np.pi)
LANE_WIDTH = 10.0
LANES = 5
DT = 1e-2
V_MAX = 70.0
THETA_VIEW = np.pi / 4
RADII = R_BASE + np.arange(LANES) * LANE_WIDTH
TRACK_LENGTHS = 2 * np.pi * RADII

SWERVE  = np.uint8(0b001)
BRAKE   = np.uint8(0b010)
CRASHED = np.uint8(0b100)


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


#16‑way collision truth‑table (unchanged)
_COLLISION_TABLE = np.zeros(16, dtype=bool)
for key in range(16):
    rs = key & np.uint8(0b0001)
    rb = key & np.uint8(0b0010)
    fb = key & np.uint8(0b0100)
    fc = key & np.uint8(0b1000)
    _COLLISION_TABLE[key] = bool(fc) or (fb and not rb)


class LaneSim:
    """NumPy‑accelerated traffic model with 8‑mode peek matrix."""

    def __init__(self, cars: List[Car]):
        self.cars = cars
        self.N = len(cars)
        self._lane   = np.array([c.lane  for c in cars], dtype=np.int16)
        self._theta  = np.array([c.theta for c in cars], dtype=np.float32)
        self._vel    = np.array([c.vel   for c in cars], dtype=np.float32)
        self._status = np.array([c.status for c in cars], dtype=np.uint8)
        self._theta_total = np.zeros(self.N, dtype=np.float32)
        self._idx = np.arange(self.N)

#public
    def step(self):
        active = self._lane >= 0
        lane_idx = self._lane[active]
        fr = self._peek_nearest(mode=1)
        new_speed, lane_delta = self._drive(fr)
        self._apply_decision(new_speed, lane_delta)
        dθ = meters_per_sec_to_theta(self._vel[active], lane_idx)
        self._theta[active] = wrap_theta(self._theta[active] + dθ)
        self._theta_total[active] += dθ
        self._crash_resolution()
        self._post_lane_change_collision(fr)
#sync back to dataclass (optional for external use)
        for i, c in enumerate(self.cars):
            c.lane, c.theta, c.vel, c.status = int(self._lane[i]), float(self._theta[i]), float(self._vel[i]), int(self._status[i])

#peek (nearest only)
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
        else:
            raise NotImplementedError("Only mode 1 used in this demo")
        return vis

    def _peek_nearest(self, mode: int = 1):
        dθ, same_lane = self._compute_distance_matrices()
        vis = self._visibility_mask(mode, dθ, same_lane)
        dist = np.abs(dθ)
        dist[~vis] = np.inf
        idx = np.argmin(dist, axis=1)
        idx[np.isinf(dist[np.arange(self.N), idx])] = -1
        return idx.astype(np.int16)

#drive / decision
    def _drive(self, fr):
        θ_max = meters_per_sec_to_theta(np.full(self.N, V_MAX, float), self._lane)
        θ2n = np.zeros(self.N, float)
        valid = fr >= 0
        θ2n[valid] = theta_distance(self._theta[valid], self._theta[fr[valid]])
        θ_min = θ_max * 0.05
        new_speed = np.where(θ2n <= θ_min, self._vel, V_MAX * np.tanh(θ2n / θ_max))
        left  = np.where(self._lane > 0, θ_max[self._lane - 1], -1)
        mid   = θ_max
        right = np.where(self._lane < LANES - 1, θ_max[self._lane + 1], -1)
        lane_delta = np.vstack([left, mid, right]).argmax(axis=0) - 1
        return new_speed, lane_delta.astype(np.int8)

    def _apply_decision(self, new_speed, lane_delta):
        swap = lane_delta != 0
        self._vel[:] = new_speed
        self._status &= ~np.uint8(~SWERVE & 0xFF)
        self._status |= swap.astype(np.uint8) * SWERVE
        self._lane += lane_delta
        np.clip(self._lane, 0, LANES - 1, out=self._lane)

#crash passes (same as previous)
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



#animation helper

def animate(sim: LaneSim, steps: int = 250, interval: int = 20, save: str = "traffic.gif"):
    """Create an animation of *steps* simulation ticks (~steps*DTseconds).

    Parameters
    sim : LaneSim            – a fully‑initialised simulator
    steps : int              – number of time‑steps to animate
    interval : int           – delay between frames (ms) in the resulting video
    save : str               – filename for the GIF/MP4 (extension decides)"""

#figure setup
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-(RADII[-1] + 5), RADII[-1] + 5)
    ax.set_ylim(-(RADII[-1] + 5), RADII[-1] + 5)
    ax.set_aspect("equal")
    ax.axis("off")

#draw lane circles (static background)
    for r in RADII:
        circle = plt.Circle((0, 0), r, color="lightgray", fill=False, lw=0.5)
        ax.add_patch(circle)

#scatter for cars
    scat = ax.scatter([], [], s=20, c="tab:blue", edgecolors="k")

    def init():
        scat.set_offsets(np.empty((0, 2)))
        return scat,

    def update(frame):
        sim.step()
        x = (RADII[sim._lane] * np.cos(sim._theta))
        y = (RADII[sim._lane] * np.sin(sim._theta))
        scat.set_offsets(np.column_stack([x, y]))
        colors = np.where(sim._status & CRASHED, "red", "tab:blue")
        scat.set_color(colors)
        return scat,

    movie = ani.FuncAnimation(fig, update, frames=steps, init_func=init,
                              interval=interval, blit=True)

#auto‑choose writer based on file extension
    if save.endswith(".gif"):
        movie.save(save, writer="pillow", fps=1000//interval)
    else:
        movie.save(save, writer="ffmpeg", fps=1000//interval)
#plt.close(fig)
    return save

def build_default_grid(
        N: int = 20,
        vavg: float = 45.,
        seed: int | None = None
) -> List[Car]:

    rng = np.random.default_rng(seed)#modern, thread‑safe RNG
    thetas = np.linspace(-np.pi, np.pi, N, endpoint=False, dtype=np.float32)
    lanes = np.arange(N) % LANES#0,1,2,3,4,0,1,…
    speeds = np.clip(rng.normal(vavg, 0.05 * vavg, N), 0, V_MAX)

    return [
        Car(
            cid=i,
            lane=int(lane),
            theta=float(theta),
            vel=float(speed)
        )
        for i, (lane, theta, speed) in enumerate(zip(lanes, thetas, speeds))
    ]

#convenience entry‑point – run a race & save animation
def main():
    """
    Quick demo: 60 simulated seconds (≈6000 ticks) written to traffic.gif.
    Comment‑out the animate() call if you just need the data‑side simulation.
    """
    cars = build_default_grid(N=40, vavg=50., seed=0)
    sim = LaneSim(cars)
    animate(sim, steps=6000, interval=16, save="traffic.gif")#≈60s at 60fps


if __name__ == "__main__":
    cars = build_default_grid(N=40, vavg=50., seed=0)
    sim  = LaneSim(cars)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-(RADII[-1] + 5), RADII[-1] + 5)
    ax.set_ylim(-(RADII[-1] + 5), RADII[-1] + 5)
    ax.set_aspect("equal");  ax.axis("off")

    for r in RADII:#static lane circles
        ax.add_patch(plt.Circle((0, 0), r, color="lightgray",
                                lw=0.5, fill=False))

    scat = ax.scatter([], [], s=20, c="tab:blue", edgecolors="k")

    def init():
        scat.set_offsets(np.empty((0, 2)))
        return scat,#must return a tuple

    def update(frame):
        sim.step()#advance the model
        x = RADII[sim._lane] * np.cos(sim._theta)
        y = RADII[sim._lane] * np.sin(sim._theta)
        scat.set_offsets(np.column_stack([x, y]))
        scat.set_color(np.where(sim._status & CRASHED, "red", "tab:blue"))
        return scat,

    anim = ani.FuncAnimation(fig, update, init_func=init,
                             frames=6000, interval=16, blit=True)

    plt.show()#preview
    anim.save("traffic.gif", writer="pillow", fps=60)