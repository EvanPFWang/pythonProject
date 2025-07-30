
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from dataclasses import dataclass, field
from typing import List, Tuple


# base radius for the innermost lane – this defines the size of the circular
# track.  Each subsequent lane is offset outwards by   LANE_WIDTH  .
R_BASE = 1000.0 / (2 * np.pi)

# Distance between adjacent lanes.  Five lanes are used in total.
LANE_WIDTH = 10.0

# number of lanes in the simulation
LANES = 5

# integration timestep in seconds.  Each call to   LaneSim.step   will
# advance the simulation by this amount of real time.
DT = 1e-2

# maximum allowed vehicle speed in metres per second.  This is used as a
# target speed; individual cars may drive slower if necessary to maintain a
# safe following distance.
V_MAX = 70.0

# angular field of view for the more advanced peek modes.  In this simplified
# implementation only the nearest neighbour within a very narrow forward cone
# is considered, so   THETA_VIEW   is not actually used.  It is kept here to
# mirror the original design and permit future expansion.
THETA_VIEW = np.pi / 4

# precompute the radius of each lane.  RADII[0] corresponds to lane 0, the
# innermost; RADII[4] corresponds to lane 4, the outermost.
RADII = R_BASE + np.arange(LANES) * LANE_WIDTH

# total circumference of each lane.  These values are used when converting
# between linear speed (m/s) and angular speed (rad/s) around the track.
TRACK_LENGTHS = 2 * np.pi * RADII

# bit flags used to encode certain status conditions for each car.    SWERVE
# marks a lane change,   BRAKE   would mark a braking manoeuvre (not used
# directly in this simplified implementation), and   CRASHED   marks cars
# involved in a collision.
SWERVE: np.uint8 = np.uint8(0b001)
BRAKE: np.uint8 = np.uint8(0b010)
CRASHED: np.uint8 = np.uint8(0b100)


def wrap_theta(theta: np.ndarray) -> np.ndarray:
    return (theta + np.pi) % (2 * np.pi) - np.pi


def theta_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return wrap_theta(b - a)


def meters_per_sec_to_theta(v: np.ndarray, lane_idx: np.ndarray) -> np.ndarray:
    return v * DT * 2 * np.pi / TRACK_LENGTHS[lane_idx]


@dataclass
class Car:
    """Dataclass representing a single vehicle in the simulation."""

    cid: int
    lane: int
    theta: float
    vel: float
    status: int = 0
    theta_total: float = 0.0
    radius: float = field(init=False)

    def __post_init__(self) -> None:
        # assign the radius based on the lane; negative lane values indicate
        # crashed cars which will not be drawn.  Use NaN for those to avoid
        # indexing errors.
        self.radius = RADII[self.lane] if self.lane >= 0 else np.nan


# Construct a 16‑entry truth table encoding which combinations of vehicle
# behaviours (swerve, brake, crash) should result in a collision when a
# following car meets a leading car.  This table derives from the original
# MATLAB code's bitwise logic.  Each key from 0b0000 to 0b1111 indexes
# whether a crash occurs for that scenario.
_COLLISION_TABLE: np.ndarray = np.zeros(16, dtype=bool)
for key in range(16):
    rs = key & np.uint8(0b0001)  # following car is swerving?
    rb = key & np.uint8(0b0010)  # following car is braking?
    fb = key & np.uint8(0b0100)  # leading car is braking?
    fc = key & np.uint8(0b1000)  # leading car has crashed?
    # A collision happens if the leading car has already crashed, or if
    # the following car is braking (fb) and the leading car is not braking (rb).
    _COLLISION_TABLE[key] = bool(fc) or (fb and not rb)


class LaneSim:
    """NumPy‑accelerated traffic model with simple peek and driving logic."""

    def __init__(self, cars: List[Car]) -> None:
        # Store a reference to the Car objects for synchronising back to
        # user‑visible state.  Internally, use NumPy arrays for fast vectorised
        # computation.
        self.cars = cars
        self.N = len(cars)
        #  _lane   holds the current lane index for each car (0 to 4 for
        # active cars; ‑1 for crashed cars).    _theta   is the angular
        # position on the track in radians.    _vel   is the linear speed in
        # metres per second.    _status   holds bit flags for SWERVE/BRAKE/
        # CRASHED.    _theta_total   accumulates the total angle travelled.
        self._lane = np.array([c.lane for c in cars], dtype=np.int16)
        self._theta = np.array([c.theta for c in cars], dtype=np.float32)
        self._vel = np.array([c.vel for c in cars], dtype=np.float32)
        self._status = np.array([c.status for c in cars], dtype=np.uint8)
        self._theta_total = np.zeros(self.N, dtype=np.float32)
        self._idx = np.arange(self.N)

    # Public API ----------------------------------------------------------------
    def step(self) -> None:
        """Advance simulation by a time step.

        Cars accelerate or decelerate based on the distance to the next
        vehicle ahead in their lane, may decide to change lanes if a
        neighbouring lane allows a higher target speed, and may crash if
        they collide.  After computing the new state, the underlying Car
        objects are updated so that external callers (like the animation
        routine) always reflect the latest positions and statuses.
        """
        # determine which cars are still active (i.e., have not crashed).
        active = self._lane >= 0
        # lane indices for the active cars.
        lane_idx = self._lane[active]
        # determine the index of the nearest car ahead in the same lane.
        fr = self._peek_nearest(mode=1)
        # compute new target speeds and lane change intentions.
        new_speed, lane_delta = self._drive(fr)
        # apply lane changes and update speeds.
        self._apply_decision(new_speed, lane_delta)
        # convert speeds to angular increments and update positions.
        dtheta = meters_per_sec_to_theta(self._vel[active], lane_idx)
        self._theta[active] = wrap_theta(self._theta[active] + dtheta)
        self._theta_total[active] += dtheta
        # resolve head‑on collisions within the same lane.
        self._crash_resolution()
        # resolve collisions immediately following a lane change.
        self._post_lane_change_collision(fr)
        # synchronise the dataclass objects with the internal arrays so that
        # external references reflect the new simulation state.
        for i, c in enumerate(self.cars):
            c.lane = int(self._lane[i])
            c.theta = float(self._theta[i])
            c.vel = float(self._vel[i])
            c.status = int(self._status[i])

    # Internal helpers ----------------------------------------------------------
    def _compute_distance_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute pairwise angular distances and lane equality masks."""
        dtheta = theta_distance(self._theta[:, None], self._theta[None, :])
        same_lane = self._lane[:, None] == self._lane[None, :]
        return dtheta, same_lane

    def _visibility_mask(self, mode: int, dtheta: np.ndarray, same_lane: np.ndarray) -> np.ndarray:
        """Determine which other cars are within the field of view of each car.

        Only mode 1 (nearest ahead in the same lane) is implemented in this
        simplified version.  Other modes from the original design would
        consider different angular ranges or include neighbouring lanes.
        """
        front_n = (0 < dtheta) & (dtheta < np.pi / 16)
        if mode == 1:
            vis = same_lane & front_n
        else:
            raise NotImplementedError("Only mode 1 (nearest ahead) is implemented")
        return vis

    def _peek_nearest(self, mode: int = 1) -> np.ndarray:
        """Find the index of the nearest vehicle ahead for each car.

        Parameters

        mode : int, optional
            Specifies the peeking strategy.  Only mode 1 is available.

        Returns

        np.ndarray
            An array of indices into   self.cars   giving the nearest
            vehicle ahead in the same lane, or -1 if none are in view.
        """
        dtheta, same_lane = self._compute_distance_matrices()
        vis = self._visibility_mask(mode, dtheta, same_lane)
        # compute absolute angular distance to all other vehicles, mask out
        # those that are not visible, then find the nearest.  Use np.inf for
        # non-visible so they are never selected by argmin.
        dist = np.abs(dtheta)
        dist[~vis] = np.inf
        idx = np.argmin(dist, axis=1)
        # mark as -1 if no vehicle was found (distance remained inf).
        idx[np.isinf(dist[np.arange(self.N), idx])] = -1
        return idx.astype(np.int16)

    def _drive(self, fr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute new target speeds and lane change decisions.

        looks at the distance to the nearest vehicle ahead in
        the current lane and adjusts speed accordingly.  It also compares
        the theoretical maximum angular speed in the current lane with the
        adjacent lanes and chooses a lane change if a neighbouring lane
        offers a higher theta_max.  Cars with no one ahead accelerate
        smoothly up to   V_MAX  .

        Parameters

        fr : np.ndarray
            Indices of the nearest car ahead for each car (or -1 if none).

        Returns

        Tuple[np.ndarray, np.ndarray]
            The new speed for each car (linear m/s) and the lane delta
            (-1 for left, 0 for staying, +1 for right).
        """
        # precompute the maximum angular speed for each lane.  This is the
        # angular increment corresponding to a car travelling at V_MAX on that
        # lane in one timestep.
        theta_max_lane = V_MAX * DT * 2 * np.pi / TRACK_LENGTHS

        # prepare arrays for per‑car theta_max and lane change options.  Use
        # NaN/negative values for crashed cars so that they do not influence
        # the argmax logic.
        theta_max_per_car = np.full(self.N, np.nan, dtype=float)
        # distances to the next car ahead (in angular units)
        theta2n = np.full(self.N, np.inf, dtype=float)
        # safe distance thresholds (5% of theta_max) per car
        safe_dist = np.full(self.N, np.inf, dtype=float)
        # lane change options: left, stay, right.  Use a default of -1 so
        # these lanes are not chosen for crashed cars.
        left_option = np.full(self.N, -1.0, dtype=float)
        mid_option = np.full(self.N, -1.0, dtype=float)
        right_option = np.full(self.N, -1.0, dtype=float)

        # Identify active (non‑crashed) cars
        active_mask = self._lane >= 0
        if not np.any(active_mask):
            # all cars are crashed; return zeros to avoid divide by zero
            return np.zeros(self.N, dtype=float), np.zeros(self.N, dtype=np.int8)

        # for active cars assign theta_max and safe_dist
        theta_max_per_car[active_mask] = theta_max_lane[self._lane[active_mask]]
        safe_dist[active_mask] = theta_max_per_car[active_mask] * 0.05
        # Compute distance to the next car ahead for active cars
        valid = (fr >= 0) & active_mask
        theta2n[valid] = theta_distance(self._theta[valid], self._theta[fr[valid]])
        theta2n[valid] = np.maximum(theta2n[valid], 0.0)

        # compute acceleration factor using tanh.  Avoid divide by zero by
        # suppressing warnings and replacing NaNs with zero.
        with np.errstate(divide='ignore', invalid='ignore'):
            accel = np.tanh(theta2n / theta_max_per_car)
        accel = np.nan_to_num(accel)
        target_speed = V_MAX * accel
        # maintain current speed if too close; otherwise adopt target speed.
        new_speed = np.where(theta2n <= safe_dist, self._vel, target_speed)
        # crashed cars should not move
        new_speed[~active_mask] = 0.0

        # populate lane change options only for active cars
        # left lane: only valid if lane > 0
        left_idx = np.where((self._lane > 0) & active_mask)[0]
        left_option[left_idx] = theta_max_lane[self._lane[left_idx] - 1]
        # middle lane: current lane
        mid_option[active_mask] = theta_max_lane[self._lane[active_mask]]
        # right lane: only valid if lane < LANES-1
        right_idx = np.where((self._lane < LANES - 1) & active_mask)[0]
        right_option[right_idx] = theta_max_lane[self._lane[right_idx] + 1]
        # determine the lane change: choose the lane with the highest theta_max
        # choose the lane with the highest theta_max; however, to prevent all
        # vehicles drifting into the innermost lane (which has the highest
        # angular speed), we disable lane changes in this simplified model.
        # Set lane_delta to zero for all cars.  Should more nuanced lane
        # change logic be desired, this line can be replaced with the
        # argmax computation below:
        # lane_delta = np.vstack([left_option, mid_option, right_option]).argmax(axis=0) - 1
        lane_delta = np.zeros(self.N, dtype=np.int8)
        # Do not change lane for crashed cars (already zero)
        return new_speed, lane_delta

    def _apply_decision(self, new_speed: np.ndarray, lane_delta: np.ndarray) -> None:
        """Apply the lane change and speed update to all cars."""
        # mark which cars are changing lanes
        swap = lane_delta != 0
        # update velocities
        self._vel[:] = new_speed
        # clear any previous SWERVE bits then set if swapping
        self._status &= ~np.uint8(~SWERVE & 0xFF)
        self._status |= swap.astype(np.uint8) * SWERVE
        # update lane indices, clipping to the valid range [0, LANES-1]
        self._lane += lane_delta
        np.clip(self._lane, 0, LANES - 1, out=self._lane)

    def _crash_resolution(self) -> None:
        """Detect and handle collisions between cars in the same lane."""
        for lane in range(LANES):
            idx = np.where(self._lane == lane)[0]
            if idx.size < 2:
                continue
            # sort cars by angular position around the track
            order = np.argsort(self._theta[idx])
            idx_sorted = idx[order]
            # Compute the angular gap between successive cars
            gap = theta_distance(self._theta[idx_sorted[:-1]], self._theta[idx_sorted[1:]])
            # A collision happens if the gap is less than a threshold
            threshold = 2 * np.pi * 2 / TRACK_LENGTHS[lane]
            collide = np.where(gap < threshold)[0]
            if collide.size > 0:
                hit = idx_sorted[collide]
                vic = idx_sorted[collide + 1]
                self._status[np.concatenate([hit, vic])] |= CRASHED
                self._lane[np.concatenate([hit, vic])] = -1

    def _post_lane_change_collision(self, fr: np.ndarray) -> None:
        """Handle collisions occurring immediately after lane changes."""
        rear_ids = self._idx
        front_ids = fr
        mask = front_ids >= 0
        rear_ids = rear_ids[mask]
        front_ids = front_ids[mask]
        # onstr a lookup key for the collision table based on the status
        key = (
                ((self._status[front_ids] & CRASHED) >> 2) * 8
                | ((self._status[front_ids] & BRAKE) >> 1) * 4
                | ((self._status[rear_ids] & BRAKE) >> 1) * 2
                | (self._status[rear_ids] & SWERVE)
        )
        need = _COLLISION_TABLE[key]
        if need.any():
            crash = np.concatenate([rear_ids[need], front_ids[need]])
            self._status[crash] |= CRASHED
            self._lane[crash] = -1


# Animation helper -------------------------------------------------------------
def animate(
        sim: LaneSim, steps: int = 250, interval: int = 20, save: str = "traffic.gif"
) -> str:
    """Create an animation of steps simulation ticks and save it.

    Parameters
    sim : LaneSim
        A fully initialised simulator instance.
    steps : int, optional
        Number of time steps to animate.  Each step corresponds to a
        simulation advance of DT seconds.  Defaults to 250.
    interval : int, optional
        Delay between frames in milliseconds.  Controls the playback speed
        of the resulting GIF.  Defaults to 20 ms.
    save : str, optional
        Filename for the output GIF or MP4.  The file extension determines
        which writer is used.  Defaults to traffic.gif.

    Returns
    -------
    str
        The filename of the saved animation.
    """
    # Set up the figure and axis for drawing
    fig, ax = plt.subplots(figsize=(6, 6))
    # define plot limits slightly larger than the outermost lane
    margin = 5.0
    ax.set_xlim(-(RADII[-1] + margin), RADII[-1] + margin)
    ax.set_ylim(-(RADII[-1] + margin), RADII[-1] + margin)
    ax.set_aspect("equal")
    ax.axis("off")

    # draw static lane circles
    for r in RADII:
        circle = plt.Circle((0, 0), r, color="lightgray", fill=False, lw=0.5)
        ax.add_patch(circle)

    # create scatter plot for car positions and edge colours for better
    # visibility + colors will be updated dynamically based on crash status.
    scat = ax.scatter([], [], s=20, c="tab:blue", edgecolors="k")

    def init() -> Tuple[ani.ArtistAnimation]:
        # init scatter plot with no points
        scat.set_offsets(np.empty((0, 2)))
        return (scat,)

    def update(frame: int) -> Tuple[ani.ArtistAnimation]:
        # advance by one step
        sim.step()
        # compute x/y positions only for active cars (lane >= 0)
        # crashed cars are not drawn
        # negative lane indices are masked out.
        active_mask = sim._lane >= 0
        x = np.empty(sim.N, dtype=float)
        y = np.empty(sim.N, dtype=float)
        # For active cars, compute positions based on their lane radius
        x[active_mask] = RADII[sim._lane[active_mask]] * np.cos(sim._theta[active_mask])
        y[active_mask] = RADII[sim._lane[active_mask]] * np.sin(sim._theta[active_mask])
        # For crashed cars, fill with NaN so they are ignored by the scatter
        x[~active_mask] = np.nan
        y[~active_mask] = np.nan
        scat.set_offsets(np.column_stack([x, y]))
        # Update colours: crashed cars turn red, others remain blue
        colors = np.where(sim._status & CRASHED, "red", "tab:blue")
        scat.set_color(colors)
        return (scat,)

    # create animation
    movie = ani.FuncAnimation(
        fig,
        update,
        frames=steps,
        init_func=init,
        interval=interval,
        blit=True,
    )

    # save the animation using an appropriate writer based on the filename
    if save.lower().endswith(".gif"):
        movie.save(save, writer="pillow", fps=1000 // interval)
    else:
        movie.save(save, writer="ffmpeg", fps=1000 // interval)
    # return the filename for convenience
    return save


def build_default_grid(N: int = 20, vavg: float = 45.0, seed: int | None = None) -> List[Car]:
    """cars evenly  assigned to lanes in a repeating pattern (0,1,2,3,4,0,1,...)
    Speeds are drawn from a normal distribution centred on vavg with
    5% standard deviation and clipped to the range [0, V_MAX].

    Parameters
    N : int, optional
        Total number of cars to generate.  Defaults to 20.
    vavg : float, optional
        The mean linear speed for the initial distribution.  Defaults to 45 m/s.
    seed : int | None, optional
        Seed for the random number generator to ensure reproducible results.
        Defaults to   None  .

    Returns
    List[Car]
        A list of Car instances ready for simulation.
    """
    rng = np.random.default_rng(seed)
    # evenly spread
    thetas = np.linspace(-np.pi, np.pi, N, endpoint=False, dtype=np.float32)
    # assign lanes round‑robinly and  draw initial speeds from a normal distribution and clamp to [0, V_MAX]
    lanes = np.arange(N) % LANES
    speeds = np.clip(rng.normal(vavg, 0.05 * vavg, N), 0.0, V_MAX)
    return [Car(cid=i, lane=int(lane), theta=float(theta), vel=float(speed)) for i, (lane, theta, speed) in enumerate(zip(lanes, thetas, speeds))]


def main() -> None:
    """Run a quick simulation and write an animation to disk."""
    cars = build_default_grid(N=20, vavg=50.0, seed=0)
    sim = LaneSim(cars)
    animate(sim, steps=1000, interval=16, save="traffic.gif")


if __name__ == "__main__":
    main()