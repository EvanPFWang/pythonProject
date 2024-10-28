import numpy as np
import matplotlib.pyplot as plt

# Simulation Constants
N = 20  # Total cars
vmax = 60
vavg = 45
drsmax = 110
lane_width = 10  # Distance between lanes
R_base = 1000 / (2 * np.pi)  # Base radius for the innermost lane
radii = np.array([R_base + i * lane_width for i in range(5)])  # Radii for 5 lanes
dt = 0.01
Tf = 25
stop_count = int(Tf / dt)

# Track lengths based on lane radii
track_lengths = 2 * np.pi * radii

# Reference matrices for lane positions
refI1 = np.zeros((5, N // 2), dtype=int)
refI2 = np.zeros((5, N // 2), dtype=int)

# Populate ref matrices with specific ordering for initial car positions
for i in range(N // 2):
    refI1[1, i] = N // 2 - i  # Lane 2, car in reverse order from 10 to 1
    refI1[3, i] = N - i  # Lane 4, car in reverse order from 20 to 11

    refI2[1, i] = N // 2 - i
    refI2[3, i] = N - i


# Car class definition
class Car:
    def __init__(self, car_id, lane, theta, velocity, max_velocity, status=0):
        self.id = car_id
        self.lane = lane
        self.radius = radii[lane - 1]  # Assign radius based on lane
        self.theta = theta
        self.velocity = velocity
        self.max_velocity = max_velocity
        self.status = status  # -1 if crashed, 0 if active

    def update_position(self, dt):
        # Update theta based on current velocity
        self.theta += self.velocity * dt / self.radius  # Account for radius in theta change
        if self.theta >= 2 * np.pi:
            self.theta -= 2 * np.pi  # Wrap around to simulate circular track

    def info(self):
        return self.id, self.lane, self.theta, self.velocity, self.status


# Initialize cars with positions based on refI1 and refI2
cars = []
for i in range(N // 2):
    # Lane 2 cars
    cars.append(Car(refI1[1, i], 2, i * 2 * np.pi / N, vavg, vmax))
    # Lane 4 cars
    cars.append(Car(refI1[3, i], 4, i * 2 * np.pi / N, vavg, vmax))

# Initialize car_info matrix for quick access to car properties
car_info = np.zeros((N, 6))  # [id, lane, theta, velocity, decision, status]
for i, car in enumerate(cars):
    car_info[i, :4] = car.id, car.lane, car.theta, car.velocity

# Simulation loop
for step in range(stop_count):
    # Update car positions
    for i, car in enumerate(cars):
        if car.status != -1:
            car.update_position(dt)  # Update car position based on velocity
            car_info[i, 2] = car.theta  # Update theta in car_info for plotting

# Visualization of initial car positions with lanes
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
thetas = np.linspace(0, 2 * np.pi, 100)

# Plot track lanes
for radius in radii:
    ax.plot(radius * np.cos(thetas), radius * np.sin(thetas), lw=0.5)

# Plot initial car positions
for car in cars:
    x = car.radius * np.cos(car.theta)
    y = car.radius * np.sin(car.theta)
    ax.plot(x, y, 'o', label=f"Car {car.id}" if car.id <= 2 else "")  # Show labels only for first few cars

plt.legend(loc='upper right')
plt.show()