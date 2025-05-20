import random
import time
import numpy as np

def error(current_angle, current_velocity):
    e_theta = 0 - current_angle
    e_vel = -0.5 - current_velocity
    return e_theta, e_vel

def ut(dt, e_theta, prev_e_theta, i_theta, utI_reset):
    if abs(e_theta) <= 1 and utI_reset == False:
        # if the error is small, it will reset the integral to allow small updates
        # But only once to allow re-build up of small error
        i_theta = 0
        utI_reset = True
    else:
        i_theta += e_theta * dt
        if i_theta > 20:
            i_theta = 20
        elif i_theta < -20:
            i_theta = -20

    d_theta = (e_theta - prev_e_theta) / dt

    u_theta = k_theta[0] * e_theta + k_theta[1] * i_theta + k_theta[2] * d_theta
    return i_theta, u_theta, utI_reset

def uv(dt, e_vel, prev_e_vel, i_vel, uvI_reset):
    if abs(e_vel) <= 1 and uvI_reset == False:
        # if the error is small, it will reset the integral to allow small updates
        # But only once to allow re-build up of small error
        i_vel = 0
        uvI_reset = True
    else:
        i_vel += e_vel * dt
        if i_vel > 20:
            i_vel = 20
        elif i_vel < -20:
            i_vel = -20

    d_vel = (e_vel - prev_e_vel) / dt

    u_vel = k_vel[0] * e_vel + k_vel[1] * i_vel + k_vel[2] * d_vel
    return i_vel, u_vel, uvI_reset

k_theta = [0.2, 0.05, 0.1]
k_vel = [1, 0.05, 0.1]
k_theta = np.array(k_theta)
k_vel = np.array(k_vel)

mass = 20#kg

i_theta = 0
i_vel = 0

# Initialize parameters
stable = False
start_time = time.time()

current_angle = random.randint(-45,45) # Randomize starting angle
current_velocity = random.randint(-500, -300) # Randomize starting velocity

prev_e_theta, prev_e_vel = error(current_angle, current_velocity)

utI_reset = False
uvI_reset = False

angular_velocity = 0

velocity = 0
displacement = 0

counter = 0 # Used to detect if the system is still oscillating or reached stability.
latest_20 = [[],[]]
oscillating = True
while stable == False: # PID controller initialises here
    time.sleep(0.03) # Simulate real world hardware latency
    end_time = time.time()
    dt = end_time - start_time
    start_time = time.time()

    e_theta, e_vel = error(current_angle, current_velocity)

    if e_theta > 1 or e_theta < -1:
        utI_reset = False
    i_theta, u_theta, utI_reset = ut(dt, e_theta, prev_e_theta, i_theta, utI_reset)

    if e_vel > 1 or e_vel < -1:
        uvI_reset = False
    i_vel, u_vel, uvI_reset = uv(dt, e_vel, prev_e_vel, i_vel, uvI_reset)

    angular_acceleration = u_theta / mass
    angular_velocity += angular_acceleration * dt # += is used instead of = because you want the current velocity, which is a result of all previous velocities added todether.
    d_angle = angular_velocity * dt # total change in angle from start would use += instead of =
    current_angle += d_angle
    current_angle *= 0.99 # Damping factor, prevents infinite oscillation

    u_vel += mass * 9.8 # u alone wasn't strong enough to overcome gravity, so I have implemented a baseline thrust
    acceleration = u_vel / mass
    d_velocity = acceleration * dt
    current_velocity += -9.8 * dt
    current_velocity += d_velocity
    current_velocity *= 0.99 # Damping factor, prevents infinite oscillation

    prev_e_theta = e_theta
    prev_e_vel = e_vel

    print(f"Velocity: {current_velocity}, angle: {current_angle}")

    latest_20[0].append(current_angle)
    latest_20[1].append(current_velocity)
    counter += 1
    if counter > 19:
        if abs(max(latest_20[0]) - min(latest_20[0])) <= 1 and abs(max(latest_20[1]) - min(latest_20[1])) <= 1:
            oscillating = False
        else:
            oscillating = True
        latest_20 = [[],[]]
        counter = 0

    if abs(e_theta) <= 1 and abs(e_vel) <= 5 and oscillating == False:
        print("Rocket is in stable descent position")
        stable = True