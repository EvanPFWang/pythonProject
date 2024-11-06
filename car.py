import numpy as np

class Car:
    # Class Constants
    rMin    = 1000 / (2 * np.pi)
    r = [rMin, rMin + 1, rMin + 2, rMin + 3, rMin + 4]
    dt = 1e-2

    def __init__(self, identity, lane, theta, vel, maxVel, status):
        self.id = identity
        self.lane = lane
        self.radius = radii[lane - 1]  # Assign radius based on lane
        self.theta = theta
        self.thetaT = 0
        self.vel = vel
        self.max_vel = n
        self.status = status
        self.ref = None  # cars in frame of reference (customizable based on context)

    def execute1(self, car_info, places, lane_count):
        # Perform pre-drive decision-making
        vmax = 70
        obj_theta = self.theta
        fr1, _ = self.peek2(car_info, places, 1, obj_theta, lane_count)
        next_car_pre_drive, next_car_post_drive, newv, objl, decision = self.drive(car_info, fr1, vmax, self.lane,
                                                                                   obj_theta)
        return next_car_pre_drive, next_car_post_drive, newv, objl, decision

    def execute2(self, car_id, car_info, execute1, lane_count):
        # Perform post-decision changes
        L = 2 * np.pi * np.array(self.r)
        next_car_pre_drive = execute1[0]
        next_car_post_drive = execute1[1]
        newv = execute1[2]
        objl = execute1[3]
        decision = execute1[4]

        new_lane = decision + objl
        lane_count[objl] -= 1
        lane_count[new_lane] += 1
        self.vel = newv

        obj_lane_length = L[new_lane]
        dtheta = newv * 2 * np.pi * self.dt / obj_lane_length
        self.theta = car_info[car_id, 2] + dtheta - 2 * np.pi * (car_info[car_id, 2] + dtheta > np.pi)
        self.thetaT += dtheta

        changes = [car_id, new_lane, self.theta, newv, decision, not objl == new_lane]
        return next_car_pre_drive, next_car_post_drive, changes, lane_count

    def execute3(self, car_id, car_info, newplaces, execute2, lane_count):
        # Final adjustments and check for crashes
        next_car_pre_drive = execute2[0]
        next_car_post_drive = execute2[1]
        fr2, _ = self.peek(car_info, newplaces, 1, self.theta)
        old_lane = execute2[3]
        changes = execute2[2]
        next_car_post_ex = fr2[changes[1], 0]
        car1, car2 = self.car_in_front2(next_car_pre_drive, next_car_post_drive, next_car_post_ex, newplaces, car_info,
                                        fr2, car_id, changes[2], old_lane, lane_count)
        return car1, car2 if car1 != 0 else (0, 0)

    def update_position(self, dt):
        # Update theta based on current velocity
        self.theta += self.velocity * dt / self.radius  # Account for radius in theta change
        if self.theta >= 2 * np.pi:
            self.theta -= 2 * np.pi  # Wrap around to simulate circular track
    def info(self):
        return self.id, self.lane, self.theta, self.velocity, self.status

    @staticmethod
    def car_in_front2(next_car_pre_ex, next_car_post_ex, newplaces, car_info, fr2, car_id, obj_theta, old_lane,
                      post_execution):
        L = 2 * np.pi * np.array([rMin + i for i in range(5)])
        crash_theta = 2 * np.pi * np.array([2, 2, 2, 2, 2]) / L
        lgc = int(1 * car_info[next_car_pre_ex, 5] + 3 * car_info[next_car_post_ex, 5] + 8 * car_info[car_id, 5])

        # Placeholders for logic to handle different cases based on `lgc` values
        car1, car2 = 0, 0
        if next_car_pre_ex == next_car_post_ex:
            if lgc <= 8:
                if lgc % 2 == 0:
                    if abs(lgc) == 2:
                        if lgc < 0:
                            # Action 1
                            pass
                        else:
                            # Action 2
                            pass
                    else:
                        # Default Action and Action 3
                        pass
                else:
                    if abs(lgc) == 1:
                        if lgc < 0:
                            # Action 1
                            pass
                        else:
                            # Action 2
                            pass
                    else:
                        # Actions for ±3
                        pass
            else:
                lgc -= 8
                if lgc % 2 == 0:
                    if abs(lgc) == 2:
                        if lgc < 0:
                            # Action 1
                            pass
                        else:
                            # Action 2
                            pass
                else:
                    # Further cases for odd values
                    pass
        return car1, car2

    # Additional helper methods
    @staticmethod
    def peek2(car_info, places, frame, obj_theta, lane_count):
        # Placeholder for the peek2 method logic
        return [], 0

    @staticmethod
    def drive(car_info, fr1, vmax, obj_lane, obj_theta):
        # Placeholder for the drive method logic
        return 0, 0, vmax, obj_lane, 0

    @staticmethod
    def speed(theta_next_vv, vmax, theta2n, theta_min, theta_max):
        if theta2n <= theta_min:
            return theta_next_vv
        elif theta2n >= theta_max:
            return vmax
        else:
            return np.log(theta2n / theta_min) / np.log(theta_max / theta_min) * (vmax - theta_next_vv) + theta_next_vv

    @staticmethod
    def theta2nDisp(obj_theta, al_theta):
        return (al_theta - 2 * np.pi * (al_theta > np.pi)) - (obj_theta - 2 * np.pi * (obj_theta > np.pi))

    @staticmethod
    def theta2nDisp_mat(obj_theta, matrix):
        return (matrix - 2 * np.pi * (matrix > np.pi)) - (obj_theta - 2 * np.pi * (obj_theta > np.pi))

    @staticmethod
    def conv_to_theta(pre_conv, lane_length):
        return pre_conv / lane_length * 2 * np.pi



import numpy as np

# Assuming theta2n_disp is already defined as:
# def theta2n_disp(obj_theta, al_theta):
#     theta_dist = (al_theta - 2 * np.pi * (al_theta > np.pi)) - (obj_theta - 2 * np.pi * (obj_theta > np.pi))
#     return theta_dist

def quicksort1(vector, car_info, obj_theta):
    """
    descending order
    """
    if len(vector) <= 1:
        return vector

    # Filter out zero elements
    vector = [v for v in vector if v != 0]

    # If empty after filtering, return as is
    if not vector:
        return []

    # Selecting the pivot as the middle element
    pivot_index = len(vector) // 2
    pivot = vector[pivot_index]

    # Partitioning the list based on custom comparator
    left = [v for v in vector if theta2n_disp(obj_theta, car_info[v, 2]) < theta2n_disp(obj_theta, car_info[pivot, 2])]
    middle = [v for v in vector if theta2n_disp(obj_theta, car_info[v, 2]) == theta2n_disp(obj_theta, car_info[pivot, 2])]
    right = [v for v in vector if theta2n_disp(obj_theta, car_info[v, 2]) > theta2n_disp(obj_theta, car_info[pivot, 2])]

    # Recursively apply quicksort1 on left and right partitions
    return quicksort1(right, car_info, obj_theta) + middle + quicksort1(left, car_info, obj_theta)

def quicksort2(vector, car_info, obj_theta):
    """
    ascending order
    """
    if len(vector) <= 1:
        return vector

    # Filter out zero elements
    vector = [v for v in vector if v != 0]

    # If vector is empty after filtering, return as is
    if not vector:
        return []

    # Selecting the pivot as the middle element
    pivot_index = len(vector) // 2
    pivot = vector[pivot_index]

    # Partitioning the list based on custom comparator
    left = [v for v in vector if theta2n_disp(obj_theta, car_info[v, 2]) > theta2n_disp(obj_theta, car_info[pivot, 2])]
    middle = [v for v in vector if theta2n_disp(obj_theta, car_info[v, 2]) == theta2n_disp(obj_theta, car_info[pivot, 2])]
    right = [v for v in vector if theta2n_disp(obj_theta, car_info[v, 2]) < theta2n_disp(obj_theta, car_info[pivot, 2])]

    # Recursively apply quicksort2 on left and right partitions
    return quicksort2(left, car_info, obj_theta) + middle + quicksort2(right, car_info, obj_theta)



def peek2(car_info, places, which, obj_theta, lane_count, objN, theta_view):
    sz = places.shape
    fr, ba = None, None

    if which < 3:
        if which == 1:
            # Front reference case
            fr_lists = [np.zeros(objN // 2) for _ in range(5)]
            fr_non_zero_counts = [0] * 5

            for i in range(sz[0]):
                if np.all(places[i, :] == 0):
                    continue
                row_indexer = 0
                for j in range(sz[1]):
                    stored = places[i, j]
                    if stored == 0 or car_info[stored, 1] == 0:
                        continue
                    theta_this = car_info[stored, 2]
                    if 0 < theta2nDisp(obj_theta, theta_this) < np.pi / 16:
                        fr_lists[i][row_indexer] = stored
                        fr_non_zero_counts[i] += 1
                        row_indexer += 1

            # Sorting and zero padding
                for i, count in enumerate(fr_non_zero_counts):
                    if count == 0:
                        fr_lists[i] = np.zeros(objN // 2)
                    else:
                        fr_lists[i] = np.concatenate([quicksort1(fr_lists[i][:count], car_info, obj_theta),
                                                      np.zeros(objN // 2 - count)])

            fr = np.vstack(fr_lists)
            ba = np.zeros((5, objN // 2))

        else:
            # Back reference case
            ba_lists = [np.zeros(objN // 2) for _ in range(5)]
            ba_non_zero_counts = [0] * 5

            for i in range(sz[0]):
                if np.all(places[i, :] == 0):
                    continue
                row_indexer = 0
                for j in range(sz[1]):
                    stored = places[i, j]
                    if stored == 0 or car_info[stored, 1] == 0:
                        continue
                    theta_this = car_info[stored, 2]
                    if -np.pi / 16 < theta2nDisp(obj_theta, theta_this) < 0:
                        ba_lists[i][row_indexer] = stored
                        ba_non_zero_counts[i] += 1
                        row_indexer += 1

                # Sorting and zero padding
                for i, count in enumerate(ba_non_zero_counts):
                    if count == 0:
                        ba_lists[i] = np.zeros(objN // 2)
                    else:
                        ba_lists[i] = np.concatenate([quicksort1(ba_lists[i][:count][::-1], car_info, obj_theta),
                                                      np.zeros(objN // 2 - count)])

            ba = np.vstack(ba_lists)
            fr = np.zeros((5, objN // 2))

    elif which > 5:
        # Complex cases (which == 6, 7, or 8)
        fr, ba = np.zeros(sz), np.zeros(sz)
        indexer_fr, indexer_ba = np.ones(5, dtype=int), np.ones(5, dtype=int)
        car_seen_fr, car_seen_ba = np.zeros(5, dtype=int), np.zeros(5, dtype=int)

        for j in range(objN):
            if car_info[j, 1] == 0:
                continue
            theta_this = car_info[j, 2]
            lane_this = car_info[j, 1]
            if abs(theta2nDisp(obj_theta, theta_this)) < theta_view:
                if theta2nDisp(obj_theta, theta_this) > 0:
                    car_seen_fr[lane_this - 1] += 1
                    fr[lane_this - 1, indexer_fr[lane_this - 1] - 1] = car_info[j, 0]
                    indexer_fr[lane_this - 1] += 1
                else:
                    car_seen_ba[lane_this - 1] += 1
                    ba[lane_this - 1, indexer_ba[lane_this - 1] - 1] = car_info[j, 0]
                    indexer_ba[lane_this - 1] += 1

        for i in range(sz[0]):
            if car_seen_fr[i] > 0:
                fr[i, :car_seen_fr[i]] = quicksort2(fr[i, :car_seen_fr[i]], car_info, obj_theta)
            if car_seen_ba[i] > 0:
                ba[i, :car_seen_ba[i]] = quicksort1(ba[i, :car_seen_ba[i]], car_info, obj_theta)
    else:
        if  which == 5:
            #which == 5
            if abs(theta_from_pi) > theta_view or theta_from_pi < 0:
                for i in range(sz[0]):
                    # Skip empty rows
                    if np.all(places[i, :] == 0):
                        continue

                    front_seen = False
                    for row_indexer1 in range(int(lane_count[i])):
                        theta_this = car_info[places[i, row_indexer1] - 1, 2]  # Adjust for zero-based index
                        distance_in_front = theta2n_disp(obj_theta, theta_this)

                        if abs(distance_in_front) < theta_view:
                            if distance_in_front > 0:
                                front_seen = True
                                for row_indexer2 in range(row_indexer1 + 1, int(lane_count[i])):
                                    theta_this = car_info[places[i, row_indexer2] - 1, 2]
                                    distance_in_front = theta2n_disp(obj_theta, theta_this)
                                    if distance_in_front <= 0:
                                        break
                                row_indexer2 -= 1

                                if front_seen:
                                    fr[i, :row_indexer2 - row_indexer1 + 1] = np.flip(
                                        places[i, row_indexer1:row_indexer2 + 1])

            else:
                for i in range(sz[0]):
                    if np.all(places[i, :] == 0):
                        continue

                    front_seen = False
                    crit_index = int(lane_count[i])
                    imp_object_in_back = theta2n_disp(obj_theta, car_info[places[i, crit_index - 1] - 1, 2])

                    if abs(imp_object_in_back) < theta_view:
                        for row_indexer5 in range(int(lane_count[i])):
                            theta_this = car_info[places[i, row_indexer5] - 1, 2]
                            distance_in_front = theta2n_disp(obj_theta, theta_this)
                            if distance_in_front > theta_view:
                                crit_index = row_indexer5
                                break

                    for row_indexer1 in range(crit_index):
                        theta_this = car_info[places[i, row_indexer1] - 1, 2]
                        distance_in_front = theta2n_disp(obj_theta, theta_this)

                        if abs(distance_in_front) < theta_view and distance_in_front > 0:
                            front_seen = True
                            for row_indexer2 in range(row_indexer1 + 1, crit_index):
                                theta_this = car_info[places[i, row_indexer2] - 1, 2]
                                distance_in_front = theta2n_disp(obj_theta, theta_this)
                                if distance_in_front <= 0:
                                    break
                            row_indexer2 -= 1
                            row_indexer5 += 1

                            if front_seen:
                                front = places[i, row_indexer1:row_indexer2 + 1]
                                crit_space = int(lane_count[i]) - crit_index
                                fr[i, :crit_space] = places[i, row_indexer5:lane_count[i]]
                                fr[i, crit_space:crit_space + row_indexer2 - row_indexer1 + 1] = np.flip(front)

        elif which == 4:
            if abs(theta_from_pi) > theta_view:
                for i in range(sz[0]):
                    if np.all(places[i, :] == 0):
                        continue

                    back_seen = False
                    for row_indexer1 in range(int(lane_count[i])):
                        theta_this = car_info[places[i, row_indexer1] - 1, 2]
                        distance_in_front = theta2n_disp(obj_theta, theta_this)

                        if -theta_view < distance_in_front < 0:
                            back_seen = True
                            for row_indexer2 in range(row_indexer1 + 1, int(lane_count[i])):
                                theta_this = car_info[places[i, row_indexer2] - 1, 2]
                                distance_in_front = theta2n_disp(obj_theta, theta_this)
                                if distance_in_front >= 0:
                                    break
                            row_indexer2 -= 1

                            if back_seen:
                                ba[i, :row_indexer2 - row_indexer1 + 1] = places[i, row_indexer1:row_indexer2 + 1]

        else: #which == 3
            if abs(theta_from_pi) > theta_view:
                for i in range(sz[0]):
                    if np.all(places[i, :] == 0):
                        continue

                    front_seen = back_seen = False
                    front_first = False

                    for row_indexer1 in range(int(lane_count[i])):
                        theta_this = car_info[places[i, row_indexer1] - 1, 2]
                        distance_in_front = theta2n_disp(obj_theta, theta_this)

                        if abs(distance_in_front) < theta_view and distance_in_front != 0:
                            if distance_in_front > 0:
                                front_seen = True
                                front_first = True
                                for row_indexer2 in range(row_indexer1 + 1, int(lane_count[i])):
                                    theta_this = car_info[places[i, row_indexer2] - 1, 2]
                                    distance_in_front = theta2n_disp(obj_theta, theta_this)
                                    if distance_in_front <= 0:
                                        break
                            else:
                                back_seen = True
                                for row_indexer2 in range(row_indexer1 + 1, int(lane_count[i])):
                                    theta_this = car_info[places[i, row_indexer2] - 1, 2]
                                    distance_in_front = theta2n_disp(obj_theta, theta_this)
                                    if distance_in_front >= 0:
                                        break
                            row_indexer2 -= 1

                            if front_seen + back_seen > 1:
                                piece1 = np.flip(places[i, row_indexer1:row_indexer2 + 1])
                                fr[i, :len(piece1)] = piece1 if front_first else ba[i, :len(piece1)]

    return fr, ba
