import numpy as np
from numpy.linalg import inv
from copy import deepcopy

"""
Simple associator node:
Assumptions
- begin proto-tracks with unasscoiated measurements
- approximate chi-square test with 2x 1D tests
- proto-tracks are static
- treat each measurement as a gps and do not account for uncertainty tied to the ownship state
"""

class Associator:

    def __init__(self, time_to_drop, lost_agent_unc, proto_track_points, proto_Q):
        """
        lost_agent_unc : float
            uncertainty size by which we should use a prototrack algorithm to associate first
        Q : np.array (2,2)
            uncertainty (per second) to add to prototracks
        """
        self.proto_tracks = {} # { "proto_X" : [mean, cov, num_points, last_meas_time] }
        self.time_to_drop = time_to_drop
        self.lost_agent_unc = lost_agent_unc
        self.proto_track_points = proto_track_points
        self.proto_Q = proto_Q
        self.last_time = None
        self.proto_track_naming_num = 0

    def _get_distances(self, agent_dict, meas, ignore_lost=True):
        """
        Find the association distances for the measurement
        """
        search_agents = []
        agents = []
        vals = []
        for agent in agent_dict:
            mean = agent_dict[agent][0]
            cov = agent_dict[agent][1]

            # Approximate as 2x 1D distributions
            x_std = np.sqrt(cov[0,0])
            y_std = np.sqrt(cov[1,1])

            if ignore_lost and (x_std > self.lost_agent_unc or y_std > self.lost_agent_unc):
                search_agents.append(agent)
                continue

            val_x = (meas[0,0] - mean[0,0]) / x_std
            val_y = (meas[1,0] - mean[1,0]) / y_std

            vals.append(np.linalg.norm([val_x, val_y]))
            agents.append(agent)
        return agents, vals, search_agents

    def associate(self, agent_dict, meas, R, t, association_sigma=2):
        """
        agent_dict : {[mean], [cov]
        meas : np.array (2,1)
            linearized position of the measurement (gps-like)
        R : np.array (2,2)
            uncertainty of the measurement
        t : time
            arbitrary time units, just needs to be comparable to self.time_to_drop
        """
        self.clean_protos(t)
        self.last_time = t

        # Loop through position_dict and use simple 1D approximation to get association values
        new_agent_dict = agent_dict.copy()

        agents, vals, search_agents = self._get_distances(new_agent_dict, meas, True)

        if vals:
            min_index = np.argmin(vals)
            agent_name = agents[min_index]
            # Attempt to associate with an agent
            if vals[min_index] < np.linalg.norm([association_sigma,association_sigma]): # is better than 2 sigma in each direction
                return agent_name, False

        if len(search_agents) > 0:
            # Can't associate with an agent --> Try to associate with a prototrack
            if len(self.proto_tracks) > 0:
                agents, vals, _ = self._get_distances(self.proto_tracks, meas, False)
                min_index = np.argmin(vals)
                agent_name = agents[min_index]
                if vals[min_index] < np.linalg.norm([association_sigma,association_sigma]):
                    self.proto_tracks[agent_name][2] += 1
                    self.proto_tracks[agent_name][4].append([meas, t, vals[min_index]])
                    print("{} ({}/{}) meas associated".format(agent_name, self.proto_tracks[agent_name][2], self.proto_track_points))

                    x = self.proto_tracks[agent_name][0]
                    P = self.proto_tracks[agent_name][1]
                    # Filter the measurement
                    H = np.eye(2)
                    K = P @ H.T @ inv(H @ P @ H.T + R)
                    x = x + K @ (meas - H @ x)
                    P = P - K @ H @ P
                    self.proto_tracks[agent_name][0] = x
                    self.proto_tracks[agent_name][1] = P

                    if self.proto_tracks[agent_name][2] >= self.proto_track_points: # We need to associate with an agent, can only do so if there is 1 unknown

                        if len(search_agents) > 1:
                            print("Cannot associate due to multiple agents being lost: " + str(search_agents))
                            return "proto", False
                        else:
                            # Associate this prototrack with the only lost agent!
                            associated_agent = search_agents[0]
                            print("Associating prototrack {} with agent {}".format(agent_name, associated_agent))

                            # Remove the prototrack
                            del self.proto_tracks[agent_name]
                            return associated_agent, True
                    else:
                        return "proto", False
            
            # Can't associate with anything? --> Start a prototrack
            name = "proto_track_" + str(self.proto_track_naming_num)
            print("Starting " + name)
            self.proto_tracks[name] = [meas, R, 1, t, [[meas, t, 0]]]
            self.proto_track_naming_num += 1
            return "proto", False
        else:
            return "none", False

    def clean_protos(self, t):
        if self.last_time == None:
            return
        else:
            delta = t - self.last_time

        keys = list(self.proto_tracks.keys())
        for proto in keys:
            # propogate prototrack estimates
            self.proto_tracks[proto][1] += self.proto_Q * t

            # Drop old measurements
            last_meas_time = self.proto_tracks[proto][3]
            if t - last_meas_time > self.time_to_drop:
                print(proto + " expiring")
                del self.proto_tracks[proto]

    def get_proto(self):
        if len(self.proto_tracks) > 0:
            proto_tracks = deepcopy(self.proto_tracks)
            _, proto = proto_tracks.popitem()
            return proto
        else:
            return None

if __name__ == "__main__":
    position_dict = {
        "agent_1" : [np.array([0,0]), np.eye(2)],
        "agent_2" : [np.array([4,4]), 2*np.eye(2)],
        "agent_3" : [np.array([0,0]), 1000*np.eye(2)]
    }
    meas = np.array([1,1])
    R = np.eye(2)
    t = 0
    A = Associator(5)
    agent = A.associate(position_dict, meas, R, t, control=False)

