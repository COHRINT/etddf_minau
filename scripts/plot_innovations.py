"""
We want to open the bag and plot the innovations at the time they happened with the latest etddf estimate
- want to make sure the etddf estimate didn't happen at the same exact time
- we have 2 agents, each with 2 estimates. That's 4 kinds of innovations.

We also have range and azimuth! Maybe just for a single agent...
We'll do 2x2 subplot, plot the innovations at the correct time along with the measurement covariance
"""
import rosbag
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

# Agent 0 --> bluerov2_7
# Agent 1 --> bluerov2_5

def normalize_times(times):
    # Normalize times
    t1 = times[0].secs
    times_new = []
    for t in times:
        tsec = t.secs - t1
        times_new.append(tsec)
    return times_new

def plot_innovations(data):

    # Should loop through the keys in data in each subplot
    colors = ["b", "r", "g"]
    i=0
    for key in data:
        scenario_data = data[key]
        color = colors[i]

        plt.suptitle("Modem Measurement Innovations")

        # BLUEROV2_7 RANGE
        plt.subplot(2,2,1)
        bluerov_data = scenario_data["bluerov2_7"]
        innovations = np.abs( bluerov_data["range_inn"] )
        innovation_covs = 2*np.sqrt(bluerov_data["range_inn_cov"])
        times = normalize_times( bluerov_data["range_inn_t"] )
        plt.plot(times, innovations, c=color, label="{}".format(key))
        plt.plot(times, innovation_covs ,"--", c=color)#, label="{} Inn. Covariance ($2\sigma$)".format(key))
        plt.title("Agent 0")
        plt.legend()
        plt.ylabel("Range (Abs value)")
            

        # BLUEROV2_7 AZIMUTH
        plt.subplot(2,2,3)
        bluerov_data = scenario_data["bluerov2_7"]
        innovations = np.abs( bluerov_data["azimuth_inn"] )
        innovation_covs = 2*np.sqrt(bluerov_data["azimuth_inn_cov"])
        times = normalize_times( bluerov_data["range_inn_t"] )
        plt.plot(times, innovations, c=color)
        plt.plot(times, innovation_covs ,"--", c=color)
        # plt.title("Modem Azimuth of Agent 0")
        plt.ylabel("Azimuth (Abs value)")

        # BLUEROV2_5 RANGE
        plt.subplot(2,2,2)
        bluerov_data = scenario_data["bluerov2_5"]
        innovations = np.abs( bluerov_data["range_inn"] )
        innovation_covs = 2*np.sqrt(bluerov_data["range_inn_cov"])
        times = normalize_times( bluerov_data["range_inn_t"] )
        if i == 0:
            plt.plot(times, innovations, c=color, label="Innovation")
            plt.plot(times, innovation_covs ,"--", c=color, label="$2\sigma$ Innovation Unc.")
            plt.legend()
        else:
            plt.plot(times, innovations, c=color)
            plt.plot(times, innovation_covs ,"--", c=color)
        plt.title("Agent 1")

        # # BLUEROV2_5 AZIMUTH
        plt.subplot(2,2,4)
        bluerov_data = scenario_data["bluerov2_5"]
        innovations = np.abs( bluerov_data["azimuth_inn"] )
        innovation_covs = 2*np.sqrt(bluerov_data["azimuth_inn_cov"])
        times = normalize_times( bluerov_data["range_inn_t"] )
        plt.plot(times, innovations, c=color)
        plt.plot(times, innovation_covs ,"--", c=color)
        # plt.title("Modem Azimuth of Agent 1")

        i += 1

    plt.show()

def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2*np.pi
    while angle < -np.pi:
        angle += 2*np.pi
    return angle

def get_range_innovation(meas_value, estimate):
    # Returns innovation, innovation variance
    position = [0, -5, -1]
    x1 = estimate.pose.position.x
    y1 = estimate.pose.position.y
    z1 = estimate.pose.position.z
    x2 = position[0]
    y2 = position[1]
    z2 = position[2]
    P = np.reshape( np.array([estimate.covariance]), (6,6) )
    P = P[:3,:3] # Just the position cov

    delta_pred = np.array([x1 - x2,y1 - y2, z1 - z2])
    if norm(delta_pred) < 1e-3:
        delta_pred[0,0] = 1e-3
        delta_pred[1,0] = 1e-3
        delta_pred[2,0] = 1e-3
    pred = norm(delta_pred)

    drdx1 = delta_pred[0] / norm(delta_pred)
    drdy1 = delta_pred[1] / norm(delta_pred)
    drdz1 = delta_pred[2] / norm(delta_pred)
    
    H = np.zeros((1, 3))
    H[0, 0] = drdx1
    H[0, 1] = drdy1
    H[0, 2] = drdz1

    innovation = meas_value - pred
    R = 0.04 # force_modem_range_var

    innovation_cov = np.dot(H, np.dot(P, H.T)) + R
    return innovation, innovation_cov[0,0]

def get_azimuth_innovation(meas_value, estimate):

    meas_value = np.radians(meas_value) # convert deg to rads

    position = [0, -5, -1]
    x1 = estimate.pose.position.x
    y1 = estimate.pose.position.y
    x2 = position[0]
    y2 = position[1]
    P = np.reshape( np.array([estimate.covariance]), (6,6) )
    P = P[:3,:3] # Just the position cov

    delta_pred = np.array([[x1 - x2],[y1 - y2]])
    if norm(delta_pred) < 1e-3:
        delta_pred[0,0] = 1e-3
        delta_pred[1,0] = 1e-3
    pred = np.arctan2(delta_pred[1,0], delta_pred[0,0])

    dadx = -delta_pred[1,0] / norm(delta_pred)**2
    dady = delta_pred[0,0] / norm(delta_pred)**2
    H = np.zeros((1, 3))
    H[0, 0] = dadx
    H[0, 1] = dady
    
    innovation = normalize_angle( meas_value - pred )

    R = 0.05 # force_modem_az_var

    innovation_cov = np.dot(H, np.dot(P, H.T)) + R
    return innovation, innovation_cov[0,0]


### LOAD BAG ###


def load_bag(bag):
    bag = rosbag.Bag(bag)

    plotting_agent = "bluerov2_5"
    data = {
        "bluerov2_5": {
            "range_inn" : [],
            "range_inn_cov" : [],
            "azimuth_inn" : [],
            "azimuth_inn_cov" : [],
            "range_inn_t" : [],
            "range_inn_cov_t" : [],
            "azimuth_inn_t" : [],
            "azimuth_inn_cov_t" : []
        },
        "bluerov2_7": {
            "range_inn" : [],
            "range_inn_cov" : [],
            "azimuth_inn" : [],
            "azimuth_inn_cov" : [],
            "range_inn_t" : [],
            "range_inn_cov_t" : [],
            "azimuth_inn_t" : [],
            "azimuth_inn_cov_t" : []
        }
    }

    # Sync estimates with measurements to get innovations
    last_estimates = {"bluerov2_5" : None, "bluerov2_7":None}
    last_estimate_t = None
    last_meas = None
    last_meas_t = None
    for topic, msg, t in bag.read_messages():
        agent = topic.split("/")[1]
        if agent != plotting_agent:
            continue
        if "network" in topic:
            for a in msg.assets:
                last_estimates[a.name] = a.odom.pose
            last_estimate_t = t
        elif "packages_in" in topic:
            last_meas = msg
            last_meas_t = t

        # Save to right place
        if last_estimate_t is not None and last_meas is not None: 

            # loop through the packet
            for m in msg.measurements:
                measured_agent = m.measured_asset
                if "range" in m.meas_type:
                    inn, inn_cov = get_range_innovation(m.data, last_estimates[measured_agent])
                    data[measured_agent]["range_inn"].append(inn)
                    data[measured_agent]["range_inn_cov"].append(inn_cov)
                    data[measured_agent]["range_inn_t"].append(last_meas_t)
                    data[measured_agent]["range_inn_cov_t"].append(last_meas_t)
                else: # azimuth
                    inn, inn_cov = get_azimuth_innovation(m.data, last_estimates[measured_agent])
                    data[measured_agent]["azimuth_inn"].append(inn)
                    data[measured_agent]["azimuth_inn_cov"].append(inn_cov)
                    data[measured_agent]["azimuth_inn_t"].append(last_meas_t)
                    data[measured_agent]["azimuth_inn_cov_t"].append(last_meas_t)

            last_estimates = {"bluerov2_5" : None, "bluerov2_7":None}
            last_estimate_t = None
            last_meas = None
            last_meas_t = None

    return data # these should be the same length!

data = {}
no_collab_bag = "no_collaboration.bag"
data["No Collaboration"] = load_bag(no_collab_bag)
omni_bag = "omniscient_bag.bag"
# data["Omniscient Old"] = load_bag(omni_bag)
omni_tuned_bag = "2021-12-13-11-36-47.bag"
data["Omniscient"] = load_bag(omni_tuned_bag)

plot_innovations(data)