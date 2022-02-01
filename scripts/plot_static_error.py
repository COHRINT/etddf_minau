from numpy.core.numeric import load
import rosbag
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

import matplotlib
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

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

def plot_error(data, events):

    def add_modem_times():
        plt.vlines(events["bluerov2_5"], -5, 5, colors="b", linestyles='dotted', label="Agent 0 Exchanging")
        plt.vlines(events["bluerov2_7"], -5, 5, colors="m", linestyles='dotted', label="Agent 1 Exchanging")

    # Should loop through the keys in data in each subplot
    colors = ["b", "r", "g"]
    i=0
    for key in data:
        scenario_data = data[key]
        color = colors[i]
        # BLUEROV2_7 X ERROR
        # plt.subplot(2,2,1)
        plt.figure()
        bluerov_data = scenario_data["bluerov2_7"]
        times = normalize_times( bluerov_data["times"] )
        rep = np.repeat(1.5, len(times))
        plt.plot(times, bluerov_data["x_err"], c="r", label="Estimate Error")
        plt.plot(times, rep, c="b", label="Ground Truth Box")
        plt.plot(times, -rep, c="b")

        two_sigma = bluerov_data["x_err_cov"]
        plt.plot(times, two_sigma, c="g", label="$2\sigma$ Uncertainty Bound")
        plt.plot(times, -np.array(two_sigma), c="g")
        plt.title("Agent 0 X Error")
        add_modem_times()
        plt.legend(prop={'size': 16}, loc="lower left", bbox_to_anchor=(0.75,0.75))
        plt.ylabel("x position [m]")
        plt.xlabel("time [s]")
        
        i += 1
    i=0
    for key in data:
        scenario_data = data[key]
        color = colors[i]
        # BLUEROV2_7 Y ERROR
        # plt.subplot(2,2,3)
        plt.figure()
        bluerov_data = scenario_data["bluerov2_7"]
        times = normalize_times( bluerov_data["times"] )
        plt.plot(times, bluerov_data["y_err"], c="r", label="Estimate Error")
        plt.plot(times, rep, c="b", label="Ground Truth Box")
        plt.plot(times, -rep, c="b")

        two_sigma = bluerov_data["y_err_cov"]
        plt.plot(times, two_sigma, c="g")
        plt.plot(times, -np.array(two_sigma), c="g", label="$2\sigma$ Uncertainty Bound")
        plt.title("Agent 0 Y Error")
        plt.ylabel("y position [m]")
        plt.xlabel("time [s]")
        add_modem_times()
        plt.legend(prop={'size': 16}, loc="lower left", bbox_to_anchor=(0.75,0.75))
        
        
        i += 1
    
    i=0
    for key in data:
        scenario_data = data[key]
        color = colors[i]
        # BLUEROV2_5 X ERROR
        # plt.subplot(2,2,2)
        plt.figure()
        bluerov_data = scenario_data["bluerov2_5"]
        times = normalize_times( bluerov_data["times"] )
        plt.plot(times, bluerov_data["x_err"], c="r", label="Estimate Error" )
        plt.plot(times, rep, c="b",label="Ground Truth Box")
        plt.plot(times, -rep, c="b")

        two_sigma = bluerov_data["x_err_cov"]
        plt.plot(times, two_sigma, c="g")
        plt.plot(times, -np.array(two_sigma), c="g", label="$2\sigma$ Uncertainty Bound")
        plt.title("Agent 1 X Error")
        plt.ylabel("x position [m]")
        plt.xlabel("time [s]")
        add_modem_times()
        plt.legend(prop={'size': 16}, loc="lower left", bbox_to_anchor=(0.75,0.75))
        
        
        i += 1

    i=0
    for key in data:
        scenario_data = data[key]
        color = colors[i]
    
        # BLUEROV2_5 Y ERROR
        # plt.subplot(2,2,4)
        plt.figure()
        bluerov_data = scenario_data["bluerov2_5"]
        times = normalize_times( bluerov_data["times"] )
        plt.plot(times, bluerov_data["y_err"], c="r", label="Estimate Error" )
        plt.plot(times, rep, c="b", label="Ground Truth Box")
        plt.plot(times, -rep, c="b")

        two_sigma = bluerov_data["y_err_cov"]
        plt.plot(times, two_sigma, c="g")
        plt.plot(times, -np.array(two_sigma), c="g", label="$2\sigma$ Uncertainty Bound")
        plt.ylabel("y position [m]")
        plt.xlabel("time [s]")
        add_modem_times()
        plt.legend(prop={'size': 16}, loc="lower left", bbox_to_anchor=(0.75,0.75))
        plt.title("Agent 1 Y Error")
        
        
        i += 1

    # plt.subplot(2,2,1)
    # plt.vlines(events["bluerov2_5"], -5, 5, colors="b", linestyles='dotted')
    # plt.vlines(events["bluerov2_7"], -5, 5, colors="m", linestyles='dotted')
    # plt.subplot(2,2,2)
    # plt.vlines(events["bluerov2_5"], -5, 5, colors="b", linestyles='dotted', label="Agent 0 Exchanging")
    # plt.vlines(events["bluerov2_7"], -5, 5, colors="m", linestyles='dotted', label="Agent 1 Exchanging")
    # plt.legend()
    # plt.subplot(2,2,3)
    # plt.vlines(events["bluerov2_5"], -5, 5, colors="b", linestyles='dotted')
    # plt.vlines(events["bluerov2_7"], -5, 5, colors="m", linestyles='dotted')
    # plt.subplot(2,2,4)
    # plt.vlines(events["bluerov2_5"], -5, 5, colors="b", linestyles='dotted')
    # plt.vlines(events["bluerov2_7"], -5, 5, colors="m", linestyles='dotted')

    plt.show()

### LOAD BAG ###
def get_events(bag):
    bag = rosbag.Bag(bag)
    # Just want an array of times for each type of event
    event_times = {
        "bluerov2_5": [],
        "bluerov2_7" : []
    }

    # Just grab the first time so we can plot wrt the other data
    for topic, msg, t in bag.read_messages():
        event_times["bluerov2_5"].append(t)
        event_times["bluerov2_7"].append(t)
        break

    
    for topic, msg, t in bag.read_messages(topics=["/etddf/event/bluerov2_5", "/etddf/event/bluerov2_7"]):
        if "bluerov2_5" in topic:
            event_times["bluerov2_5"].append(t)
        else:
            event_times["bluerov2_7"].append(t)
    
    event_times["bluerov2_5"] = normalize_times(event_times["bluerov2_5"])
    event_times["bluerov2_7"] = normalize_times(event_times["bluerov2_7"])
    event_times["bluerov2_5"].pop(0)
    event_times["bluerov2_7"].pop(0)
    print(len(event_times["bluerov2_5"]))
    print(len(event_times["bluerov2_7"]))


    last = None
    if len(event_times["bluerov2_5"]) % 2 != 0:
        last = event_times["bluerov2_5"].pop(-1)

    arr = np.array(event_times["bluerov2_5"])
    new_arr = np.reshape(arr, (-1, 2))
    event_times["bluerov2_7"] = new_arr[:,1]
    event_times["bluerov2_5"] = new_arr[:,0]
    if last is not None:
        event_times["bluerov2_5"] = np.append(event_times["bluerov2_5"], last)
    return event_times


def load_bag(bag):
    bag = rosbag.Bag(bag)

    ground_truth = {
        "bluerov2_5" : {"x" : 4, "y" : 1},
        "bluerov2_7" : {"x" : 4, "y" : -1}
    }

    plotting_agent = "bluerov2_5"
    data = {
        "bluerov2_5": {
            "x_err" : [],
            "x_err_cov" : [],
            "y_err" : [],
            "y_err_cov" : [],
            "times" : []
        },
        "bluerov2_7": {
            "x_err" : [],
            "x_err_cov" : [],
            "y_err" : [],
            "y_err_cov" : [],
            "times" : []
        }
    }

    for topic, msg, t in bag.read_messages():
        agent = topic.split("/")[1]
        if agent != plotting_agent:
            continue
        if "network" in topic:
            for a in msg.assets:
                if "red" in a.name:
                    continue
                unc = np.reshape(a.odom.pose.covariance, (6,6))
                data[a.name]["x_err"].append( ground_truth[a.name]["x"] - a.odom.pose.pose.position.x )
                data[a.name]["x_err_cov"].append( 2*np.sqrt(unc[0,0]) )
                data[a.name]["y_err"].append( ground_truth[a.name]["y"] - a.odom.pose.pose.position.y )
                data[a.name]["y_err_cov"].append( 2*np.sqrt(unc[1,1]) )
                data[a.name]["times"].append( t )

    return data # these should be the same length!

data = {}
# data["DeltaTier"] = load_bag("deltatier_static.bag")
# data["DeltaTier"] = load_bag("latest_deltatier.bag")
# data["DeltaTier"] = load_bag("2021-12-15-10-23-14.bag")
bag = "2021-12-15-11-20-29.bag"
bag = "2021-12-15-12-02-40.bag"
bag = "2021-12-15-12-07-42.bag"
bag = "2021-12-15-12-28-36.bag"
bag = "2021-12-15-14-58-32.bag"
bag = "2021-12-15-15-13-14.bag"
bag = "2021-12-15-16-05-21.bag"
bag = "2021-12-15-16-28-22.bag"
bag = "2021-12-15-16-47-34.bag"
# bag = "split_bag.bag"
bag = "2021-12-15-17-05-32.bag"
# bag = "2021-12-15-17-22-21.bag"
bag = "2021-12-15-17-41-11.bag"
bag = "2021-12-15-18-17-14.bag"
bag = "2021-12-15-18-34-02.bag"

bag = "final/final_multi_red.bag"
# bag = "final/final_no_red.bag"
data["DeltaTier"] = load_bag(bag)
events = get_events(bag)
# data["No Collaboration"] = load_bag(no_collab_bag)
# omni_bag = "omniscient_bag.bag"
# data["Omniscient Old"] = load_bag(omni_bag)
# omni_tuned_bag = "2021-12-13-11-36-47.bag"
# data["Omniscient"] = load_bag(omni_tuned_bag)

plot_error(data, events)