"""
We want to plot the deltatiers selected over time for each agent
We also want to generate a chart showing the number of measurements shared
The key is to get Nisar's feedback on this metric
"""
from numpy.core.numeric import load
import rosbag
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib

# Agent 0 --> bluerov2_7
# Agent 1 --> bluerov2_5

import matplotlib
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

def normalize_times(times, ref_time):
    # Normalize times
    t1 = ref_time.secs
    times_new = []
    for t in times:
        tsec = t.secs - t1
        times_new.append(tsec)
    return times_new

### LOAD BAG ###

def load_bag(bag):
    bag = rosbag.Bag(bag)

    mults_7 = []
    times_7 = []
    mults_5 = []
    times_5 = []

    ref_time = None
    for topic, msg, t in bag.read_messages():
        ref_time = t
        break

    for topic, msg, t in bag.read_messages(topics=["/bluerov2_5/etddf/delta_multiplier", "/bluerov2_7/etddf/delta_multiplier"]):
        if "5" in topic:
            mults_5.append(msg.data)
            times_5.append(t)
        else:
            mults_7.append(msg.data)
            times_7.append(t)
    times_5 = normalize_times(times_5, ref_time)
    times_7 = normalize_times(times_7, ref_time)


    plt.scatter(times_5, mults_5, c="b", label="Agent 1") # 1 is 5
    plt.scatter(times_7, mults_7, c="g", label="Agent 0") # 0 is 7
    plt.ylim([0,11])
    plt.title("Delta-Bands Selected vs Time")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Delta-band")
    plt.show()
    
bag = "2021-12-15-14-58-32.bag"
bag = "2021-12-15-15-13-14.bag"
bag = "2021-12-15-16-05-21.bag"
# bag = "2021-12-15-16-28-22.bag"
bag = "split_bag.bag"
bag = "2021-12-15-17-05-32.bag"
bag = "2021-12-15-17-41-11.bag"
bag = "final/final_multi_red.bag"
bag = "final/final_no_red.bag"
load_bag(bag)

## TOTAL MEASUREMENTS
## RED AGENTS
if bag == "final/final_multi_red.bag":
    blue5_explicit = 111
    blue5_implicit = 255
    blue5_total = 952

    blue7_explicit = 107
    blue7_implicit = 265
    blue7_total = 540

# Just blue agents
if bag == "final/final_no_red.bag":
    blue5_explicit = 90
    blue5_implicit = 46
    blue5_total = 160

    blue7_explicit = 90
    blue7_implicit = 52
    blue7_total = 206

omni_total = blue5_total + blue7_total
implicit_total = blue5_implicit + blue7_implicit
explicit_total = blue5_explicit + blue5_explicit
width = 0.35
bottom = [omni_total, explicit_total]

fig, ax = plt.subplots()
labels = ["Centralized", "DeltaTier"]
ax.bar(labels, bottom, width, label='Explicit') # Explicit
ax.bar(labels, [0, implicit_total], width, bottom=bottom, label='Implicit')
ax.set_ylabel('Measurements Shared')
ax.set_title('Measurements Shared by Algorithm')
ax.legend()
plt.show()
