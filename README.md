# etddf_minau

This repo contains CU Boulder's filtering solution to the MinAu project. The et-ddf node is the primary filtering solution node but included in this repo are also an association node and a sonar control node. The function of all of these nodes are described below.

## Setup
```
git submodule init
git submodule update
```

## ET-DDF Node
This is the primary filtering solution node. It is this node's job to provide estimates (means and covariances) of all tracked quantities. The name is somewhat of a misnomer because the filtering solution doesn't actually do "DDF" (Distributed Data Fusion), just "ET" (Event-Triggering). 

### Functionality
This node's theoretical functionality (the DeltaTier algorithm) is described in [this paper](https://drive.google.com/file/d/1nSslfkytGxTQvBNfssFxo53TezaEpmbx/view?usp=sharing) submitted to ACC 2022. 

### Relevant Files
* scripts/etddf_node_new.py : ROS interface for etddf
* src/deltatier/kf_filter.py : DeltaTier implementation source file

### Interfaces
The et-ddf node exposes two types of topics to gain access to measurements.
* "etddf/estimate/network" contains all estimates (nav_msgs/Odometry.msg) in a single topic.
* "etddf/estimate/$agent" is the estimate topic for a specific agent.

### Inputs
THe DeltaTier filtering solution receives ownship information (measurements of the onboard vehicle) through the robot_localization (RL) package. This can be seen in the following diagram.
![alt text](https://github.com/COHRINT/etddf_minau/blob/master/MultiFilterArchitecture.png?raw=true)

The "Nav Filter" is RL. PSCI stands for Partial-State Covariance Intersection [1] which is an algorithm that allows combination of two estimates from two different nodes using the overlap of common states between the estimates. For Shared Information - see the buffer sharing described described by the paper referenced in the Functionality section. 

## Sonar Control Node & Associator
The sonar control node is responsible for modifying the ping360 sonar scan for the purposes of tracking or searching for an agent. The associator node is responsible for associating a detection with an agent. 

#### Relevant files
* src/deltatier/sonar_controller.py
* src/deltatier/associator.py
* scripts/associator_node.py


While both the sonar control node and associator have their own source files, they get loaded into ROS through the same node, associator_node.py. This design decision was made because of the coupled logic of these two components.

### Associator Logic
The associator first calculates the detection's global position using the relative range and bearing measurement transformed by ET-DDF's pose of the onboard vehicle. The logic of the associator node is shown below:

![alt text](https://github.com/COHRINT/etddf_minau/blob/master/AssociatorLogicFlow.png?raw=true)

#### Definitions
The 2sigma uncertainty of an agent is based off of the ET-DDF position covariance of the agent. The associator labels an agent as "LOST" if the ET-DDF uncertainty of that agent is greater than the "lost_agent_unc". If a detection cannot be associated with a non-LOST agent then it is a candidate to be associated with the LOST agent. This happens by first starting a track based on that detection, a "proto-track", and associating further detections with the proto-track. After a certain number of detections have been associated, the proto-track is associated with the lost agent. The proto-track process serves as a filtering technique to remove false detections, requiring multiple detections near the same point to be considered eligible for association with a LOST agent. Note that if >1 agent is LOST, proto-tracks cannot be associated. This is because the associator cannot determine which LOST agent the proto-track belongs to, so it waits for one agent to exit the LOST state.

Proto-tracks are discarded after a certain period of time (param: time_to_drop).

#### Publications
* Topic: "sonar_processing/target_list/associated". Data-type:  "minau/SonarTargetList.msg"

### Sonar Control Node
If sonar tracking is enabled (enable_sonar_control) first we check if a non-LOST agent's uncertainty is greater than a threshold (ping_thresh). If this is the case we attempt to hone in uncertainty by scanning the agent. If all non-LOST agents are below the ping_thresh and we have at least one LOST agent, we perform a search by scanning 360 degrees around the agent. If If all non-LOST agents are below the ping_thresh and no agents are lost, we ping the agent with the largest uncertainty. 

#### Definitions
"Ping" refers to commanding the ping360 to perform a scan over a range of radians. 

#### Publications
* Topic: "ping360_node/sonar/set_scan". Data-type: "ping360_sonar/SonarSettings.msg"

## Citations
1. N. R. Ahmed, W. W. Whitacre, S. Moon, and E. W. Frew, “Factorized
covariance intersection for scalable partial state decentralized data
fusion,” in 2016 19th International Conference on Information Fusion
(FUSION). IEEE, 2016, pp. 1049–1056.