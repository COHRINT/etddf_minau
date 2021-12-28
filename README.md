# etddf_minau

## Setup
```
git submodule init
git submodule update
```

## ET-DDF Node
TBD

## Sonar Control Node & Associator
The sonar control node is responsible for modifying the ping360 sonar scan for the purposes of tracking or searching for an agent. The associator node is responsible for associating a detection with an agent. 

#### Relevant files
* src/deltatier/sonar_controller.py
* src/deltatier/associator.py
* scripts/associator_node.py


While both the sonar control node and associator have their own source files, they get loaded into ROS through the same node, associator_node.py. This design decision was made because of the coupled logic of these two components.

### Associator Logic
The associator first calculates the detection's global position using the relative range and bearing measurement transformed by ET-DDF's pose of the onboard vehicle. The logic of the associator node is shown below:

![alt text](https://github.com/COHRINT/etddf_minau/blob/master/AssociatorLogic.png?raw=true)

The associator labels an agent as "LOST" if the ET-DDF uncertainty of that agent is greater than the "lost_agent_unc". If at least one agent is lost the associator will first create a "proto-track". The proto-track serves as a filtering technique to remove false detections, requiring multiple detections near the same point to be considered eligible for association with a LOST agent. If more than 1 agent is in the lost state, the associator cannot determine which LOST agent the proto-track belongs to, so it waits for further information (in the case of 1 lost blue agent and )

- responsibilities
- where
- deeper into the logic
- interfaces
