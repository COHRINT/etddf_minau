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
The associator first calculates the detection's global position using the relative range and bearing measurement transformed by ET-DDF's pose of the onboard vehicle. The logic

![alt text](https://github.com/COHRINT/etddf_minau/blob/master/AssociatorLogic.png?raw=true)

It rejects or associates this detection using ET-DDF's estimate of all agents. If the detection posititon does not fall within the 2-sigma uncertainty bounds of an agent, the detecion is discarded. 

- responsibilities
- where
- deeper into the logic
- interfaces
