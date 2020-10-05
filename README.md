# MEC_RP
1. The uploaded document including the reading notes of the EdgeCloudSim study. The original reference paper link is: https://ieeexplore.ieee.org/document/7946405

2. Youtube Channel for studying how to operate the EdgeCloudSim code in Eclipse:https://www.youtube.com/channel/UC2gnXTWHHN6h4bk1D5gpcIA/

3. Github link for the original code source: https://github.com/CagataySonmez/EdgeCloudSim

4. Some more helpful document provided by the author: https://github.com/CagataySonmez/EdgeCloudSim/wiki

Currently EdgeCloudSim provided necessary functionality in terms of computation and networking abilities. 
But for my project, the task migration among the Edge or Cloud VMs and energy consumption model for the mobile and edge devices would need to be considered.

5. Use configuration files to manage the parameters. EdgeCloudSim reads parameters dynamically from the following files:
     
     MainApp.java
   
    •	config.properties: Simulation settings are managed in configuration file

    •	applications.xml: Application properties are stored in xml file

    •	edge_devices.xml: Edge devices (datacenters, hosts, VMs etc.) are defined in xml file

6. Some helpful link regarding future work: 

   Q1: Use the matlab file to plot graphics: https://github.com/CagataySonmez/EdgeCloudSim/issues/29
   
   Q2: Build an energy model in EdgeCloudSim: https://github.com/CagataySonmez/EdgeCloudSim/issues/12
   
7. Helpful Report: https://ieeexplore.ieee.org/document/7962674
   
