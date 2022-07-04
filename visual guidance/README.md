# MPC-DCOACH
Repository for MPC + D-COACH paper code

## Run MPC-DCOACH:
1. start carla server (alias carla): bash /home/rodrigo/Documents/CARLA_0.9.10.1/CarlaUE4.sh -benchmark -fps=20 -quality-level=Low -windowed -ResX=500 -ResY=500 -opengl4
2. Load Carla: load_carla
3. Lunch lmpcc: roslaunch lmpcc carla_lmpcc.launch
4. Lunch teleop twist keyboard: rosrun teleop_twist_keyboard teleop_twist_keyboard.py or run joystick: rosrun joy joy_node
5. Run DCOACH from src in MPC-DOACH repo: python main_actor.py
