# Visual Guidance
Visual guidance implementation

## Usage
1. start carla server (alias carla): bash <carla directory>/CARLA_0.9.10.1/CarlaUE4.sh -benchmark -fps=20 -quality-level=Low -windowed -ResX=500 -ResY=500 -opengl4
2. Load Carla: export PYTHONPATH=$PYTHONPATH:<carla directory>/CARLA_0.9.10.1/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg:/home/rodrigo/Documents/CARLA_0.9.10.1/PythonAPI/carla
3. Lunch lmpcc: roslaunch lmpcc carla_lmpcc.launch
4. Lunch teleop twist keyboard: rosrun teleop_twist_keyboard teleop_twist_keyboard.py or run joystick: rosrun joy joy_node
5. Run iDAgger: python main.py
