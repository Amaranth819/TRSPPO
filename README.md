# Sim-to-Real project on Unitree Go2

## Overview 

This repository is forked from [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways), which is a Go1 Sim-to-Real Locomotion Starter Kit. It seems that [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways) can be untilized on Unitree [A1](https://github.com/fan-ziqi/dog_rl_deploy) with simple modifications, since those robots are base on [unitree-legged-sdk](https://github.com/unitreerobotics/unitree_legged_sdk). 

However, the brand-new architecture [unitree-sdk2 ](https://github.com/unitreerobotics/unitree_sdk2)is not based on UDP anymore, so this project aims to train and deploy walk-these-ways on Unitree Go2 by modifying SDK interfaces.

## Requirements 
* miniconda
* pytorch 1.10 with cuda-11.3
* Isaac Gym
* Nvidia GPU with at least 8GB of VRAM

---
## Train and Play
Clone this repository and install:

``` bash
git clone https://github.com/Teddy-Liao/walk-these-ways-go2.git
cd walk-these-ways-go2
pip install -e .
```

Start training: 
```bash
python scripts/train.py
```

Select time-reversal PPO with:
```bash
python scripts/train.py --algorithm trsppo --headless
```
TRS-PPO adds actor consistency with weight 0.1; value consistency is disabled.
Runs are saved under `runs/gait-conditioned-agility/{robot}/{algorithm}/{date_time}/`,
with `robot=go2` by default and `algorithm` equal to `ppo` or `trsppo`.
The UTC timestamp combines date and time, for example `2026-10-22_14-30-12.123456`.
Select the training environment with `--robot go2`, `--robot x1`, or `--robot x1_go2bend`.
The default algorithm is the original PPO (`--algorithm ppo`). See
[time-reversal PPO training](docs/time_reversal_ppo.md) for options, implementation details, and tests.

View the latest PPO/TRS-PPO reward curves with `python scripts/plot_training.py`;
the plot is saved to `tmp/training_rewards.png`.

`train.sh` enables TensorBoard logging for both algorithms. Set up and run it with:

```bash
conda activate trsppo
python -m pip install -e '.[tensorboard]'
bash train.sh
```

In another terminal, run `conda activate trsppo` and `tensorboard --logdir runs`,
then open http://localhost:6006. Each run writes a `tensorboard/` directory with
reward components, PPO losses, timing, and (for TRS-PPO) time-reversal metrics.
Scalars use training iteration as the horizontal axis and the same summaries as
`metrics.pkl` (every 10 iterations and at the end of training).
For a direct launch, add `--tensorboard` to `scripts/train.py`; use
`--tensorboard-log-dir <directory>` to choose a different event directory.
`train.sh` defaults to `x1_go2bend`; select a robot with `--robot <name>` or `ROBOT=<name> bash train.sh`.
Plot a robot's latest runs with `python scripts/plot_training.py --robot <name>`.

Train MLP policies on X1 or the variant with rear legs bending in the same direction as the front:

```bash
bash train.sh --robot x1
bash train.sh --robot x1_go2bend
```

Each command runs PPO followed by TRS-PPO with TensorBoard logging. All three robots
share the Go2 training task: pronk, trot, pace, and bound, with a configured gait-frequency
range of 2–4 Hz, the same command ranges, curricula, rewards, termination thresholds,
terrain, and domain randomization. All use 70 observations per frame, including the full
15-command vector, and a 20-frame history (1400 policy inputs).
Both X1 environments use position PD control with `Kp=30`, `Kd=1.2`;
Go2 uses its actuator network. X1's front thigh/calf reference angles are `0.7/-1.5`
radians and rear angles are `-0.7/1.5`; `x1_go2bend` changes only the rear thigh/calf
references to `0.7/-1.5`. Both use the same imported X1 URDF, meshes, and joint limits.
The existing MLP actor, adaptation network, and privileged critic are retained.
Go2, X1, and `x1_go2bend` all use 20 history frames and default to 24 rollout steps
for both PPO and TRS-PPO. Override the rollout length with `--num-steps-per-env`.
Arguments such as `--num-envs 16 --iterations 2` are forwarded to both runs.
Earlier X1 checkpoints used a trot-only task and 58 observations per frame (1160 inputs
with 20 history frames, or 1740 with 30). They require their saved configuration for
inference. Start new X1 training runs for the shared 70-observation, four-gait task.

Evaluate saved policies on all four gaits and six forward/backward speeds:

```bash
bash evaluation_metrics.sh --dry-run
bash evaluation_metrics.sh
```

This selects all three robots and both algorithms, with 100 trials per gait/speed.
Use `--robot go2 --algorithm ppo` to select a subset. Results are saved under
`evaluations/`. See [evaluation instructions](docs/evaluation.md) for checkpoint
selection, metric definitions, gait commands, and evaluation conditions.

`go2_gym` and `go2_gym_learn` folders are the main folders for training process.

Play the model:
```bash
cd scripts
python play.py
```
![Alt text](media/go2_training.jpg)

Go2 pretrained model is provided in [./runs](runs/gait-conditioned-agility/pretrain-go2), you can choose whether to use provide pretrained model by modifying the label line `label = "gait-conditioned-agility/pretrain-go2/train"` to your own trained model.

### Known Issues
* `flip_visual_attachments` in [go2_config](go2_gym/envs/go2/go2_config.py) should be set to `True`, otherwise errors would occur when visualizing.
* To change configuration parameters of env or the robot, you should modify parameters in [go2_config](go2_gym/envs/go2/go2_config.py), not in [legged_robot_config](go2_gym/envs/base/legged_robot_config.py)


---
## Deploy on PC
Trained policy is only supported to be deployed through your PC or laptop now, because I am not familiar with Jetson Orin, and hope I can fix it and deploy on Jetson Orin.

### Requirements
#### Install LCM
Since [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways) implement an interface based on Lightweight Communications and Marshalling ([LCM](https://github.com/lcm-proj/lcm)) to pass sensor data, motor commands, and joystick state between their code and the low-level control SDK provided by Unitree, LCM should be installed firstly in your PC or laptlop.

Clone LCM repository to any location (where you usually place installed softwares), then install LCM:
```bash
git clone https://github.com/lcm-proj/lcm.git
mkdir build
cd build
cmake ..
make
sudo make install
```

#### Unitree_SDK2
unitree_sdk2 has been inclued in `go2_gym_deploy/unitree_sdk2_bin/library/unitree_sdk2`,  you can also clone from [Unitree Robotics](https://github.com/unitreerobotics/unitree_sdk2) to make sure the sdk is updated version.

```bash
cd go2_gym_deploy/unitree_sdk2_bin/library/unitree_sdk2
```
Delete build file
```bash
rm -r build
```
Install and build:
```bash
sudo ./install.sh
mkdir build
cd build
cmake ..
make
```

### Build lcm_position_go2
`go2_gym_deploy/unitree_sdk2_bin/lcm_position_go2.cpp` is the core file of this project, which is similar to `lcm_position.cpp` in [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways), but replace unitree_legged_sdk with unitree_sdk2.

Build lcm_position_go2 and generate runfile `lcm_position_go2`
```bash
cd go2_gym_deploy
rm -r build
mkdir build
cd build
cmake ..
make -j
```

All LCM messages files in `go2_gym_deploy/lcm_types` are set as the same format shown in [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways) to ensure successful connection with python files. LCM message files are provided in this project, and you can also generate customized LCM message files through the following instructions: 

`xxx_lcmt.hpp` files are generated by:
```bash
lcm-gen -x xxx.lcm
```

### Verify connection
Connect your PC/Laptop with Go2 robot with ethernet cable and check connection by:
```bash
ping 192.168.123.161
```

Check the network interface address, and copy the network interface address.
```bash
ifconfig
```
If error occurs, please check [Unitree Support](https://support.unitree.com/home/zh/developer/Quick_start) for details.

### Test communication between LCM and unitree_sdk2
You can verify LCM send by opening a new terminal:
```bash
cd go2_gym_deploy/build
sudo ./lcm_receive
```

If LCM and unitree_sdk2 are correctly connected with each other, messages will be shown in the terminal:

![Alt text](media/lcm_receive.png)

### Start LCM
Before starting LCM, ensure that lcm_receive has been properly shut down. 
**It's important not to run lcm_receive and lcm_position_go2 simultaneously.**

```bash
cd go2_gym_deploy/build
sudo ./lcm_position_go2 eth0
```
Replace `eth0` with your own network interface address. According to the messages shown in terminal, press `Enter` for several times and the communication between LCM and unitree_sdk2 will set up.

This command will automatically shut down Unitree sport_mode Service and set the robot to LOW-LEVEL. Please make sure This will Go2 is hung up or lie on the ground.


### Load and run policy
Open a new terminate and run:
```bash
cd go2_gym_deploy/scripts
python deploy_policy.py
```
According to the hints shown in terminal, Press button [R2] to start the controller. You can check RC mapping in the following subsection.


### Joystick Mapping

![Joystick Mapping](media/rc_map.png)


To view the details of joystick mapping or even modify default mapping logic, please refer to the `get_command` function within the [cheetah_state_estimator.py](go2_gym_deploy/utils/cheetah_state_estimator.py) file. In this project, the default gait is set to trot.


**Caution**:
* Press [L2+B] to switch to damping mode if any unexpected situation occurs!!!
* This is research code; use at your own risk; we do not take responsibility for any damage.

Test Video on Unitree Go2: 
- Test in my bedroom: https://www.bilibili.com/video/BV1tQ4y1c7ZG/?spm_id_from=333.999.0.0&vd_source=07873ebe2a113dac57775e264a210929
- Test by other contributors: https://www.bilibili.com/video/BV1Ut421H7Fr/?spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=07873ebe2a113dac57775e264a210929


---
## Deploy on Nvidia Jetson Orin

The Unitree Go2 robot is equipped with an onboard Nvidia Jetson Orin Nano/NX, which operates on an ARM-based architecture. Default information of this onboard computer is shown below, and you can connnect to Jetson by SSH, VScode(remote development) or plugging a HDMI cable.

```
IP:192.168.123.18
user name：unitree
password：123
```

### Requirements for Jetson
- cuda
- pytorch
- miniconda (Omitted here; please install it by yourself)
- cudnn (Omitted here; please install it by yourself)


Two different ways are provided to set up correct environments in Jetson: through Internet or through Docker.

### Through Internet
Connecting a Nvidia Jetson device to the internet can be done in two primary ways:

1. **Wired Connection**: Directly plug an Ethernet cable with internet access into the Jetson's Ethernet port. This method provides a stable and fast internet connection, suitable for tasks that require high bandwidth or low latency.

2. **Wireless Connection via USB Wi-Fi Adapter**: Purchase a USB Wi-Fi adapter compatible with the Jetson device. This method adds wireless connectivity, offering the flexibility to connect to the internet without the need for physical cables. However, it's important to ensure the USB Wi-Fi adapter is supported by the Jetson's operating system and drivers.



#### Check Jetpack Version
Jetpack toolbox has been preinstalled on Jetson, you should check the jetpack vertsion firstly.
```bash
sudo -H pip install jetson-stats  #Install jetson-stats toolkit
sudo jtop
```
According to the detail information printed in the terminal window, the Jetpack version of my Unitree Go2 is `Jetpack 5.1.1 [L4T 35.3.1]`

![](media/sudo_jtop.png)

You can also check libraries that have been preinstalled: 
```bash
sudo jetson_release
```
#### Install cuda for jetson
Check if there is a preinstalled version of cuda. 
```bash
nvcc -V # check preinstalled cuda version
```

If the preinstalled version if too high, you should uninstall it because, for instance, there is no Pytorch version that is compatible with cuda-12.2.

```bash
sudo apt-get remove cuda
sudo apt autoremove 
sudo apt-get remove cuda*
sudo dpkg -l |grep cuda # check if any residual cuda file exists
sudo dpkg -P Residual filename
```

Personally, I recommend to install cuda-11.8. Click the link, [CUDA Toolkit 11.8 Downloads](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=aarch64-jetson&Compilation=Native&Distribution=Ubuntu&target_version=20.04&target_type=deb_local) , to check installation commands.

#### Install Pytorch for Jetson

Please [download](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) pre-built PyTorch pip wheel installers for Jetson Nano, which is different from the way we install Pytorch on PC. Note that correct pytorch version should be chosen to make it compatible with specific version of cuda and Jetpack. 

#### Run codes without cable
As long as the environment and requirements on the Jetson are properly configured, you can follow the same deployment guidelines as you would on a PC. This liberates the robot! Now, you can test the code cable-free, offering more freedom to the robot's movements and applications.

### Through Docker

To be continue ...

--- 

🌟🌟🌟  **Please star this repository if it does help you! Many Thanks!** 🌟🌟🌟 


---
## Acknowledgements
* Many thanks to [Leolar](https://github.com/NihaoyaLeolar), who provide Nvidia 3060ti and supporting.
* Many thanks to [Jony](https://github.com/jonyzhang2023) and Peter for their support and encourage me to learn basic kownledge about RL.
* Many thanks to [Simonforyou](https://github.com/Simonforyou), who provide Go2 pretrained model.

---
## TO DO
- [x] Do not inherit config and env from go1_gym, build customized config and env files for Go2
- [x] Deploy on Jeston Orin Nano
- [ ] Deploy through Docker
