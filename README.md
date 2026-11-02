# PPO vs SAC on MuJoCo Reacher-v5 🤖

Goal of this project is comparing two reinforcement-learning approaches 
**PPO (on-policy)** and **SAC (off-policy)** with [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) in [Gymnasium's Reacher-v5](https://gymnasium.farama.org/environments/mujoco/reacher/) 
to controlling a two-jointed robotic arm. Both models were trained for 200,000 environment steps
and evaluated using the same initial conditions.
The performance of trained models for 3 different target positions, [0,1,2], has been shown in below.
 
 |                     SAC_Target_0                     |                     SAC_Target_1                     |                     SAC_Target_2                     |
 |:----------------------------------------------------:|:----------------------------------------------------:|:----------------------------------------------------:|
 | <img src="videos/sac/sac_episode_0.gif" width="400"> | <img src="videos/sac/sac_episode_1.gif" width="400"> | <img src="videos/sac/sac_episode_2.gif" width="400"> |

|                     PPO_Target_0                     |                     PPO_Target_1                     |                     PPO_Target_2                     |
|:----------------------------------------------------:|:----------------------------------------------------:|:----------------------------------------------------:|
| <img src="videos/ppo/ppo_episode_0.gif" width="400"> | <img src="videos/ppo/ppo_episode_1.gif" width="400"> | <img src="videos/ppo/ppo_episode_2.gif" width="400"> |



## 1. Introduction and Goals

Reacher-v5 models a **two-joint arm**. Each policy observes a
**10-element observation** describing the arm's state and target
geometry, and outputs **2 bounded joint-torque commands** in `[-1, 1]`.
Based on the same number of **environment interactions**, how do PPO and SAC 
compare at reaching a target in Reacher-v5?

## 2. Set up Requirments

Recommended: **Python 3.10–3.12**, an up-to-date operating system, ~8 GB+ RAM. For the first run, a CPU is sufficient. MuJoCo installs through Gymnasium's extra.

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. The robotics / RL problem
Markove decision process consist of (State, Action, Transition function, Reward) that has been described in below.

### 3.1. Observation Space (State)
The agent receives a 10-dimensional continuous observation vector:

$$
s_t =
\begin{bmatrix}
\cos(\theta_1), &
\cos(\theta_2), &
\sin(\theta_1), &
\sin(\theta_2), &
x_g, &
y_g, &
\dot{\theta}_1, &
\dot{\theta}_2, &
\Delta x, &
\Delta y
\end{bmatrix}
$$

Notes:

Joint angles ($\theta_1$, $\theta_2$): The angular positions of the two joints.

Joint velocities ($\dot{\theta}_1$, $\dot{\theta}_2$): The angular velocities of the joints, 
measured in radians per second.

Target coordinates ($x_g$, $y_g$): The target's position in the coordinate system.

Position differences ($\Delta x$, $\Delta y$): The difference between the fingertip position 
and the target position, calculated as:

$$
\Delta x = x_{\text{fingertip}} - x_g
$$

$$
\Delta y = y_{\text{fingertip}} - y_g
$$


### 3.2. Action Space
The agent controls the robotic arm by applying torque to its two joints. 

$$
a_t = [\tau_1,\tau_2]
$$

Where:

$\tau_1$: Torque command for the first joint.


$\tau_2$: Torque command for the second joint.

The action is a 2-dimensional continuous vector, so that its space is defined as:

Box(-1.0, 1.0, (2,), float32)

### 3.3. Transition Space Shaping

The MuJoCo simulator then calculates how these torques affect the robot's position and velocity.
The RL agent does not directly specify the desired joint angles or fingertip position. It learns to control the robot indirectly through torque commands.

### 3.4. Reward
The goal is to move the robot's fingertip close to the target while avoiding unnecessarily large control inputs.

The default Reacher-v5 reward combines two components:

$$
{r_t = r_{\text{distance}} + r_{\text{control}}}
$$

#### 3.4.1. Distance Reward
The distance reward measures how close the fingertip is to the target:

$$
r_{\text{distance}} =
-\sqrt{(x_f-x_g)^2+(y_f-y_g)^2}
$$

Where:

$(x_f,y_f)$: Fingertip position.

$(x_g,y_g)$: Target position.

The distance is measured in meters and the negative sign means that the agent receives a larger reward when the fingertip is closer to the target.

#### 3.4.2. Control Penalty

The control penalty discourages the agent from applying unnecessarily large torques:

$$
r_{\text{control}} =
-0.1(\tau_1^2+\tau_2^2)
$$

Where:

$(\tau_1,\tau_2)$: Actions applied to the two joints.

$(0.1)$: Default control penalty weight.

The penalty increases as the magnitude of the applied torque increases. This encourages the policy to reach the target using smaller control inputs when possible.

#### 3.4.3. Total Reward

Combining the distance and control components gives the default reward function. The agent learns to maximize cumulative reward, which encourages reaching the target while limiting control effort.

$$
{
r_t =
-\sqrt{(x_f-x_g)^2+(y_f-y_g)^2}
-0.1(\tau_1^2+\tau_2^2)
}
$$


## 4. PPO vs SAC: conceptual comparison

| Property | PPO | SAC |
|---|---|---|
| Data collection | On-policy | Off-policy with replay buffer |
| Policy optimization | Clipped policy-gradient objective | Entropy-regularized actor-critic |
| Experience reuse | Limited (latest rollout) | Reuses past transitions |
| Continuous actions | Yes | Yes |
| Memory requirement | Relatively modest | Replay buffer costs more RAM |


**Fair-comparison scope:** same task, reward, network hidden-layer sizes (`[64, 64]`), device (`cpu`), training-step budget and evaluation seeds. 







## 5. Step-by-step experiments



### Step A — Training experiments (three seeds)

Training happens *without rendering* to keep it fast. 
The Neural Network has 2 layers with this amount of neurons: [64, 64].
The default duration of an episode is 50 timesteps.
3 different seeds has been considered for training each model; Training seeds = (0, 1, 2).
By training three independent models, we can see whether an 
algorithm consistently performs well or only performed well in 
one particular run.
 
Train three different students using the same learning method.
```bash
python scripts/train.py --alg both --steps 200000 --eval-freq 10000 --seeds 0 1 2
```
To investigate the performance of training model, we need to 
evaluate the current model in specific timestep using five fixed
tasks that specified by evaluation seeds. 
For each of tasks/seeds the mean reward, mean fingertip distance 
and mean control cost for 50 timesteps or 1 episode have been measured.

<p align="center">

[//]: # (  <img src="figures/five_seeds.png" width="78%">)
  <img src="figures/training_steps.png " width="88%">
</p>

```bash
python scripts/plot_training.py
```
### Learning Performance
SAC demonstrates faster and more stable learning than PPO.

SAC improves its average episode return from approximately -9 to -5 within the first 30,000 training steps and maintains relatively stable performance afterward.
In contrast, PPO experiences a significant performance decline during early training, with returns dropping to approximately -60 around 40,000 steps. Although PPO subsequently recovers, it reaches only approximately -7 by 200,000 steps.


  <img src="results/plots/learning_return.png" width="88%">

### Stability Across Training Seeds
The learning curves show that PPO experiences substantial 
performance fluctuations during early training. SAC maintains
relatively stable returns after its initial improvement.
In final evaluation, SAC outperforms PPO across all three 
independently trained models in both episode return and reaching accuracy.

<img src="results/plots/learning_mean_distance.png" width="48%">
  

<img src="results/plots/learning_mean_control_cost.png" width="48%">





```bash
python scripts/evaluate.py --alg both --seeds 0 1 2 --episodes 30
```

### Step C — Evaluation

After training finishes, we want to know how well the models actually
perform. Then evaluate each model on 30 fixed evaluation episodes, 
evaluation seeds in (20000–20029) for three different pretrained 
models.

```bash
python scripts/plot_eval.py
```
### Reaching Precision vs. Control Cost
SAC achieves an average fingertip distance of approximately 8.3 cm, 
compared with 13.8 cm for PPO. 
SAC achieves better reaching accuracy at the expense of 
greater control effort with differnce 0.0061.
<p align="center">
  <img src="results/plots/scatter_return.png" width="48%">
  <img src="results/plots/scatter_mean_control_cost.png" width="48%">
  <img src="results/plots/scatter_mean_distance.png" width="48%">
  <img src="results/plots/scatter_final_distance.png" width="48%">
</p>




Each command plays one deterministic 50-step episode, 
then closes the viewer. For headless systems, omit this step.



| Measure | PPO | SAC | Interpretation |
|---|---|---|---|
| Final held-out mean return  | -7.277| -4.855 | Higher is better |
| Mean fingertip distance  | 0.138 | 0.083| Lower is better |
| Final fingertip distance  | 0.091 | 0.031 | Lower is better |
| Mean  control cost  | 0.0079 | 0.0140| Lower is better |



## References

- [Gymnasium Reacher-v5 task definition](https://gymnasium.farama.org/environments/mujoco/reacher/)
- [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [Stable-Baselines3 SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
- [MuJoCo documentation](https://mujoco.readthedocs.io/)


