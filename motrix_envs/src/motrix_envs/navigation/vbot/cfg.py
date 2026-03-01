# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
from dataclasses import dataclass, field

from motrix_envs import registry
from motrix_envs.base import EnvCfg

model_file = os.path.dirname(__file__) + "/xmls/scene.xml"

@dataclass
class NoiseConfig:
    level: float = 1.0
    scale_joint_angle: float = 0.03
    scale_joint_vel: float = 1.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_linvel: float = 0.1

@dataclass
class ControlConfig:
    stiffness = 60  # [N*m/rad]
    damping = 0.8   # [N*m*s/rad]
    # action scale: target angle = actionScale * action + defaultAngle
    # 0.25 gives effective PD force = 60 * 0.25 = 15 N*m, enough for navigation maneuvers
    action_scale = 0.25

@dataclass
class InitState:
    # the initial position of the robot in the world frame
    pos = [0.0, 0.0, 0.40]
    
    # 位置随机化范围 [x_min, y_min, x_max, y_max]
    pos_randomization_range = [-10.0, -10.0, 10.0, 10.0]  # 在ground上随机分散20m x 20m范围

    # the default angles for all joints. key = joint name, value = target angle [rad]
    # 使用locomotion的关节角度配置
    default_joint_angles = {
        "FR_hip_joint": -0.0,     # 右前髋关节
        "FR_thigh_joint": 0.9,    # 右前大腿
        "FR_calf_joint": -1.8,    # 右前小腿
        "FL_hip_joint": 0.0,      # 左前髋关节
        "FL_thigh_joint": 0.9,    # 左前大腿
        "FL_calf_joint": -1.8,    # 左前小腿
        "RR_hip_joint": -0.0,     # 右后髋关节
        "RR_thigh_joint": 0.9,    # 右后大腿
        "RR_calf_joint": -1.8,    # 右后小腿
        "RL_hip_joint": 0.0,      # 左后髋关节
        "RL_thigh_joint": 0.9,    # 左后大腿
        "RL_calf_joint": -1.8,    # 左后小腿
    }

@dataclass
class Commands:
    # 目标位置相对于机器人初始位置的偏移范围 [dx_min, dy_min, yaw_min, dx_max, dy_max, yaw_max]
    # dx/dy: 相对机器人初始位置的偏移（米）
    # yaw: 目标绝对朝向（弧度），水平方向随机
    pose_command_range = [-5.0, -5.0, -3.14, 5.0, 5.0, 3.14]

@dataclass
class Normalization:
    lin_vel = 2.0
    ang_vel = 0.25
    dof_pos = 1.0
    dof_vel = 0.05

@dataclass
class Asset:
    body_name = "base"
    foot_names = ["FR", "FL", "RR", "RL"]
    terminate_after_contacts_on = ["collision_middle_box", "collision_head_box"]
    ground_name = "ground"  # 平地场景中地面geom名称
    ground_subtree = "C_"  # 地形根节点，用于subtree接触检测
   
@dataclass
class Sensor:
    base_linvel = "base_linvel"
    base_gyro = "base_gyro"
    feet = ["FR", "FL", "RR", "RL"]  # 足部接触力传感器名称

@dataclass
class RewardConfig:
    scales: dict[str, float] = field(
        default_factory=lambda: {
            # ===== 运动奖励（与 walk_np 一致） =====
            "termination": -200.0,
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.5,
            "lin_vel_z": -2.0,
            "ang_vel_xy": -0.05,
            "orientation": -0.0,
            "torques": -0.00001,
            "dof_vel": -0.0,
            "dof_acc": -2.5e-7,
            "base_height": -0.0,
            "feet_air_time": 1.0,
            "collision": -0.0,
            "action_rate": -0.001,
            "stand_still": -0.0,
            "hip_pos": -1.0,
            "calf_pos": -0.0,
            # ===== 导航专用奖励 =====
            "approach": 1.0,
            "arrival_bonus": 10.0,
            "stop_bonus": 1.0,
        }
    )
    tracking_sigma: float = 0.25
    max_foot_height: float = 0.1

@registry.envcfg("vbot_navigation_flat")
@dataclass
class VBotEnvCfg(EnvCfg):
    model_file: str = model_file
    reset_noise_scale: float = 0.01
    max_episode_seconds: float = 20.0
    max_episode_steps: int = 2000
    sim_dt: float = 0.01    # 仿真步长 10ms = 100Hz
    ctrl_dt: float = 0.01
    reset_yaw_scale: float = 0.1
    max_dof_vel: float = 100.0  # 最大关节速度阈值，训练初期给予更大容忍度

    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    normalization: Normalization = field(default_factory=Normalization)
    asset: Asset = field(default_factory=Asset)
    sensor: Sensor = field(default_factory=Sensor)


@registry.envcfg("vbot_navigation_stairs")
@dataclass
class VBotStairsEnvCfg(VBotEnvCfg):
    """VBot在楼梯地形上的导航配置，继承flat配置"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_stairs.xml"
    max_episode_seconds: float = 20.0  # 增加到20秒，给更多时间学习转向
    max_episode_steps: int = 2000
    
    @dataclass
    class ControlConfig:
        action_scale = 0.25  # 楼梯navigation使用0.2，足够转向但比平地更谨慎
    
    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("VBotStairsMultiTarget-v0")
@dataclass
class VBotStairsMultiTargetEnvCfg(VBotStairsEnvCfg):
    """VBot楼梯多目标导航配置，继承单目标配置"""
    max_episode_seconds: float = 60.0  # 多目标需要更长时间
    max_episode_steps: int = 6000


@registry.envcfg("vbot_navigation_stairs_obstacles")
@dataclass
class VBotStairsObstaclesEnvCfg(VBotStairsEnvCfg):
    """VBot楼梯地形带障碍球的导航配置"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_stairs_obstacles.xml"
    max_episode_seconds: float = 20.0
    max_episode_steps: int = 2000

@registry.envcfg("vbot_navigation_section01")
@dataclass
class VBotSection01EnvCfg(VBotStairsEnvCfg):
    """VBot Section01单独训练配置 - 高台楼梯地形"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_section01.xml"
    max_episode_seconds: float = 40.0  # 拉长一倍：从20秒增加到40秒
    max_episode_steps: int = 4000  # 拉长一倍：从2000步增加到4000步
    
    @dataclass
    class InitState:
        # 起始位置：随机化范围内生成
        pos = [0.0, -2.4, 0.35]  # 中心位置（降低初始高度，减少自由落体）
        
        pos_randomization_range = [-0.5, -0.5, 0.5, 0.5]  # X±0.5m, Y±0.5m随机
        
        default_joint_angles = {
            "FR_hip_joint": -0.0,
            "FR_thigh_joint": 0.9,
            "FR_calf_joint": -1.8,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "RR_hip_joint": -0.0,
            "RR_thigh_joint": 0.9,
            "RR_calf_joint": -1.8,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.9,
            "RL_calf_joint": -1.8,
        }
    
    @dataclass
    class Commands:
        # 目标位置：缩短距离，固定目标点
        # 起始位置Y=-2.4, 目标Y=3.6, 距离=6米（与vbot_np相近）
        # pose_command_range = [0.0, 3.6, 0.0, 0.0, 3.6, 0.0]
        
        # 原始配置（已注释）：
        # 目标位置：固定在终止角范围远端（完全无随机化）
        # 固定目标点: X=0, Y=10.2, Z=2 (Z通过XML控制)
        # 起始位置Y=-2.4, 目标Y=10.2, 距离=12.6米
        pose_command_range = [0.0, 10.2, 0.0, 0.0, 10.2, 0.0]
    
    @dataclass
    class ControlConfig:
        action_scale = 0.25
    
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("vbot_navigation_section02")
@dataclass
class VBotSection02EnvCfg(VBotStairsEnvCfg):
    """VBot Section02单独训练配置 - 中间楼梯地形"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_section02.xml"
    max_episode_seconds: float = 60.0  # Section02较复杂，需要更多时间
    max_episode_steps: int = 6000
    
    @dataclass
    class InitState:
        # 起始位置：section02的起始位置（继承自locomotion）
        # pos = [-2.5, 8.5, 1.8]
        # pos = [-2.5, 8.5, 1.8]
        pos = [-2.5, 12.0, 1.8]  # Y坐标对应section02的起点，高度1.8m
        # pos = [-2.5, 15.0, 3.3]  # Y坐标对应section02的起点，高度1.8m
        # pos = [-2.5, 21.0, 3.3]  # Y坐标对应section02的起点，高度1.8m
        # pos = [-2.5, 24.6, 1.8]  # Y坐标对应section02的起点，高度1.8m
        # pos_randomization_range = [-0.5, -0.5, 0.5, 0.5]  # 小范围随机±0.5m
        pos_randomization_range = [-0., -0., 0., 0.]  # 小范围随机±0.5m
        
        default_joint_angles = {
            "FR_hip_joint": -0.0,
            "FR_thigh_joint": 0.9,
            "FR_calf_joint": -1.8,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "RR_hip_joint": -0.0,
            "RR_thigh_joint": 0.9,
            "RR_calf_joint": -1.8,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.9,
            "RL_calf_joint": -1.8,
        }
    
    @dataclass
    class Commands:
        # 目标范围：覆盖section02区域（10-20米）
        pose_command_range = [-3.0, 16.0, 3.14, -3.0, 26.0, 3.14]
    
    @dataclass
    class ControlConfig:
        action_scale = 0.25
    
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("vbot_navigation_section03")
@dataclass
class VBotSection03EnvCfg(VBotStairsEnvCfg):
    """VBot Section03单独训练配置 - 终点楼梯地形"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_section03.xml"
    max_episode_seconds: float = 50.0  # 拉长一倍：从25秒增加到50秒
    max_episode_steps: int = 5000  # 拉长一倍：从2500步增加到5000步
    
    @dataclass
    class InitState:
        # 起始位置：section03的起始位置（继承自locomotion）
        pos = [0.0, 26.0, 1.8]  # Y坐标对应section03的起点，高度1.8m
        pos_randomization_range = [-0.5, -0.5, 0.5, 0.5]  # 小范围随机±0.5m
        
        default_joint_angles = {
            "FR_hip_joint": -0.0,
            "FR_thigh_joint": 0.9,
            "FR_calf_joint": -1.8,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "RR_hip_joint": -0.0,
            "RR_thigh_joint": 0.9,
            "RR_calf_joint": -1.8,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.9,
            "RL_calf_joint": -1.8,
        }
    
    @dataclass
    class Commands:
        # 目标范围：覆盖section03区域（20-32米）
        pose_command_range = [-3.0, 20.0, -3.14, 3.0, 32.0, 3.14]
    
    @dataclass
    class ControlConfig:
        action_scale = 0.25
    
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("vbot_navigation_long_course")
@dataclass
class VBotLongCourseEnvCfg(VBotStairsEnvCfg):
    """VBot三段地形完整导航配置（比赛任务）- 完整统一地图"""
    # 使用完整地图包含所有三个阶段 (section011, section012, section013)
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_section_full.xml"
    max_episode_seconds: float = 180.0  # 三个阶段需要更长时间：3分钟
    max_episode_steps: int = 18000  # 对应180秒 @ 100Hz
    
    @dataclass
    class InitState:
        # 起始位置：section011的起点
        pos = [0.0, -2.4, 0.5]  # 第一阶段起点
        pos_randomization_range = [-0.5, -0.5, 0.5, 0.5]  # 小范围随机±0.5m
        
        default_joint_angles = {
            "FR_hip_joint": -0.0,
            "FR_thigh_joint": 0.9,
            "FR_calf_joint": -1.8,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "RR_hip_joint": -0.0,
            "RR_thigh_joint": 0.9,
            "RR_calf_joint": -1.8,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.9,
            "RL_calf_joint": -1.8,
        }
    
    @dataclass
    class Commands:
        # 目标范围：覆盖三个阶段的完整路线 (Y: -2.4到30米)
        pose_command_range = [-3.0, 5.0, -3.14, 3.0, 30.0, 3.14]
    
    @dataclass
    class ControlConfig:
        action_scale = 0.25  # 与stairs保持一致
    
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)

@registry.envcfg("vbot_navigation_section001")
#通过 @registry.envcfg("vbot_navigation_section001") 注册
@dataclass
class VBotSection001EnvCfg(VBotStairsEnvCfg):
    """VBot Section01单独训练配置 - 高台楼梯地形"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_section001.xml"
    max_episode_seconds: float = 40.0  # 拉长一倍：从20秒增加到40秒
    max_episode_steps: int = 4000  # 拉长一倍：从2000步增加到4000步
    render_spacing: float = 0  # 多环境渲染时不添加间距
    @dataclass
    class InitState:
        # 起始位置：随机化范围内生成
        pos = [0.0, -2.4, 0.5]  # 中心位置
        pos_randomization_range = [-0.5, -0.5, 0.5, 0.5]  # X±0.5m, Y±0.5m随机

        default_joint_angles = {
            "FR_hip_joint": -0.0,
            "FR_thigh_joint": 0.9,
            "FR_calf_joint": -1.8,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "RR_hip_joint": -0.0,
            "RR_thigh_joint": 0.9,
            "RR_calf_joint": -1.8,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.9,
            "RL_calf_joint": -1.8,
        }
    @dataclass
    class Commands:
        # 目标位置：缩短距离，固定目标点
        # 起始位置Y=-2.4, 目标Y=3.6, 距离=6米（与vbot_np相近）
        # pose_command_range = [0.0, 3.6, 0.0, 0.0, 3.6, 0.0]
        # 原始配置（已注释）：
        # 目标位置：固定在终止角范围远端（完全无随机化）
        # 固定目标点: X=0, Y=10.2, Z=2 (Z通过XML控制)
        # 起始位置Y=-2.4, 目标Y=10.2, 距离=12.6米
        pose_command_range = [0.0, 10.2, 0.0, 0.0, 10.2, 0.0]
    @dataclass
    class ControlConfig:
        stiffness = 60   # [N*m/rad] 与 VBotEnvCfg 基类一致
        damping = 0.8    # [N*m*s/rad] 与 VBotEnvCfg 基类一致
        action_scale = 0.25
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)

    # 竞赛场地参数
    min_spawn_distance: float = 2.0   # 出生点距目标(圆心)的最小距离(m) — 从8.0降低，便于探索
    boundary_radius: float = 10.0      # 竞赛场地边界半径(m)
    arena_center: list = field(default_factory=lambda: [0.0, 0.0])  # 圆心坐标

@dataclass
class CompetitionConfig:
    """竞赛模式配置 - 用于支持emoji、红包、终点等竞赛元素"""
    # 起点区域
    start_zone_center: list = field(default_factory=lambda: [0.0, -2.4])
    start_zone_radius: float = 1.0
    
    # Emoji位置 (3个)
    smiley_positions: list = field(default_factory=lambda: [
        [0.0, 0.0],      # Emoji 1
        [0.0, 5.0],      # Emoji 2
        [0.0, 10.0],     # Emoji 3
    ])
    smiley_radius: float = 0.5
    
    # 红包位置 (3个)
    hongbao_positions: list = field(default_factory=lambda: [
        [1.0, 2.5],      # 红包 1
        [1.0, 7.5],      # 红包 2
        [1.0, 12.5],     # 红包 3
    ])
    hongbao_radius: float = 0.5
    
    # 终点区域
    finish_zone_center: list = field(default_factory=lambda: [0.0, 10.2])
    finish_zone_radius: float = 1.0
    
    # 边界限制
    boundary_x_min: float = -20.0
    boundary_x_max: float = 20.0
    boundary_y_min: float = -10.0
    boundary_y_max: float = 20.0
    
    # 庆祝参数
    celebration_duration: float = 1.0
    celebration_movement_threshold: float = 0.1


@registry.envcfg("vbot_navigation_section011")
#通过 @registry.envcfg("vbot_navigation_section011") 注册
@dataclass
class VBotSection011EnvCfg(VBotStairsEnvCfg):
    """VBot Section01单独训练配置 - 高台楼梯地形"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_section011.xml"
    max_episode_seconds: float = 40.0  # 拉长一倍：从20秒增加到40秒
    max_episode_steps: int = 4000  # 拉长一倍：从2000步增加到4000步
    render_spacing: float = 0  # 多环境渲染时不添加间距

    # ===== 课程学习模式 =====
    # 设为True以兼容从section001预训练模型继续训练
    # 开启后：观测空间=54维(与001一致)，动作控制=PD力矩(与001一致)
    # 关闭时：观测空间=81维(含竞赛特征)，动作控制=位置控制
    curriculum_from_001: bool = False

    @dataclass
    class InitState:
        # 起始位置：随机化范围内生成
        pos = [0.0, -2.4, 0.5]  # 中心位置
        pos_randomization_range = [-0.5, -0.5, 0.5, 0.5]  # X±0.5m, Y±0.5m随机

        default_joint_angles = {
            "FR_hip_joint": -0.0,
            "FR_thigh_joint": 0.9,
            "FR_calf_joint": -1.8,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "RR_hip_joint": -0.0,
            "RR_thigh_joint": 0.9,
            "RR_calf_joint": -1.8,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.9,
            "RL_calf_joint": -1.8,
        }
    @dataclass
    class Commands:
        # 目标位置：缩短距离，固定目标点
        # 起始位置Y=-2.4, 目标Y=3.6, 距离=6米（与vbot_np相近）
        # pose_command_range = [0.0, 3.6, 0.0, 0.0, 3.6, 0.0]
        # 原始配置（已注释）：
        # 目标位置：固定在终止角范围远端（完全无随机化）
        # 固定目标点: X=0, Y=10.2, Z=2 (Z通过XML控制)
        # 起始位置Y=-2.4, 目标Y=10.2, 距离=12.6米
        pose_command_range = [0.0, 10.2, 0.0, 0.0, 10.2, 0.0]
    @dataclass
    class ControlConfig:
        stiffness = 60   # [N*m/rad] PD控制刚度（课程模式使用，与section001一致）
        damping = 0.8    # [N*m*s/rad] PD控制阻尼（课程模式使用，与section001一致）
        action_scale = 0.25
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    # 竞赛配置
    competition: CompetitionConfig = field(default_factory=CompetitionConfig)

@registry.envcfg("vbot_navigation_section012")
#通过 @registry.envcfg("vbot_navigation_section012") 注册
@dataclass
class VBotSection012EnvCfg(VBotStairsEnvCfg):
    """VBot Section012 竞赛任务配置 - 完整赛道导航（2026平台 → 丙午大吉平台）"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_section012.xml"
    max_episode_seconds: float = 120.0  # 完整赛道需要充足时间
    max_episode_steps: int = 12000  # 120秒 @ 100Hz

    # ===== 课程学习模式 =====
    # 设为True以兼容从section001预训练模型继续训练
    # 开启后：使用可配置PD参数（stiffness/damping）
    # 关闭时：保持section012默认控制逻辑
    curriculum_from_001: bool = False

    @dataclass
    class InitState:
        # 起始位置："2026"平台中心
        # 平台box中心Z=1.044, 半高0.25 → 表面Z=1.294
        # 机器人站立高度0.462 → 出生Z = 1.294 + 0.462 = 1.756
        pos = [0.0, 10.33, 1.756]
        # 位置随机化范围 [x_min, y_min, x_max, y_max]，相对于pos中心的偏移
        # 平台X范围: -5.0~5.0, Y范围: 8.83~11.83
        pos_randomization_range = [-3.0, -1.0, 3.0, 1.0]

        default_joint_angles = {
            "FR_hip_joint": -0.0,
            "FR_thigh_joint": 0.9,
            "FR_calf_joint": -1.8,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "RR_hip_joint": -0.0,
            "RR_thigh_joint": 0.9,
            "RR_calf_joint": -1.8,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.9,
            "RL_calf_joint": -1.8,
        }
    @dataclass
    class Commands:
        # 终点平台"丙午大吉"的绝对坐标 [x, y, yaw, x, y, yaw]
        # 由于环境代码中直接使用FINISH_ZONE_CENTER，此处仅作参考
        pose_command_range = [0.0, 24.3, 0.0, 0.0, 24.3, 0.0]
    @dataclass
    class ControlConfig:
        stiffness = 60   # [N*m/rad] 课程模式PD刚度（兼容section001/section011）
        damping = 0.8    # [N*m*s/rad] 课程模式PD阻尼（兼容section001/section011）
        action_scale = 0.25
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)

@registry.envcfg("vbot_navigation_section013")
#通过 @registry.envcfg("vbot_navigation_section013") 注册
@dataclass
class VBotSection013EnvCfg(VBotStairsEnvCfg):
    """VBot Section01单独训练配置 - 高台楼梯地形"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_section013.xml"
    max_episode_seconds: float = 40.0  # 拉长一倍：从20秒增加到40秒
    max_episode_steps: int = 4000  # 拉长一倍：从2000步增加到4000步
    @dataclass
    class InitState:
        # 起始位置：随机化范围内生成
        pos = [0.0, 26.0, 3.3]  # 中心位置
        pos_randomization_range = [-1.0, -0.5, 1.0, 0.5]  # X±1.0m, Y±0.5m随机（一票否决：位置必须随机）

        default_joint_angles = {
            "FR_hip_joint": -0.0,
            "FR_thigh_joint": 0.9,
            "FR_calf_joint": -1.8,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "RR_hip_joint": -0.0,
            "RR_thigh_joint": 0.9,
            "RR_calf_joint": -1.8,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.9,
            "RL_calf_joint": -1.8,
        }
    @dataclass
    class Commands:
        # 目标位置：缩短距离，固定目标点
        # 起始位置Y=-2.4, 目标Y=3.6, 距离=6米（与vbot_np相近）
        # pose_command_range = [0.0, 3.6, 0.0, 0.0, 3.6, 0.0]
        # 原始配置（已注释）：
        # 目标位置：固定在终止角范围远端（完全无随机化）
        # 固定目标点: X=0, Y=10.2, Z=2 (Z通过XML控制)
        # 起始位置Y=-2.4, 目标Y=10.2, 距离=12.6米
        pose_command_range = [0.0, 10.2, 0.0, 0.0, 10.2, 0.0]
    @dataclass
    class ControlConfig:
        action_scale = 0.25
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)

