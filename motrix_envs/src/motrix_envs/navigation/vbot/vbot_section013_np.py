import numpy as np
import motrixsim as mtx
import gymnasium as gym

from motrix_envs import registry
from motrix_envs.np.env import NpEnv, NpEnvState
from motrix_envs.math.quaternion import Quaternion

from .cfg import VBotSection013EnvCfg


# ==================== 竞赛场景常量 ====================

# 终点平台（"中国结"）- 视觉模型中心 (0, 32.3)
FINISH_ZONE_CENTER = np.array([0.0, 32.3], dtype=np.float32)
FINISH_ZONE_RADIUS = 1.5  # 终点判定半径（m），身体任何部位进入即算到达

# Y轴方向检查点（递增排列，用于RL稠密奖励塑形）
CHECKPOINTS_Y = np.array([27.5, 29.0, 30.5, 32.0], dtype=np.float32)

# 金球位置（滚动球障碍），来自0126_C_section03.xml
GOLD_BALL_POSITIONS = np.array([
    [3.0, 31.23],
    [0.0, 31.23],
    [-3.0, 31.23],
], dtype=np.float32)
GOLD_BALL_CONTACT_RADIUS = 1.5  # 判定与球接触的半径

# 随机地形区域（heightfield centered at Y≈29.33）
RANDOM_TERRAIN_PASSED_Y = 30.5  # Y坐标超过此值视为穿越随机地形

# 赛道边界（来自XML中的bianjie碰撞体）
BOUNDARY_X_MIN = -4.9
BOUNDARY_X_MAX = 4.9
BOUNDARY_Y_MIN = 24.5
BOUNDARY_Y_MAX = 35.0

# 物理检测参数
FALL_THRESHOLD_ROLL_PITCH = np.deg2rad(45.0)  # roll/pitch超过45°视为摔倒
MIN_STANDING_HEIGHT_RATIO = 0.4  # 低于目标高度40%视为趴下
GRACE_PERIOD_STEPS = 10  # 重置后的物理宽限期（步数）

# 庆祝参数
CELEBRATION_DURATION = 2.0  # 需要在终点停留的时间（秒）

# ==================== 竞赛计分（满分25分） ====================
# 阶段一：穿越滚动球区域（二选一，互斥）
#   策略A（避障优先）：不触碰滚球，安全通过 → +10分
#   策略B（鲁棒优先）：触碰滚球且保持不摔倒 → +15分
PHASE1_COMPLETE_Y = 31.5   # Y阈值：过了球区初始位置Y=31.23后算完成阶段一
PHASE1_AVOID_SCORE = 10.0  # 策略A得分
PHASE1_ROBUST_SCORE = 15.0 # 策略B得分
# 阶段二：穿越随机地形并到达终点"中国结" → +5分
PHASE2_FINISH_SCORE = 5.0
# 阶段三：终点庆祝动作 → +5分
PHASE3_CELEBRATION_SCORE = 5.0

# ==================== RL训练塑形奖励（非竞赛计分） ====================
TERMINATION_PENALTY = -200.0   # 摔倒/越界/姿态失控 → 重罚
CHECKPOINT_SHAPING = 2.0      # Y轴检查点稠密引导
BALL_SURVIVE_SHAPING = 2.0     # 碰球后仍站立的即时鼓励（引导策略B）
HEIGHT_REWARD_SCALE = 2.0
HEIGHT_REWARD_SIGMA = 0.05
FORWARD_VEL_SCALE = 1.0
PROGRESS_SCALE = 2.0


def generate_repeating_array(num_period, num_reset, period_counter):
    """
    生成重复数组，用于在固定位置中循环选择
    num_period: 位置总数
    num_reset: 需要重置的环境数
    period_counter: 当前计数器
    """
    idx = []
    for i in range(num_reset):
        idx.append((period_counter + i) % num_period)
    return np.array(idx)


@registry.env("vbot_navigation_section013", "np")
class VBotSection013Env(NpEnv):
    """
    VBot在Section013地形上的导航任务
    继承自NpEnv，使用VBotSection013EnvCfg配置
    """
    _cfg: VBotSection013EnvCfg
    
    def __init__(self, cfg: VBotSection013EnvCfg, num_envs: int = 1):
        # 调用父类NpEnv初始化
        super().__init__(cfg, num_envs=num_envs)
        
        # 初始化机器人body和接触
        self._body = self._model.get_body(cfg.asset.body_name)
        self._init_contact_geometry()
        
        # 获取目标标记的body
        self._target_marker_body = self._model.get_body("target_marker")
        
        # 获取箭头body（用于可视化，不影响物理）
        try:
            self._robot_arrow_body = self._model.get_body("robot_heading_arrow")
            self._desired_arrow_body = self._model.get_body("desired_heading_arrow")
        except Exception:
            self._robot_arrow_body = None
            self._desired_arrow_body = None
        
        # 动作和观测空间
        self._action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        # 观测空间：67维（55 + 12维接触力）
        self._observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(54,), dtype=np.float32)
        
        self._num_dof_pos = self._model.num_dof_pos
        self._num_dof_vel = self._model.num_dof_vel
        self._num_action = self._model.num_actuators
        
        self._init_dof_pos = self._model.compute_init_dof_pos()
        self._init_dof_vel = np.zeros((self._model.num_dof_vel,), dtype=np.float32)
        
        # 查找target_marker的DOF索引
        self._find_target_marker_dof_indices()
        
        # 查找箭头的DOF索引
        if self._robot_arrow_body is not None and self._desired_arrow_body is not None:
            self._find_arrow_dof_indices()
        
        # 初始化缓存
        self._init_buffer()
        
        # 初始位置生成参数：从配置文件读取
        self.spawn_center = np.array(cfg.init_state.pos, dtype=np.float32)  # 从配置读取
        # 使用配置中的随机化范围 [x_min, y_min, x_max, y_max]
        self.spawn_range = np.array(cfg.init_state.pos_randomization_range, dtype=np.float32)
    
        # 机器人站立目标高度（用于高度维持奖励和摔倒检测）
        # 起始平台表面Z≈1.294, 机器人站立高度≈0.4m
        self.base_height_target = 1.7
    
        # 导航统计计数器
        self.navigation_stats_step = 0
    
    def _init_buffer(self):
        """初始化缓存和参数"""
        cfg = self._cfg
        self.default_angles = np.zeros(self._num_action, dtype=np.float32)
        
        # 归一化系数
        self.commands_scale = np.array(
            [cfg.normalization.lin_vel, cfg.normalization.lin_vel, cfg.normalization.ang_vel],
            dtype=np.float32
        )
        
        # 设置默认关节角度
        for i in range(self._model.num_actuators):
            for name, angle in cfg.init_state.default_joint_angles.items():
                if name in self._model.actuator_names[i]:
                    self.default_angles[i] = angle
        
        self._init_dof_pos[-self._num_action:] = self.default_angles
        self.action_filter_alpha = 0.3
    
    def _find_target_marker_dof_indices(self):
        """查找target_marker在dof_pos中的索引位置"""
        self._target_marker_dof_start = 0
        self._target_marker_dof_end = 3
        self._init_dof_pos[0:3] = [0.0, 0.0, 0.0]
        self._base_quat_start = 6
        self._base_quat_end = 10
    
    def _find_arrow_dof_indices(self):
        """查找箭头在dof_pos中的索引位置"""
        self._robot_arrow_dof_start = 22
        self._robot_arrow_dof_end = 29
        self._desired_arrow_dof_start = 29
        self._desired_arrow_dof_end = 36
        
        arrow_init_height = self._cfg.init_state.pos[2] + 0.5 
        if self._robot_arrow_dof_end <= len(self._init_dof_pos):
            self._init_dof_pos[self._robot_arrow_dof_start:self._robot_arrow_dof_end] = [0.0, 0.0, arrow_init_height, 0.0, 0.0, 0.0, 1.0]
        if self._desired_arrow_dof_end <= len(self._init_dof_pos):
            self._init_dof_pos[self._desired_arrow_dof_start:self._desired_arrow_dof_end] = [0.0, 0.0, arrow_init_height, 0.0, 0.0, 0.0, 1.0]
    
    def _init_contact_geometry(self):
        """初始化接触检测所需的几何体索引"""
        self._init_termination_contact()
        self._init_foot_contact()
    
    def _init_termination_contact(self):
        """初始化终止接触检测：基座geom与地面geom的碰撞"""
        termination_contact_names = self._cfg.asset.terminate_after_contacts_on
        
        # 获取所有地面geom（遍历所有geom，找到包含ground_subtree名称的）
        ground_geoms = []
        ground_prefix = self._cfg.asset.ground_subtree  # "0ground_root"
        for geom_name in self._model.geom_names:
            if geom_name is not None and ground_prefix in geom_name:
                ground_geoms.append(self._model.get_geom_index(geom_name))
        
        # if len(ground_geoms) == 0:
        #     print(f"[Warning] 未找到以 '{ground_prefix}' 开头的地面geom！")
        #     self.termination_contact = np.zeros((0, 2), dtype=np.uint32)
        #     self.num_termination_check = 0
        #     return
        
        # 构建碰撞对：每个基座geom × 每个地面geom
        termination_contact_list = []
        for base_geom_name in termination_contact_names:
            try:
                base_geom_idx = self._model.get_geom_index(base_geom_name)
                for ground_idx in ground_geoms:
                    termination_contact_list.append([base_geom_idx, ground_idx])
            except Exception as e:
                print(f"[Warning] 无法找到基座geom '{base_geom_name}': {e}")
        
        if len(termination_contact_list) > 0:
            self.termination_contact = np.array(termination_contact_list, dtype=np.uint32)
            self.num_termination_check = len(termination_contact_list)
            print(f"[Info] 初始化终止接触检测: {len(termination_contact_names)}个基座geom × {len(ground_geoms)}个地面geom = {self.num_termination_check}个检测对")
        else:
            self.termination_contact = np.zeros((0, 2), dtype=np.uint32)
            self.num_termination_check = 0
            print("[Warning] 未找到任何终止接触geom，基座接触检测将被禁用！")
    
    def _init_foot_contact(self):
        self.foot_contact_check = np.zeros((0, 2), dtype=np.uint32)
        self.num_foot_check = 4  
    
    def get_dof_pos(self, data: mtx.SceneData):
        return self._body.get_joint_dof_pos(data)
    
    def get_dof_vel(self, data: mtx.SceneData):
        return self._body.get_joint_dof_vel(data)
    
    def _extract_root_state(self, data):
        """从self._body中提取根节点状态"""
        pose = self._body.get_pose(data)
        root_pos = pose[:, :3]
        root_quat = pose[:, 3:7]
        root_linvel = self._model.get_sensor_value(self._cfg.sensor.base_linvel, data)
        return root_pos, root_quat, root_linvel
    
    @property
    def observation_space(self):
        return self._observation_space
    
    @property
    def action_space(self):
        return self._action_space
    
    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        # 保存上一步的关节速度（用于计算加速度）
        state.info["last_dof_vel"] = self.get_dof_vel(state.data)
        
        state.info["last_actions"] = state.info["current_actions"]
        
        if "filtered_actions" not in state.info:
            state.info["filtered_actions"] = actions
        else:
            state.info["filtered_actions"] = (
                self.action_filter_alpha * actions + 
                (1.0 - self.action_filter_alpha) * state.info["filtered_actions"]
            )
        
        state.info["current_actions"] = state.info["filtered_actions"]

        state.data.actuator_ctrls = self._compute_torques(state.info["filtered_actions"], state.data)
        
        return state
    
    def _compute_torques(self, actions, data):
        """计算PD控制力矩（VBot使用motor执行器，需要力矩控制）"""
        action_scaled = actions * self._cfg.control_config.action_scale
        target_pos = self.default_angles + action_scaled
        
        # 获取当前关节状态
        current_pos = self.get_dof_pos(data)  # [num_envs, 12]
        current_vel = self.get_dof_vel(data)  # [num_envs, 12]
        
        # PD控制器：tau = kp * (target - current) - kv * vel
        kp = 80.0   # 位置增益
        kv = 6.0    # 速度增益
        
        pos_error = target_pos - current_pos
        torques = kp * pos_error - kv * current_vel
        
        # 限制力矩范围（与XML中的forcerange一致）
        # hip/thigh: ±17 N·m, calf: ±34 N·m
        torque_limits = np.array([17, 17, 34] * 4, dtype=np.float32)  # FR, FL, RR, RL
        torques = np.clip(torques, -torque_limits, torque_limits)
        
        return torques
    
    def _compute_projected_gravity(self, root_quat: np.ndarray) -> np.ndarray:
        """计算机器人坐标系中的重力向量"""
        gravity_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        gravity_vec = np.tile(gravity_vec, (root_quat.shape[0], 1))
        return Quaternion.rotate_inverse(root_quat, gravity_vec)
    
    def _get_heading_from_quat(self, quat: np.ndarray) -> np.ndarray:
        """从四元数计算yaw角（朝向）"""
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        heading = np.arctan2(siny_cosp, cosy_cosp)
        return heading
    
    def _update_target_marker(self, data: mtx.SceneData, pose_commands: np.ndarray):
        """更新目标位置标记的位置和朝向"""
        num_envs = data.shape[0]
        all_dof_pos = data.dof_pos.copy()
        
        for env_idx in range(num_envs):
            target_x = float(pose_commands[env_idx, 0])
            target_y = float(pose_commands[env_idx, 1])
            target_yaw = float(pose_commands[env_idx, 2])
            all_dof_pos[env_idx, self._target_marker_dof_start:self._target_marker_dof_end] = [
                target_x, target_y, target_yaw
            ]
        
        data.set_dof_pos(all_dof_pos, self._model)
        self._model.forward_kinematic(data)
    
    def _update_heading_arrows(self, data: mtx.SceneData, robot_pos: np.ndarray, desired_vel_xy: np.ndarray, base_lin_vel_xy: np.ndarray):
        """更新箭头位置（使用DOF控制freejoint，不影响物理）"""
        if self._robot_arrow_body is None or self._desired_arrow_body is None:
            return
        
        num_envs = data.shape[0]
        arrow_offset = 0.5  # 箭头相对于机器人的高度偏移
        all_dof_pos = data.dof_pos.copy()
        
        for env_idx in range(num_envs):
            # 算箭头高度 = 机器人当前高度 + 偏移
            arrow_height = robot_pos[env_idx, 2] + arrow_offset
            
            # 当前运动方向箭头
            cur_v = base_lin_vel_xy[env_idx]
            if np.linalg.norm(cur_v) > 1e-3:
                cur_yaw = np.arctan2(cur_v[1], cur_v[0])
            else:
                cur_yaw = 0.0
            robot_arrow_pos = np.array([robot_pos[env_idx, 0], robot_pos[env_idx, 1], arrow_height], dtype=np.float32)
            robot_arrow_quat = self._euler_to_quat(0, 0, cur_yaw)
            quat_norm = np.linalg.norm(robot_arrow_quat)
            if quat_norm > 1e-6:
                robot_arrow_quat = robot_arrow_quat / quat_norm
            else:
                robot_arrow_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            all_dof_pos[env_idx, self._robot_arrow_dof_start:self._robot_arrow_dof_end] = np.concatenate([
                robot_arrow_pos, robot_arrow_quat
            ])
            
            # 期望运动方向箭头
            des_v = desired_vel_xy[env_idx]
            if np.linalg.norm(des_v) > 1e-3:
                des_yaw = np.arctan2(des_v[1], des_v[0])
            else:
                des_yaw = 0.0
            desired_arrow_pos = np.array([robot_pos[env_idx, 0], robot_pos[env_idx, 1], arrow_height], dtype=np.float32)
            desired_arrow_quat = self._euler_to_quat(0, 0, des_yaw)
            quat_norm = np.linalg.norm(desired_arrow_quat)
            if quat_norm > 1e-6:
                desired_arrow_quat = desired_arrow_quat / quat_norm
            else:
                desired_arrow_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            all_dof_pos[env_idx, self._desired_arrow_dof_start:self._desired_arrow_dof_end] = np.concatenate([
                desired_arrow_pos, desired_arrow_quat
            ])
        
        data.set_dof_pos(all_dof_pos, self._model)
        self._model.forward_kinematic(data)
    
    def _euler_to_quat(self, roll, pitch, yaw):
        """欧拉角转四元数 [qx, qy, qz, qw] - Motrix格式"""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return np.array([qx, qy, qz, qw], dtype=np.float32)
    
    def update_state(self, state: NpEnvState) -> NpEnvState:
        """
        更新环境状态，计算观测、奖励和终止条件
        """
        data = state.data
        cfg = self._cfg
        
        # 获取基础状态
        root_pos, root_quat, root_vel = self._extract_root_state(data)
        joint_pos = self.get_dof_pos(data)
        joint_vel = self.get_dof_vel(data)
        joint_pos_rel = joint_pos - self.default_angles
        
        # 传感器数据
        base_lin_vel = root_vel[:, :3]  # 世界坐标系线速度
        gyro = self._model.get_sensor_value(cfg.sensor.base_gyro, data)
        projected_gravity = self._compute_projected_gravity(root_quat)
        
        # 导航目标
        pose_commands = state.info["pose_commands"]
        robot_position = root_pos[:, :2]
        robot_heading = self._get_heading_from_quat(root_quat)
        target_position = pose_commands[:, :2]
        target_heading = pose_commands[:, 2]
        
        # 计算位置误差
        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)
        
        # 计算朝向误差
        heading_diff = target_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2*np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2*np.pi, heading_diff)
        
        # 达到判定（使用终点判定半径，与收集状态一致）
        reached_all = distance_to_target < FINISH_ZONE_RADIUS
        
        # 计算期望速度命令（与平地navigation一致，简单P控制器）
        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)
        
        # 角速度命令：跟踪运动方向（从当前位置指向目标）
        # 与vbot_np保持一致的增益和上限，确保转向足够快
        desired_heading = np.arctan2(position_error[:, 1], position_error[:, 0])
        heading_to_movement = desired_heading - robot_heading
        heading_to_movement = np.where(heading_to_movement > np.pi, heading_to_movement - 2*np.pi, heading_to_movement)
        heading_to_movement = np.where(heading_to_movement < -np.pi, heading_to_movement + 2*np.pi, heading_to_movement)
        desired_yaw_rate = np.clip(heading_to_movement * 1.0, -1.0, 1.0)  # 增益和上限与vbot_np一致
        deadband_yaw = np.deg2rad(8)
        desired_yaw_rate = np.where(np.abs(heading_to_movement) < deadband_yaw, 0.0, desired_yaw_rate)
        desired_yaw_rate = np.where(reached_all, 0.0, desired_yaw_rate)
        
        if desired_yaw_rate.ndim > 1:
            desired_yaw_rate = desired_yaw_rate.flatten()
        
        velocity_commands = np.concatenate(
            [desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1
        )
        
        # 归一化观测
        noisy_linvel = base_lin_vel * cfg.normalization.lin_vel
        noisy_gyro = gyro * cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * cfg.normalization.dof_vel
        command_normalized = velocity_commands * self.commands_scale
        last_actions = state.info["current_actions"]
        
        # 任务相关观测
        position_error_normalized = position_error / 5.0
        heading_error_normalized = heading_diff / np.pi
        distance_normalized = np.clip(distance_to_target / 5.0, 0, 1)
        reached_flag = reached_all.astype(np.float32)
        
        stop_ready = np.logical_and(
            reached_all,
            np.abs(gyro[:, 2]) < 5e-2
        )
        stop_ready_flag = stop_ready.astype(np.float32)
        
        obs = np.concatenate(
            [
                noisy_linvel,       # 3
                noisy_gyro,         # 3
                projected_gravity,  # 3
                noisy_joint_angle,  # 12
                noisy_joint_vel,    # 12
                last_actions,       # 12
                command_normalized, # 3
                position_error_normalized,  # 2
                heading_error_normalized[:, np.newaxis],  # 1 - 最终朝向误差（保留）
                distance_normalized[:, np.newaxis],  # 1
                reached_flag[:, np.newaxis],  # 1
                stop_ready_flag[:, np.newaxis],  # 1
            ],
            axis=-1,
        )
        assert obs.shape == (data.shape[0], 54)  # 54 + 1 = 55维
        
        # 更新目标标记和箭头
        self._update_target_marker(data, pose_commands)
        base_lin_vel_xy = base_lin_vel[:, :2]
        self._update_heading_arrows(data, root_pos, desired_vel_xy, base_lin_vel_xy)
        
        # 更新收集/进度状态（检查点、球接触、地形穿越、终点、庆祝）
        self._update_collection_states(data, root_pos, state.info)
        
        # 计算奖励（传递root状态）
        reward = self._compute_reward(data, state.info, root_pos, root_quat, root_vel)
        
        # 计算终止条件（完整的摔倒/越界/姿态/超时检测）
        terminated = self._compute_terminated(data, state.info, root_pos, root_quat, root_vel)
        
        # 将终止惩罚叠加到奖励中
        termination_penalty = state.info.get("termination_penalty", np.zeros(data.shape[0], dtype=np.float32))
        reward = reward + termination_penalty
        
        # 更新时间和步数
        state.info["time_elapsed"] = state.info.get(
            "time_elapsed", np.zeros(data.shape[0], dtype=np.float32)
        ) + self._cfg.sim_dt
        state.info["steps"] = state.info.get(
            "steps", np.zeros(data.shape[0], dtype=np.int32)
        ) + 1
        
        state.obs = obs
        state.reward = reward
        state.terminated = terminated
        
        return state
    
    def _compute_terminated(self, data: mtx.SceneData, info: dict,
                            root_pos: np.ndarray, root_quat: np.ndarray,
                            root_vel: np.ndarray) -> np.ndarray:
        """
        终止条件（满足任一则终止且施加重罚）：
        1. 基座接触地面（摔倒）
        2. 姿态失控（roll/pitch > 45°）
        3. 越界（超出赛道范围）
        4. 高度过低（≤ 40%目标高度 = 坍塌/趴下）
        5. 关节速度异常（overflow / NaN / Inf）
        6. episode超时
        """
        n_envs = data.shape[0]
        steps = info.get("steps", np.zeros(n_envs, dtype=np.int32))
        in_grace = steps < GRACE_PERIOD_STEPS
        
        # 1. 基座接触地面（摔倒）
        try:
            base_contact_value = self._model.get_sensor_value("base_contact", data)
            if base_contact_value.ndim == 0:
                base_contact = np.array([base_contact_value > 0.01], dtype=bool)
            elif base_contact_value.shape[0] != n_envs:
                base_contact = np.full(n_envs, base_contact_value.flatten()[0] > 0.01, dtype=bool)
            else:
                base_contact = (base_contact_value > 0.01).flatten()[:n_envs]
        except Exception:
            base_contact = np.zeros(n_envs, dtype=bool)
        
        # 2. 姿态失控
        qx, qy, qz, qw = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0)
        pitch = np.arcsin(sinp)
        attitude_fail = (np.abs(roll) > FALL_THRESHOLD_ROLL_PITCH) | (np.abs(pitch) > FALL_THRESHOLD_ROLL_PITCH)
        
        # 3. 越界检测
        out_of_bounds = (
            (root_pos[:, 0] < BOUNDARY_X_MIN) |
            (root_pos[:, 0] > BOUNDARY_X_MAX) |
            (root_pos[:, 1] < BOUNDARY_Y_MIN) |
            (root_pos[:, 1] > BOUNDARY_Y_MAX)
        )
        
        # 4. 高度过低
        height_too_low = root_pos[:, 2] < (self.base_height_target * MIN_STANDING_HEIGHT_RATIO)
        
        # 5. 关节速度异常
        dof_vel = self.get_dof_vel(data)
        if dof_vel.ndim > 1:
            vel_max = np.abs(dof_vel).max(axis=1)
            vel_nan_inf = np.isnan(dof_vel).any(axis=1) | np.isinf(dof_vel).any(axis=1)
        else:
            vel_max = np.abs(dof_vel)
            vel_nan_inf = np.isnan(dof_vel) | np.isinf(dof_vel)
        vel_overflow = vel_max > 100.0
        
        # 6. 超时
        timeout = steps >= self._cfg.max_episode_steps
        
        # 综合：物理终止条件在宽限期内不生效
        physical_termination = (
            base_contact | attitude_fail | out_of_bounds |
            height_too_low | vel_overflow | vel_nan_inf
        ) & ~in_grace
        
        terminated = physical_termination | timeout
        
        # 物理终止施加重罚（超时不额外罚分）
        info["termination_penalty"] = np.where(
            physical_termination, TERMINATION_PENALTY, 0.0
        ).astype(np.float32)
        
        return terminated
    
    # ==================== 收集/进度状态更新 ====================
    
    def _update_collection_states(self, data: mtx.SceneData,
                                  root_pos: np.ndarray, info: dict):
        """
        更新三阶段竞赛状态 + RL辅助检查点
        
        竞赛三阶段（满分25分）：
          阶段一：穿越滚动球区域（Y > PHASE1_COMPLETE_Y）
            - 策略A（未碰球）→ +10分
            - 策略B（碰球且存活）→ +15分
          阶段二：到达终点"中国结" → +5分
          阶段三：终点庆祝动作 → +5分
        """
        robot_xy = root_pos[:, :2]
        robot_y = root_pos[:, 1]
        n_envs = root_pos.shape[0]
        
        # 判断当前是否站立（用于球接触存活判定）
        is_standing = root_pos[:, 2] > (self.base_height_target * 0.6)
        
        # ===== Y轴检查点（RL稠密塑形，非竞赛计分） =====
        checkpoints_reached = info["checkpoints_reached"]
        for i, cp_y in enumerate(CHECKPOINTS_Y):
            newly_reached = (robot_y >= cp_y) & ~checkpoints_reached[:, i]
            checkpoints_reached[:, i] |= newly_reached
        
        # ===== 金球接触检测（2D距离检测） =====
        ball_contacted = info["ball_contacted"]
        for i, ball_pos in enumerate(GOLD_BALL_POSITIONS):
            dist = np.linalg.norm(robot_xy - ball_pos, axis=-1)
            newly_contacted = (dist < GOLD_BALL_CONTACT_RADIUS) & ~ball_contacted[:, i]
            ball_contacted[:, i] |= newly_contacted
        
        # 即时球接触存活标记（碰球 + 仍站立 → 用于RL塑形奖励）
        ball_survival_rewarded = info["ball_survival_rewarded"]
        for i in range(len(GOLD_BALL_POSITIONS)):
            newly = ball_contacted[:, i] & ~ball_survival_rewarded[:, i] & is_standing
            ball_survival_rewarded[:, i] |= newly
        
        # ===== 随机地形穿越检测 =====
        terrain_passed = info["terrain_passed"]
        newly_passed = (robot_y >= RANDOM_TERRAIN_PASSED_Y) & ~terrain_passed
        terrain_passed |= newly_passed
        info["terrain_passed"] = terrain_passed
        
        # ===== 阶段一完成检测：Y > PHASE1_COMPLETE_Y（通过球区） =====
        phase1_completed = info["phase1_completed"]
        newly_phase1 = (robot_y >= PHASE1_COMPLETE_Y) & ~phase1_completed
        phase1_completed |= newly_phase1
        info["phase1_completed"] = phase1_completed
        
        # ===== 阶段二：终点到达检测（身体任何部位进入中国结区域即算到达） =====
        finish_dist = np.linalg.norm(robot_xy - FINISH_ZONE_CENTER, axis=-1)
        finish_reached = info["finish_reached"]
        newly_finished = (finish_dist < FINISH_ZONE_RADIUS) & ~finish_reached
        finish_reached |= newly_finished
        info["finish_reached"] = finish_reached
        
        # ===== 阶段三：庆祝 =====
        # 庆祝计时开始
        time_elapsed = info.get("time_elapsed", np.zeros(n_envs, dtype=np.float32))
        celebration_start = info["celebration_start_time"]
        for idx in np.where(newly_finished)[0]:
            if celebration_start[idx] < 0:
                celebration_start[idx] = time_elapsed[idx]
        
        # 庆祝完成检测：到达终点后停留指定时间并有明显关节运动
        celebration_completed = info["celebration_completed"]
        joint_vel = self.get_dof_vel(data)
        for idx in range(n_envs):
            if finish_reached[idx] and not celebration_completed[idx] and celebration_start[idx] >= 0:
                elapsed = time_elapsed[idx] - celebration_start[idx]
                if finish_dist[idx] < FINISH_ZONE_RADIUS:
                    # 在终点区域内，检查是否有明显的关节运动（庆祝动作）
                    joint_speed = np.linalg.norm(joint_vel[idx])
                    if elapsed >= CELEBRATION_DURATION and joint_speed > 0.1:
                        celebration_completed[idx] = True
                else:
                    # 离开终点区域，重置计时
                    celebration_start[idx] = time_elapsed[idx]
    
    def _compute_reward(self, data: mtx.SceneData, info: dict,
                        root_pos: np.ndarray, root_quat: np.ndarray,
                        root_vel: np.ndarray) -> np.ndarray:
        """
        Section013 导航任务奖励计算
        
        === 竞赛计分（满分25分，互斥阶段一 + 阶段二 + 阶段三）===
        阶段一（二选一，互斥）：
          策略A：不触碰滚球，安全通过球区 → +10分
          策略B：触碰滚球且保持不摔倒 → +15分（鼓励抗扰动）
        阶段二：穿越随机地形并到达终点 → +5分
        阶段三：终点庆祝动作 → +5分
        
        === RL训练塑形（稠密信号，非竞赛计分）===
        高度维持、前进速度、Y进度、距离缩减、
        检查点引导、球接触存活鼓励、稳定性惩罚
        """
        cfg = self._cfg
        n_envs = data.shape[0]
        reward = np.zeros(n_envs, dtype=np.float32)
        
        base_lin_vel = root_vel[:, :3]
        base_height = root_pos[:, 2]
        gyro = self._model.get_sensor_value(cfg.sensor.base_gyro, data)
        robot_y = root_pos[:, 1]
        robot_xy = root_pos[:, :2]
        
        # 判断是否站立
        is_standing = base_height > (self.base_height_target * 0.6)
        standing_mask = is_standing.astype(np.float32)
        
        # ================================================================
        #  竞赛计分奖励（一次性里程碑）
        # ================================================================
        
        # ============ 阶段一：穿越滚动球区域（一次性，互斥） ============
        # 当机器人Y > PHASE1_COMPLETE_Y时判定完成
        # 根据是否碰过球选择策略A(+10)或策略B(+15)
        phase1_completed = info["phase1_completed"]
        phase1_rewarded = info["phase1_rewarded"]
        newly_phase1 = phase1_completed & ~phase1_rewarded
        if np.any(newly_phase1):
            # 判断每个环境是否碰过任何球
            any_ball_contacted = info["ball_contacted"].any(axis=1)
            phase1_score = np.where(any_ball_contacted,
                                    PHASE1_ROBUST_SCORE,   # 策略B: +15
                                    PHASE1_AVOID_SCORE)    # 策略A: +10
            reward += newly_phase1.astype(np.float32) * phase1_score
            phase1_rewarded |= newly_phase1
        info["phase1_rewarded"] = phase1_rewarded
        
        # ============ 阶段二：到达终点"中国结"（一次性 +5） ============
        finish_reached = info["finish_reached"]
        finish_rewarded = info["finish_rewarded"]
        newly_finish = finish_reached & ~finish_rewarded
        reward += newly_finish.astype(np.float32) * PHASE2_FINISH_SCORE
        finish_rewarded |= newly_finish
        info["finish_rewarded"] = finish_rewarded
        
        # ============ 阶段三：庆祝完成（一次性 +5） ============
        celebration_completed = info["celebration_completed"]
        celebration_rewarded = info["celebration_rewarded"]
        newly_celebration = celebration_completed & ~celebration_rewarded
        reward += newly_celebration.astype(np.float32) * PHASE3_CELEBRATION_SCORE
        celebration_rewarded |= newly_celebration
        info["celebration_rewarded"] = celebration_rewarded
        
        # ================================================================
        #  RL训练塑形奖励（稠密信号）
        # ================================================================
        
        # ============ 1. 站立高度维持（高斯核奖励） ============
        height_error_sq = np.square(base_height - self.base_height_target)
        reward += HEIGHT_REWARD_SCALE * np.exp(-height_error_sq / HEIGHT_REWARD_SIGMA)
        
        # 存活奖励（站立+0.01，趴下-0.02）
        reward += np.where(is_standing, 0.01, -0.02)
        
        # ============ 2. Y方向前进速度奖励 ============
        forward_vel_y = base_lin_vel[:, 1]
        reward += FORWARD_VEL_SCALE * np.clip(forward_vel_y, -0.3, 1.5) * standing_mask
        
        # ============ 3. Y坐标进度奖励 ============
        last_y = info.get("last_y", robot_y.copy())
        delta_y = robot_y - last_y
        reward += PROGRESS_SCALE * np.clip(delta_y, -0.05, 0.3) * standing_mask
        info["last_y"] = robot_y.copy()
        
        # ============ 4. 终点距离缩减奖励 ============
        distance_to_target = np.linalg.norm(robot_xy - FINISH_ZONE_CENTER, axis=-1)
        last_dist = info.get("last_distance_to_finish", distance_to_target.copy())
        dist_reduction = last_dist - distance_to_target
        reward += np.clip(dist_reduction * 0.5, -0.05, 0.3)
        info["last_distance_to_finish"] = distance_to_target.copy()
        
        # ============ 5. Y轴检查点引导（RL塑形，非竞赛分） ============
        cp_reached = info["checkpoints_reached"]
        cp_rewarded = info["checkpoints_rewarded"]
        for i in range(len(CHECKPOINTS_Y)):
            newly = cp_reached[:, i] & ~cp_rewarded[:, i]
            reward += newly.astype(np.float32) * CHECKPOINT_SHAPING
            cp_rewarded[:, i] |= newly
        
        # ============ 6. 球接触存活鼓励（RL塑形，鼓励策略B） ============
        # 碰到球且仍站立 → 小额即时奖励，引导agent接受碰撞而非绕路
        ball_contacted = info["ball_contacted"]
        ball_survival_rewarded = info["ball_survival_rewarded"]
        ball_survival_shaping_given = info.get("ball_survival_shaping_given",
            np.zeros((n_envs, len(GOLD_BALL_POSITIONS)), dtype=bool))
        for i in range(len(GOLD_BALL_POSITIONS)):
            newly = ball_survival_rewarded[:, i] & ~ball_survival_shaping_given[:, i]
            reward += newly.astype(np.float32) * BALL_SURVIVE_SHAPING
            ball_survival_shaping_given[:, i] |= newly
        info["ball_survival_shaping_given"] = ball_survival_shaping_given
        
        # ============ 稳定性惩罚 ============
        # 姿态惩罚（roll/pitch）
        qx, qy, qz, qw = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0)
        pitch = np.arcsin(sinp)
        reward += (np.abs(roll) + np.abs(pitch)) * -0.5
        
        # 垂直速度惩罚
        reward += np.square(base_lin_vel[:, 2]) * -0.5
        
        # XY角速度惩罚
        reward += np.sum(np.square(gyro[:, :2]), axis=-1) * -0.05
        
        # 动作变化率惩罚
        current_actions = info["current_actions"]
        last_actions = info.get("last_actions", current_actions)
        action_rate = np.sum(np.square(current_actions - last_actions), axis=-1)
        reward += action_rate * -0.01
        
        # 力矩惩罚
        reward += np.sum(np.square(data.actuator_ctrls), axis=-1) * -1e-5
        
        # 关节速度惩罚
        joint_vel = self.get_dof_vel(data)
        reward += np.sum(np.square(joint_vel), axis=-1) * -5e-5
        
        # NaN保护
        reward = np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=-10.0)
        return reward

    def reset(self, data: mtx.SceneData, done: np.ndarray = None) -> tuple[np.ndarray, dict]:
        """
        重置环境
        
        关键要求：机器人初始位置必须随机分布在"丙午大吉"平台上（一票否决项）
        """
        cfg: VBotSection013EnvCfg = self._cfg
        num_envs = data.shape[0]
        
        # ===== 在"丙午大吉"平台上随机生成位置 =====
        # spawn_range = [x_min, y_min, x_max, y_max] 相对于spawn_center的偏移
        random_offset = np.random.uniform(
            low=self.spawn_range[:2],    # [x_min, y_min] 偏移
            high=self.spawn_range[2:],   # [x_max, y_max] 偏移
            size=(num_envs, 2)
        )
        robot_init_xy = self.spawn_center[:2] + random_offset  # [num_envs, 2]
        terrain_heights = np.full(num_envs, self.spawn_center[2], dtype=np.float32)
        robot_init_xyz = np.column_stack([robot_init_xy, terrain_heights])  # [num_envs, 3]
        
        # 随机初始朝向（大致朝向+Y方向 = 赛道前进方向）
        spawn_yaw = np.random.uniform(-np.pi / 6, np.pi / 6, num_envs)
        spawn_quat = np.array([self._euler_to_quat(0, 0, yaw) for yaw in spawn_yaw])
        
        dof_pos = np.tile(self._init_dof_pos, (num_envs, 1))
        dof_vel = np.tile(self._init_dof_vel, (num_envs, 1))
        
        # 设置 base 的 XYZ位置（DOF 3-5）和朝向（DOF 6-9）
        dof_pos[:, 3:6] = robot_init_xyz
        dof_pos[:, self._base_quat_start:self._base_quat_end] = spawn_quat
        
        # 目标位置：固定为中国结平台（绝对坐标）
        target_positions = np.tile(FINISH_ZONE_CENTER, (num_envs, 1))  # [num_envs, 2]
        target_headings = np.zeros((num_envs, 1), dtype=np.float32)   # 到达即可，朝向不限
        pose_commands = np.concatenate([target_positions, target_headings], axis=1)
        
        # 归一化所有四元数
        for env_idx in range(num_envs):
            # base四元数
            quat = dof_pos[env_idx, self._base_quat_start:self._base_quat_end]
            qn = np.linalg.norm(quat)
            dof_pos[env_idx, self._base_quat_start:self._base_quat_end] = (
                quat / qn if qn > 1e-6 else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            )
            
            # 箭头四元数（如果存在）
            if self._robot_arrow_body is not None:
                for start, end in [
                    (self._robot_arrow_dof_start + 3, self._robot_arrow_dof_end),
                    (self._desired_arrow_dof_start + 3, self._desired_arrow_dof_end),
                ]:
                    if end <= len(dof_pos[env_idx]):
                        aq = dof_pos[env_idx, start:end]
                        aqn = np.linalg.norm(aq)
                        dof_pos[env_idx, start:end] = (
                            aq / aqn if aqn > 1e-6 else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                        )
        
        data.reset(self._model)
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)
        
        # 更新目标位置标记
        self._update_target_marker(data, pose_commands)
        
        # 获取根节点状态
        root_pos, root_quat, root_vel = self._extract_root_state(data)
        
        # 关节状态
        joint_pos = self.get_dof_pos(data)
        joint_vel = self.get_dof_vel(data)
        joint_pos_rel = joint_pos - self.default_angles
        
        # 传感器数据
        base_lin_vel = root_vel[:, :3]
        gyro = self._model.get_sensor_value(self._cfg.sensor.base_gyro, data)
        projected_gravity = self._compute_projected_gravity(root_quat)
        
        # 计算速度命令（指向终点的P控制器）
        robot_position = root_pos[:, :2]
        robot_heading = self._get_heading_from_quat(root_quat)
        target_position = pose_commands[:, :2]
        target_heading = pose_commands[:, 2]
        
        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)
        
        reached_all = distance_to_target < FINISH_ZONE_RADIUS
        
        # 计算期望速度
        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)
        
        base_lin_vel_xy = base_lin_vel[:, :2]
        self._update_heading_arrows(data, root_pos, desired_vel_xy, base_lin_vel_xy)
        
        heading_diff = target_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2*np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2*np.pi, heading_diff)
        
        # 角速度跟踪运动方向
        desired_heading = np.arctan2(position_error[:, 1], position_error[:, 0])
        heading_to_movement = desired_heading - robot_heading
        heading_to_movement = np.where(heading_to_movement > np.pi, heading_to_movement - 2*np.pi, heading_to_movement)
        heading_to_movement = np.where(heading_to_movement < -np.pi, heading_to_movement + 2*np.pi, heading_to_movement)
        desired_yaw_rate = np.clip(heading_to_movement * 1.0, -1.0, 1.0)
        
        deadband_yaw = np.deg2rad(8)
        desired_yaw_rate = np.where(np.abs(heading_to_movement) < deadband_yaw, 0.0, desired_yaw_rate)
        desired_yaw_rate = np.where(reached_all, 0.0, desired_yaw_rate)
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)
        
        if desired_yaw_rate.ndim > 1:
            desired_yaw_rate = desired_yaw_rate.flatten()
        
        velocity_commands = np.concatenate(
            [desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1
        )
        
        # 归一化观测
        noisy_linvel = base_lin_vel * self._cfg.normalization.lin_vel
        noisy_gyro = gyro * self._cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * self._cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * self._cfg.normalization.dof_vel
        command_normalized = velocity_commands * self.commands_scale
        last_actions = np.zeros((num_envs, self._num_action), dtype=np.float32)
        
        # 任务相关观测
        position_error_normalized = position_error / 5.0
        heading_error_normalized = heading_diff / np.pi
        distance_normalized = np.clip(distance_to_target / 5.0, 0, 1)
        reached_flag = reached_all.astype(np.float32)
        
        stop_ready = np.logical_and(
            reached_all,
            np.abs(gyro[:, 2]) < 5e-2
        )
        stop_ready_flag = stop_ready.astype(np.float32)
        
        obs = np.concatenate(
            [
                noisy_linvel,       # 3
                noisy_gyro,         # 3
                projected_gravity,  # 3
                noisy_joint_angle,  # 12
                noisy_joint_vel,    # 12
                last_actions,       # 12
                command_normalized, # 3
                position_error_normalized,  # 2
                heading_error_normalized[:, np.newaxis],  # 1
                distance_normalized[:, np.newaxis],  # 1
                reached_flag[:, np.newaxis],  # 1
                stop_ready_flag[:, np.newaxis],  # 1
            ],
            axis=-1,
        )
        assert obs.shape == (num_envs, 54), f"Expected obs shape (*, 54), got {obs.shape}"
        
        # ===== 构建初始信息（包含所有追踪字段） =====
        distance_to_finish = np.linalg.norm(robot_init_xy - FINISH_ZONE_CENTER, axis=-1)
        
        info = {
            # 目标命令
            "pose_commands": pose_commands,
            # 动作
            "last_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "current_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "filtered_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            # 计步与计时
            "steps": np.zeros(num_envs, dtype=np.int32),
            "time_elapsed": np.zeros(num_envs, dtype=np.float32),
            # Y进度追踪
            "last_y": robot_init_xy[:, 1].copy(),
            "last_distance_to_finish": distance_to_finish.copy(),
            # Y轴检查点（RL稠密塑形）
            "checkpoints_reached": np.zeros((num_envs, len(CHECKPOINTS_Y)), dtype=bool),
            "checkpoints_rewarded": np.zeros((num_envs, len(CHECKPOINTS_Y)), dtype=bool),
            # 滚动球接触检测（3个金球）
            "ball_contacted": np.zeros((num_envs, len(GOLD_BALL_POSITIONS)), dtype=bool),
            "ball_survival_rewarded": np.zeros((num_envs, len(GOLD_BALL_POSITIONS)), dtype=bool),
            "ball_survival_shaping_given": np.zeros((num_envs, len(GOLD_BALL_POSITIONS)), dtype=bool),
            # 随机地形穿越
            "terrain_passed": np.zeros(num_envs, dtype=bool),
            # === 竞赛三阶段追踪 ===
            # 阶段一：穿越球区（互斥评分：策略A +10 / 策略B +15）
            "phase1_completed": np.zeros(num_envs, dtype=bool),
            "phase1_rewarded": np.zeros(num_envs, dtype=bool),
            # 阶段二：到达终点（+5）
            "finish_reached": np.zeros(num_envs, dtype=bool),
            "finish_rewarded": np.zeros(num_envs, dtype=bool),
            # 阶段三：庆祝（+5）
            "celebration_start_time": np.full(num_envs, -1.0, dtype=np.float32),
            "celebration_completed": np.zeros(num_envs, dtype=bool),
            "celebration_rewarded": np.zeros(num_envs, dtype=bool),
            # 物理
            "last_dof_vel": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "contacts": np.zeros((num_envs, self.num_foot_check), dtype=np.bool_),
        }
        
        return obs, info
    