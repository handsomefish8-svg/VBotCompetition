import numpy as np
import motrixsim as mtx
import gymnasium as gym

from motrix_envs import registry
from motrix_envs.np.env import NpEnv, NpEnvState
from motrix_envs.math.quaternion import Quaternion

from .cfg import VBotSection012EnvCfg


# ==================== 竞赛场景常量 ====================

# 终点平台（"丙午大吉"）- 中心 (0, 24.33)，范围 Y: 23.33~25.33，X: -5~5
FINISH_ZONE_CENTER = np.array([0.0, 24.33], dtype=np.float32)
FINISH_ZONE_RADIUS = 1.5  # 终点判定半径（m）

# Y轴方向检查点（递增排列，机器人Y坐标超过后获得一次性奖励）
# 对应赛道进度：平台边缘 → 楼梯顶部 → 中段（桥/河床区域） → 接近北侧出口 → 下楼梯区域
CHECKPOINTS_Y = np.array([12.0, 14.5, 18.0, 21.0, 23.0], dtype=np.float32)
CHECKPOINT_BONUS = 5.0

# 贺礼红包（右侧河床石头上，5个，每个3分，共15分）
HELI_HONGBAO_POS = np.array([
    [0.35, 16.27],   # 石头1附近
    [3.50, 16.28],   # 石头2附近
    [2.00, 18.10],   # 中心石头附近
    [0.36, 19.79],   # 石头4附近
    [3.49, 19.82],   # 石头5附近
], dtype=np.float32)
HELI_HONGBAO_RADIUS = 1.5   # 收集半径
HELI_HONGBAO_BONUS = 3.0

# 拜年红包（吊桥区域，2个，每个2.5分，共5分）
BAINIAN_HONGBAO_POS = np.array([
    [-3.0, 17.82],   # 吊桥桥面
    [-2.0, 17.82],   # 吊桥下方
], dtype=np.float32)
BAINIAN_HONGBAO_RADIUS = 1.5
BAINIAN_HONGBAO_BONUS = 2.5

# 赛道边界
BOUNDARY_X_MIN = -5.5
BOUNDARY_X_MAX = 5.5
BOUNDARY_Y_MIN = 8.0
BOUNDARY_Y_MAX = 26.0

# 物理检测参数
FALL_THRESHOLD_ROLL_PITCH = np.deg2rad(45.0)
MIN_STANDING_HEIGHT_RATIO = 0.4   # 低于目标高度40%视为摔倒
GRACE_PERIOD_STEPS = 10           # 重置后的宽限期（步数）

# 庆祝参数
CELEBRATION_DURATION = 2.0        # 需要在终点停留的时间（秒）

# 奖励参数
FINISH_BONUS = 20.0
CELEBRATION_BONUS = 5.0
TERMINATION_PENALTY = -200.0
HEIGHT_REWARD_SCALE = 2.0
HEIGHT_REWARD_SIGMA = 0.05
FORWARD_VEL_SCALE = 1.0
PROGRESS_SCALE = 2.0


def generate_repeating_array(num_period, num_reset, period_counter):
    """生成重复数组，用于在固定位置中循环选择"""
    idx = []
    for i in range(num_reset):
        idx.append((period_counter + i) % num_period)
    return np.array(idx)


@registry.env("vbot_navigation_section012", "np")
class VBotSection012Env(NpEnv):
    """
    VBot Section012 竞赛导航环境

    任务：从"2026"平台出发，穿越复杂地形（波浪→楼梯→吊桥/河床→楼梯→终点），
    到达"丙午大吉"终点平台，并完成庆祝动作。

    得分项：
    - 突破波浪地形 (+10)
    - 路线选择 (+5)
    - 吊桥路线 (+10) / 河床路线 (+5)
    - 终点下楼 (+5)
    - 庆祝动作 (+5)
    - 贺礼红包 (5×3=+15) / 拜年红包 (+5)

    一票否决：摔倒、越界、出发位置不随机
    """
    _cfg: VBotSection012EnvCfg

    def __init__(self, cfg: VBotSection012EnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)

        # 课程学习模式（兼容section001预训练）
        self._curriculum_from_001 = bool(getattr(cfg, "curriculum_from_001", False))

        # 机器人body和接触
        self._body = self._model.get_body(cfg.asset.body_name)
        self._init_contact_geometry()

        # 目标标记body
        self._target_marker_body = self._model.get_body("target_marker")

        # 箭头body（可视化用）
        try:
            self._robot_arrow_body = self._model.get_body("robot_heading_arrow")
            self._desired_arrow_body = self._model.get_body("desired_heading_arrow")
        except Exception:
            self._robot_arrow_body = None
            self._desired_arrow_body = None

        # 动作和观测空间
        self._action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(54,), dtype=np.float32)

        self._num_dof_pos = self._model.num_dof_pos
        self._num_dof_vel = self._model.num_dof_vel
        self._num_action = self._model.num_actuators

        self._init_dof_pos = self._model.compute_init_dof_pos()
        self._init_dof_vel = np.zeros((self._model.num_dof_vel,), dtype=np.float32)

        # DOF索引
        self._find_target_marker_dof_indices()
        if self._robot_arrow_body is not None and self._desired_arrow_body is not None:
            self._find_arrow_dof_indices()

        # 初始化缓存
        self._init_buffer()

        # 起始位置参数
        self.spawn_center = np.array(cfg.init_state.pos, dtype=np.float32)
        self.spawn_range = np.array(cfg.init_state.pos_randomization_range, dtype=np.float32)

        # 站立高度目标（= 平台表面高度 + 机器人站立高度）
        self.base_height_target = cfg.init_state.pos[2]

    # ==================== 属性 ====================

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    # ==================== 初始化辅助 ====================

    def _init_buffer(self):
        """初始化缓存和参数"""
        cfg = self._cfg
        self.default_angles = np.zeros(self._num_action, dtype=np.float32)

        self.commands_scale = np.array(
            [cfg.normalization.lin_vel, cfg.normalization.lin_vel, cfg.normalization.ang_vel],
            dtype=np.float32
        )

        for i in range(self._model.num_actuators):
            for name, angle in cfg.init_state.default_joint_angles.items():
                if name in self._model.actuator_names[i]:
                    self.default_angles[i] = angle

        self._init_dof_pos[-self._num_action:] = self.default_angles
        self.action_filter_alpha = 0.3

        # PD参数（与section011一致：来自cfg.control_config）
        self.kps = np.ones(self._num_action, dtype=np.float32) * getattr(cfg.control_config, "stiffness", 60.0)
        self.kds = np.ones(self._num_action, dtype=np.float32) * getattr(cfg.control_config, "damping", 0.8)

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
            self._init_dof_pos[self._robot_arrow_dof_start:self._robot_arrow_dof_end] = [
                0.0, 0.0, arrow_init_height, 0.0, 0.0, 0.0, 1.0
            ]
        if self._desired_arrow_dof_end <= len(self._init_dof_pos):
            self._init_dof_pos[self._desired_arrow_dof_start:self._desired_arrow_dof_end] = [
                0.0, 0.0, arrow_init_height, 0.0, 0.0, 0.0, 1.0
            ]

    def _init_contact_geometry(self):
        """初始化接触检测"""
        self._init_termination_contact()
        self._init_foot_contact()

    def _init_termination_contact(self):
        """初始化终止接触检测"""
        termination_contact_names = self._cfg.asset.terminate_after_contacts_on

        ground_geoms = []
        ground_prefix = self._cfg.asset.ground_subtree
        for geom_name in self._model.geom_names:
            if geom_name is not None and ground_prefix in geom_name:
                ground_geoms.append(self._model.get_geom_index(geom_name))

        termination_contact_list = []
        for base_geom_name in termination_contact_names:
            try:
                base_geom_idx = self._model.get_geom_index(base_geom_name)
                for ground_idx in ground_geoms:
                    termination_contact_list.append([base_geom_idx, ground_idx])
            except Exception as e:
                print(f"[Warning] 无法找到geom '{base_geom_name}': {e}")

        if len(termination_contact_list) > 0:
            self.termination_contact = np.array(termination_contact_list, dtype=np.uint32)
            self.num_termination_check = len(termination_contact_list)
        else:
            self.termination_contact = np.zeros((0, 2), dtype=np.uint32)
            self.num_termination_check = 0

    def _init_foot_contact(self):
        self.foot_contact_check = np.zeros((0, 2), dtype=np.uint32)
        self.num_foot_check = 4

    # ==================== 状态访问 ====================

    def get_dof_pos(self, data: mtx.SceneData):
        return self._body.get_joint_dof_pos(data)

    def get_dof_vel(self, data: mtx.SceneData):
        return self._body.get_joint_dof_vel(data)

    def _extract_root_state(self, data):
        """提取根节点状态"""
        pose = self._body.get_pose(data)
        root_pos = pose[:, :3]
        root_quat = pose[:, 3:7]
        root_vel = self._model.get_sensor_value(self._cfg.sensor.base_linvel, data)
        return root_pos, root_quat, root_vel

    def _compute_projected_gravity(self, root_quat: np.ndarray) -> np.ndarray:
        """计算机器人坐标系中的重力向量"""
        gravity_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        gravity_vec = np.tile(gravity_vec, (root_quat.shape[0], 1))
        return Quaternion.rotate_inverse(root_quat, gravity_vec)

    def _get_heading_from_quat(self, quat: np.ndarray) -> np.ndarray:
        """从四元数提取yaw角"""
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        return np.arctan2(siny_cosp, cosy_cosp)

    def _euler_to_quat(self, roll, pitch, yaw):
        """欧拉角 → 四元数 [qx, qy, qz, qw]"""
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

    # ==================== 动作控制 ====================

    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        """应用动作：带低通滤波的PD力矩控制"""
        state.info["last_dof_vel"] = self.get_dof_vel(state.data)
        state.info["last_actions"] = state.info["current_actions"]

        if "filtered_actions" not in state.info:
            state.info["filtered_actions"] = actions
        else:
            state.info["filtered_actions"] = (
                self.action_filter_alpha * actions
                + (1.0 - self.action_filter_alpha) * state.info["filtered_actions"]
            )

        state.info["current_actions"] = state.info["filtered_actions"]
        state.data.actuator_ctrls = self._compute_torques(
            state.info["filtered_actions"], state.data
        )
        return state

    def _compute_torques(self, actions, data):
        """PD力矩控制"""
        action_scaled = actions * self._cfg.control_config.action_scale
        current_pos = self.get_dof_pos(data)
        current_vel = self.get_dof_vel(data)

        if self._curriculum_from_001:
            # 课程模式：与section001/section011一致的PD公式
            torques = self.kps * (action_scaled + self.default_angles - current_pos) - self.kds * current_vel
            return torques

        # section012默认模式：与section011同源PD配置，并保留力矩限幅
        target_pos = self.default_angles + action_scaled
        pos_error = target_pos - current_pos
        torques = self.kps * pos_error - self.kds * current_vel
        torque_limits = np.array([17, 17, 34] * 4, dtype=np.float32)
        torques = np.clip(torques, -torque_limits, torque_limits)
        return torques

    # ==================== 观测计算 ====================

    def _compute_obs(self, data: mtx.SceneData, root_pos: np.ndarray,
                     root_quat: np.ndarray, root_vel: np.ndarray,
                     last_actions: np.ndarray, info: dict) -> np.ndarray:
        """
        计算54维观测向量
        [linvel:3, gyro:3, gravity:3, joint_angle:12, joint_vel:12,
         last_actions:12, commands:3, pos_error:2, heading_error:1,
         distance:1, reached:1, stop_ready:1] = 54
        """
        cfg = self._cfg
        num_envs = data.shape[0]

        joint_pos = self.get_dof_pos(data)
        joint_vel = self.get_dof_vel(data)
        joint_pos_rel = joint_pos - self.default_angles

        base_lin_vel = root_vel[:, :3]
        gyro = self._model.get_sensor_value(cfg.sensor.base_gyro, data)
        projected_gravity = self._compute_projected_gravity(root_quat)

        # 导航目标：终点平台
        target_position = np.tile(FINISH_ZONE_CENTER, (num_envs, 1))
        target_heading = np.zeros(num_envs, dtype=np.float32)

        robot_position = root_pos[:, :2]
        robot_heading = self._get_heading_from_quat(root_quat)

        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)

        # 朝向误差
        heading_diff = target_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2 * np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2 * np.pi, heading_diff)

        # 到达判定
        reached_all = distance_to_target < FINISH_ZONE_RADIUS

        # 速度命令（P控制器，指向终点）
        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)

        # 角速度命令（跟踪朝向终点的运动方向）
        desired_heading = np.arctan2(position_error[:, 1], position_error[:, 0])
        heading_to_target = desired_heading - robot_heading
        heading_to_target = np.where(heading_to_target > np.pi, heading_to_target - 2 * np.pi, heading_to_target)
        heading_to_target = np.where(heading_to_target < -np.pi, heading_to_target + 2 * np.pi, heading_to_target)
        desired_yaw_rate = np.clip(heading_to_target * 1.0, -1.0, 1.0)
        deadband_yaw = np.deg2rad(8)
        desired_yaw_rate = np.where(np.abs(heading_to_target) < deadband_yaw, 0.0, desired_yaw_rate)
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

        position_error_normalized = position_error / 5.0
        heading_error_normalized = heading_diff / np.pi
        distance_normalized = np.clip(distance_to_target / 5.0, 0, 1)
        reached_flag = reached_all.astype(np.float32)

        stop_ready = np.logical_and(reached_all, np.abs(gyro[:, 2]) < 5e-2)
        stop_ready_flag = stop_ready.astype(np.float32)

        obs = np.concatenate([
            noisy_linvel,                                    # 3
            noisy_gyro,                                      # 3
            projected_gravity,                               # 3
            noisy_joint_angle,                               # 12
            noisy_joint_vel,                                 # 12
            last_actions,                                    # 12
            command_normalized,                              # 3
            position_error_normalized,                       # 2
            heading_error_normalized[:, np.newaxis],         # 1
            distance_normalized[:, np.newaxis],              # 1
            reached_flag[:, np.newaxis],                     # 1
            stop_ready_flag[:, np.newaxis],                  # 1
        ], axis=-1)

        assert obs.shape == (num_envs, 54), f"Expected obs shape (*, 54), got {obs.shape}"

        # 存储中间量供奖励和可视化使用
        info["velocity_commands"] = velocity_commands
        info["desired_vel_xy"] = desired_vel_xy
        info["distance_to_target"] = distance_to_target
        info["reached_finish"] = reached_all

        return obs

    # ==================== 状态更新 ====================

    def update_state(self, state: NpEnvState) -> NpEnvState:
        """更新环境状态（观测、奖励、终止）"""
        data = state.data
        info = state.info

        # 提取根节点状态
        root_pos, root_quat, root_vel = self._extract_root_state(data)

        # 计算观测
        obs = self._compute_obs(data, root_pos, root_quat, root_vel,
                                info["current_actions"], info)

        # 更新可视化
        pose_commands_vis = np.tile(
            np.array([FINISH_ZONE_CENTER[0], FINISH_ZONE_CENTER[1], 0.0], dtype=np.float32),
            (data.shape[0], 1)
        )
        self._update_target_marker(data, pose_commands_vis)
        base_lin_vel_xy = root_vel[:, :2]
        self._update_heading_arrows(
            data, root_pos,
            info.get("desired_vel_xy", np.zeros((data.shape[0], 2), dtype=np.float32)),
            base_lin_vel_xy
        )

        # 更新收集/触发状态
        self._update_collection_states(data, root_pos, info)

        # 计算奖励
        reward = self._compute_reward(data, info, root_pos, root_quat, root_vel)

        # 计算终止条件
        terminated = self._compute_terminated(data, info, root_pos, root_quat, root_vel)

        # 将终止惩罚叠加到奖励中
        termination_penalty = info.get("termination_penalty", np.zeros(data.shape[0], dtype=np.float32))
        reward = reward + termination_penalty

        # 更新时间和步数
        info["time_elapsed"] = info.get(
            "time_elapsed", np.zeros(data.shape[0], dtype=np.float32)
        ) + self._cfg.sim_dt
        info["steps"] = info.get(
            "steps", np.zeros(data.shape[0], dtype=np.int32)
        ) + 1

        state.obs = obs
        state.reward = reward
        state.terminated = terminated
        return state

    # ==================== 收集状态更新 ====================

    def _update_collection_states(self, data: mtx.SceneData,
                                  root_pos: np.ndarray, info: dict):
        """更新检查点、红包、终点的收集/触发状态"""
        robot_xy = root_pos[:, :2]
        robot_y = root_pos[:, 1]
        n_envs = root_pos.shape[0]

        # ===== Y轴检查点 =====
        checkpoints_reached = info["checkpoints_reached"]
        for i, cp_y in enumerate(CHECKPOINTS_Y):
            newly_reached = (robot_y >= cp_y) & ~checkpoints_reached[:, i]
            checkpoints_reached[:, i] |= newly_reached

        # ===== 贺礼红包收集（2D距离检测） =====
        heli_collected = info["heli_collected"]
        for i, hb_pos in enumerate(HELI_HONGBAO_POS):
            dist = np.linalg.norm(robot_xy - hb_pos, axis=-1)
            newly_collected = (dist < HELI_HONGBAO_RADIUS) & ~heli_collected[:, i]
            heli_collected[:, i] |= newly_collected

        # ===== 拜年红包收集 =====
        bainian_collected = info["bainian_collected"]
        for i, hb_pos in enumerate(BAINIAN_HONGBAO_POS):
            dist = np.linalg.norm(robot_xy - hb_pos, axis=-1)
            newly_collected = (dist < BAINIAN_HONGBAO_RADIUS) & ~bainian_collected[:, i]
            bainian_collected[:, i] |= newly_collected

        # ===== 终点到达 =====
        finish_dist = np.linalg.norm(robot_xy - FINISH_ZONE_CENTER, axis=-1)
        finish_reached = info["finish_reached"]
        newly_finished = (finish_dist < FINISH_ZONE_RADIUS) & ~finish_reached
        finish_reached |= newly_finished

        # 庆祝计时开始
        time_elapsed = info.get("time_elapsed", np.zeros(n_envs, dtype=np.float32))
        celebration_start = info["celebration_start_time"]
        for idx in np.where(newly_finished)[0]:
            if celebration_start[idx] < 0:
                celebration_start[idx] = time_elapsed[idx]

        # 庆祝完成检测
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

    # ==================== 奖励计算 ====================

    def _compute_reward(self, data: mtx.SceneData, info: dict,
                        root_pos: np.ndarray, root_quat: np.ndarray,
                        root_vel: np.ndarray) -> np.ndarray:
        """
        计算每步奖励

        组成：
        1. 站立高度维持（高斯核） — 防止趴下
        2. Y方向前进速度 — 鼓励前进
        3. Y坐标增量 — 进度奖励
        4. 终点距离缩减 — 全局引导
        5. 检查点到达（一次性）
        6. 红包收集（一次性）
        7. 终点到达（一次性）
        8. 庆祝完成（一次性）
        9. 红包靠近引导（密集）
        10. 稳定性惩罚
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
        distance_to_target = info.get("distance_to_target",
                                      np.linalg.norm(robot_xy - FINISH_ZONE_CENTER, axis=-1))
        last_dist = info.get("last_distance_to_finish", distance_to_target.copy())
        dist_reduction = last_dist - distance_to_target
        reward += np.clip(dist_reduction * 0.5, -0.05, 0.3)
        info["last_distance_to_finish"] = distance_to_target.copy()

        # ============ 5. 检查点奖励（一次性） ============
        cp_reached = info["checkpoints_reached"]
        cp_rewarded = info["checkpoints_rewarded"]
        for i in range(len(CHECKPOINTS_Y)):
            newly = cp_reached[:, i] & ~cp_rewarded[:, i]
            reward += newly.astype(np.float32) * CHECKPOINT_BONUS
            cp_rewarded[:, i] |= newly

        # ============ 6. 贺礼红包收集奖励（一次性） ============
        heli_collected = info["heli_collected"]
        heli_rewarded = info["heli_rewarded"]
        for i in range(len(HELI_HONGBAO_POS)):
            newly = heli_collected[:, i] & ~heli_rewarded[:, i]
            reward += newly.astype(np.float32) * HELI_HONGBAO_BONUS
            heli_rewarded[:, i] |= newly

        # ============ 7. 拜年红包收集奖励（一次性） ============
        bainian_collected = info["bainian_collected"]
        bainian_rewarded = info["bainian_rewarded"]
        for i in range(len(BAINIAN_HONGBAO_POS)):
            newly = bainian_collected[:, i] & ~bainian_rewarded[:, i]
            reward += newly.astype(np.float32) * BAINIAN_HONGBAO_BONUS
            bainian_rewarded[:, i] |= newly

        # ============ 8. 终点到达奖励（一次性） ============
        finish_reached = info["finish_reached"]
        finish_rewarded = info["finish_rewarded"]
        newly_finish = finish_reached & ~finish_rewarded
        reward += newly_finish.astype(np.float32) * FINISH_BONUS
        finish_rewarded |= newly_finish
        info["finish_rewarded"] = finish_rewarded

        # ============ 9. 庆祝完成奖励（一次性） ============
        celebration_completed = info["celebration_completed"]
        celebration_rewarded = info["celebration_rewarded"]
        newly_celebration = celebration_completed & ~celebration_rewarded
        reward += newly_celebration.astype(np.float32) * CELEBRATION_BONUS
        celebration_rewarded |= newly_celebration
        info["celebration_rewarded"] = celebration_rewarded

        # ============ 10. 红包靠近引导（密集，小量） ============
        for env_idx in range(n_envs):
            min_dist = np.inf
            # 贺礼红包
            for i in range(len(HELI_HONGBAO_POS)):
                if not heli_collected[env_idx, i]:
                    d = np.linalg.norm(robot_xy[env_idx] - HELI_HONGBAO_POS[i])
                    min_dist = min(min_dist, d)
            # 拜年红包
            for i in range(len(BAINIAN_HONGBAO_POS)):
                if not bainian_collected[env_idx, i]:
                    d = np.linalg.norm(robot_xy[env_idx] - BAINIAN_HONGBAO_POS[i])
                    min_dist = min(min_dist, d)
            if min_dist < 5.0:
                reward[env_idx] += 0.05 * np.exp(-min_dist / 2.0)

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

    # ==================== 终止条件 ====================

    def _compute_terminated(self, data: mtx.SceneData, info: dict,
                            root_pos: np.ndarray, root_quat: np.ndarray,
                            root_vel: np.ndarray) -> np.ndarray:
        """
        终止条件（满足任一则终止并施加重罚）：
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

        # 1. 基座接触地面
        try:
            base_contact_val = self._model.get_sensor_value("base_contact", data)
            if base_contact_val.ndim == 0:
                base_contact = np.array([base_contact_val > 0.01], dtype=bool)
            elif base_contact_val.shape[0] != n_envs:
                base_contact = np.full(n_envs, base_contact_val.flatten()[0] > 0.01, dtype=bool)
            else:
                base_contact = (base_contact_val > 0.01).flatten()[:n_envs]
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

    # ==================== 可视化辅助 ====================

    def _update_target_marker(self, data: mtx.SceneData, pose_commands: np.ndarray):
        """更新目标位置标记"""
        num_envs = data.shape[0]
        all_dof_pos = data.dof_pos.copy()

        for env_idx in range(num_envs):
            target_x = float(pose_commands[env_idx, 0])
            target_y = float(pose_commands[env_idx, 1])
            target_yaw = float(pose_commands[env_idx, 2]) if pose_commands.shape[1] > 2 else 0.0
            all_dof_pos[env_idx, self._target_marker_dof_start:self._target_marker_dof_end] = [
                target_x, target_y, target_yaw
            ]

        data.set_dof_pos(all_dof_pos, self._model)
        self._model.forward_kinematic(data)

    def _update_heading_arrows(self, data: mtx.SceneData, robot_pos: np.ndarray,
                               desired_vel_xy: np.ndarray, base_lin_vel_xy: np.ndarray):
        """更新箭头可视化"""
        if self._robot_arrow_body is None or self._desired_arrow_body is None:
            return

        num_envs = data.shape[0]
        arrow_offset = 0.5
        all_dof_pos = data.dof_pos.copy()

        for env_idx in range(num_envs):
            arrow_height = robot_pos[env_idx, 2] + arrow_offset

            # 当前运动方向箭头
            cur_v = base_lin_vel_xy[env_idx]
            cur_yaw = np.arctan2(cur_v[1], cur_v[0]) if np.linalg.norm(cur_v) > 1e-3 else 0.0
            robot_arrow_quat = self._euler_to_quat(0, 0, cur_yaw)
            qn = np.linalg.norm(robot_arrow_quat)
            robot_arrow_quat = robot_arrow_quat / qn if qn > 1e-6 else np.array([0, 0, 0, 1], dtype=np.float32)
            all_dof_pos[env_idx, self._robot_arrow_dof_start:self._robot_arrow_dof_end] = np.concatenate([
                np.array([robot_pos[env_idx, 0], robot_pos[env_idx, 1], arrow_height], dtype=np.float32),
                robot_arrow_quat
            ])

            # 期望运动方向箭头
            des_v = desired_vel_xy[env_idx]
            des_yaw = np.arctan2(des_v[1], des_v[0]) if np.linalg.norm(des_v) > 1e-3 else 0.0
            desired_arrow_quat = self._euler_to_quat(0, 0, des_yaw)
            qn = np.linalg.norm(desired_arrow_quat)
            desired_arrow_quat = desired_arrow_quat / qn if qn > 1e-6 else np.array([0, 0, 0, 1], dtype=np.float32)
            all_dof_pos[env_idx, self._desired_arrow_dof_start:self._desired_arrow_dof_end] = np.concatenate([
                np.array([robot_pos[env_idx, 0], robot_pos[env_idx, 1], arrow_height], dtype=np.float32),
                desired_arrow_quat
            ])

        data.set_dof_pos(all_dof_pos, self._model)
        self._model.forward_kinematic(data)

    # ==================== 重置 ====================

    def reset(self, data: mtx.SceneData, done: np.ndarray = None) -> tuple[np.ndarray, dict]:
        """
        重置环境

        关键要求：机器人初始位置必须随机分布在"2026"平台上（一票否决项）
        平台中心: (0, 10.33, 1.044)，半尺寸: X=5.0, Y=1.5
        """
        cfg: VBotSection012EnvCfg = self._cfg
        num_envs = data.shape[0]

        # ===== 在"2026"平台上随机生成位置 =====
        random_offset = np.random.uniform(
            low=self.spawn_range[:2],    # [x_min, y_min] 偏移
            high=self.spawn_range[2:],   # [x_max, y_max] 偏移
            size=(num_envs, 2)
        )
        robot_init_xy = self.spawn_center[:2] + random_offset
        terrain_heights = np.full(num_envs, self.spawn_center[2], dtype=np.float32)
        robot_init_xyz = np.column_stack([robot_init_xy, terrain_heights])

        # 随机初始朝向（大致朝向+Y方向 = 赛道前进方向）
        spawn_yaw = np.random.uniform(-np.pi / 6, np.pi / 6, num_envs)
        spawn_quat = np.array([self._euler_to_quat(0, 0, yaw) for yaw in spawn_yaw])

        dof_pos = np.tile(self._init_dof_pos, (num_envs, 1))
        dof_vel = np.tile(self._init_dof_vel, (num_envs, 1))

        # 设置base位置和朝向
        dof_pos[:, 3:6] = robot_init_xyz
        dof_pos[:, self._base_quat_start:self._base_quat_end] = spawn_quat

        # 归一化所有四元数
        for env_idx in range(num_envs):
            # base四元数
            quat = dof_pos[env_idx, self._base_quat_start:self._base_quat_end]
            qn = np.linalg.norm(quat)
            dof_pos[env_idx, self._base_quat_start:self._base_quat_end] = (
                quat / qn if qn > 1e-6 else np.array([0, 0, 0, 1], dtype=np.float32)
            )

            # 箭头四元数
            if self._robot_arrow_body is not None:
                for start, end in [
                    (self._robot_arrow_dof_start + 3, self._robot_arrow_dof_end),
                    (self._desired_arrow_dof_start + 3, self._desired_arrow_dof_end),
                ]:
                    if end <= len(dof_pos[env_idx]):
                        aq = dof_pos[env_idx, start:end]
                        aqn = np.linalg.norm(aq)
                        dof_pos[env_idx, start:end] = (
                            aq / aqn if aqn > 1e-6 else np.array([0, 0, 0, 1], dtype=np.float32)
                        )

        data.reset(self._model)
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)

        # ===== 构建初始信息 =====
        root_pos, root_quat, root_vel = self._extract_root_state(data)
        distance_to_finish = np.linalg.norm(root_pos[:, :2] - FINISH_ZONE_CENTER, axis=-1)

        info = {
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
            # 检查点（5个Y坐标里程碑）
            "checkpoints_reached": np.zeros((num_envs, len(CHECKPOINTS_Y)), dtype=bool),
            "checkpoints_rewarded": np.zeros((num_envs, len(CHECKPOINTS_Y)), dtype=bool),
            # 贺礼红包（5个，河床石头上）
            "heli_collected": np.zeros((num_envs, len(HELI_HONGBAO_POS)), dtype=bool),
            "heli_rewarded": np.zeros((num_envs, len(HELI_HONGBAO_POS)), dtype=bool),
            # 拜年红包（2个，吊桥区域）
            "bainian_collected": np.zeros((num_envs, len(BAINIAN_HONGBAO_POS)), dtype=bool),
            "bainian_rewarded": np.zeros((num_envs, len(BAINIAN_HONGBAO_POS)), dtype=bool),
            # 终点
            "finish_reached": np.zeros(num_envs, dtype=bool),
            "finish_rewarded": np.zeros(num_envs, dtype=bool),
            # 庆祝
            "celebration_start_time": np.full(num_envs, -1.0, dtype=np.float32),
            "celebration_completed": np.zeros(num_envs, dtype=bool),
            "celebration_rewarded": np.zeros(num_envs, dtype=bool),
            # 物理
            "last_dof_vel": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "contacts": np.zeros((num_envs, self.num_foot_check), dtype=np.bool_),
        }

        # 计算初始观测
        obs = self._compute_obs(
            data, root_pos, root_quat, root_vel,
            np.zeros((num_envs, self._num_action), dtype=np.float32),
            info,
        )

        # 更新可视化
        pose_commands_vis = np.tile(
            np.array([FINISH_ZONE_CENTER[0], FINISH_ZONE_CENTER[1], 0.0], dtype=np.float32),
            (num_envs, 1),
        )
        self._update_target_marker(data, pose_commands_vis)
        self._update_heading_arrows(
            data, root_pos,
            info.get("desired_vel_xy", np.zeros((num_envs, 2), dtype=np.float32)),
            root_vel[:, :2],
        )

        return obs, info
