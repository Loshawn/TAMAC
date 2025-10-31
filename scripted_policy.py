import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from constants import SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env

import IPython
e = IPython.embed


class BasePolicy:
    def __init__(self, inject_noise=False):
        # 构造函数：初始化政策类的状态。
        # inject_noise: 是否在位置生成过程中注入噪声（默认不注入噪声）
        self.inject_noise = inject_noise
        self.step_count = 0  # 当前的时间步数，从 0 开始
        self.left_trajectory = None  # 左臂的轨迹
        self.right_trajectory = None  # 右臂的轨迹

    def generate_trajectory(self, ts_first):
        # 抽象方法：生成左、右机械臂的轨迹。
        # 这个方法在子类中被实现，根据任务要求生成具体的轨迹。
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        """
        该方法用于插值计算当前位置（xyz，姿态quat和夹持器状态gripper）。
        它根据当前和下一个轨迹点以及当前时间步 t，计算当前位置。

        参数:
        curr_waypoint: 当前轨迹点的字典，包含 't'（时间戳），'xyz'（位置），'quat'（姿态四元数），'gripper'（夹持器状态）
        next_waypoint: 下一个轨迹点的字典，包含相同的信息
        t: 当前时间步（或时间戳），用于确定当前位置

        返回:
        插值后的 xyz, quat 和 gripper 状态
        """
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        
        # 计算位置、姿态和夹持器的插值
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        
        return xyz, quat, gripper

    def __call__(self, ts):
        """
        该方法在每一个时间步都会被调用，执行轨迹生成、插值和噪声注入等操作，
        最后返回左右机械臂的控制动作。

        参数:
        ts: 当前时间步的状态，通常包含机器人当前的状态信息。

        返回:
        当前时间步的控制动作，包含左右机械臂的位姿、姿态和夹持器状态
        """
        # 在第一时间步生成轨迹（只生成一次，之后通过插值获得每一步的动作）
        if self.step_count == 0:
            self.generate_trajectory(ts)

        # 获取当前时间步左机械臂和右机械臂的轨迹点
        if self.left_trajectory[0]['t'] == self.step_count:
            # 如果左臂轨迹中当前时间步的点是第一个，取出这个点
            self.curr_left_waypoint = self.left_trajectory.pop(0)
        next_left_waypoint = self.left_trajectory[0]  # 获取下一个左臂轨迹点

        if self.right_trajectory[0]['t'] == self.step_count:
            # 如果右臂轨迹中当前时间步的点是第一个，取出这个点
            self.curr_right_waypoint = self.right_trajectory.pop(0)
        next_right_waypoint = self.right_trajectory[0]  # 获取下一个右臂轨迹点

        # 插值计算当前位置和姿态
        left_xyz, left_quat, left_gripper = self.interpolate(self.curr_left_waypoint, next_left_waypoint, self.step_count)
        right_xyz, right_quat, right_gripper = self.interpolate(self.curr_right_waypoint, next_right_waypoint, self.step_count)

        # 注入噪声：如果 inject_noise 为 True，则为位置增加一些随机噪声
        if self.inject_noise:
            scale = 0.01  # 噪声的幅度
            left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)  # 为左臂位置增加噪声
            right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)  # 为右臂位置增加噪声

        # 组合左右臂的动作信息
        action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])  # 左臂的动作信息
        action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])  # 右臂的动作信息

        self.step_count += 1  # 增加时间步计数
        return np.concatenate([action_left, action_right])  # 返回拼接后的左右臂控制动作


class PickAndTransferPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        box_info = np.array(ts_first.observation['env_state'])
        box_xyz = box_info[:3]
        box_quat = box_info[3:]
        # print(f"Generate trajectory for {box_xyz=}")

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)

        meet_xyz = np.array([0, 0.5, 0.25])

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 100, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1}, # approach meet position
            {"t": 260, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1}, # move to meet position
            {"t": 310, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 0}, # close gripper
            {"t": 360, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # move left
            {"t": 400, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # stay
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 90, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 1}, # approach the cube
            {"t": 130, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 1}, # go down
            {"t": 170, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 0}, # close gripper
            {"t": 200, "xyz": meet_xyz + np.array([0.05, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 0}, # approach meet position
            {"t": 220, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 0}, # move to meet position
            {"t": 310, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 1}, # open gripper
            {"t": 360, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1}, # move to right
            {"t": 400, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1}, # stay
        ]


class InsertionPolicy(BasePolicy):
    """
    InsertionPolicy 类继承自 BasePolicy，用于生成一个用于机械臂插入任务的轨迹。
    这个轨迹包含左右臂的位置、姿态和夹持器控制，模拟机械臂完成插入任务。
    """
    
    def generate_trajectory(self, ts_first):
        """
        生成插入任务的轨迹。
        根据任务的初始状态，生成左右机械臂的轨迹，执行插入任务。

        参数:
        ts_first: 当前时间步的状态，通常包含环境和机械臂的初始状态信息。

        返回:
        None
        """
        # 获取初始的左右机械臂的位姿（位置 + 姿态四元数）
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        # 获取插入任务的环境状态，提取标靶物体（peg）和插槽（socket）的位置信息
        peg_info = np.array(ts_first.observation['env_state'])[:7]  # 取出前7个数据：peg的位置和姿态
        peg_xyz = peg_info[:3]  # 获取peg物体的位置
        peg_quat = peg_info[3:]  # 获取peg物体的姿态（四元数）

        socket_info = np.array(ts_first.observation['env_state'])[7:]  # 取出后面数据：socket的位置和姿态
        socket_xyz = socket_info[:3]  # 获取socket位置
        socket_quat = socket_info[3:]  # 获取socket姿态（四元数）

        # 计算机械臂夹持器的目标姿态，针对右臂和左臂分别定义夹持器的姿态
        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])  # 右臂夹持器的初始姿态
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)  # 旋转60度

        gripper_pick_quat_left = Quaternion(init_mocap_pose_left[3:])  # 左臂夹持器的初始姿态
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)  # 旋转60度

        # 设定插入任务的目标位置和右臂的提升高度
        meet_xyz = np.array([0, 0.5, 0.15])  # 两个机械臂会合的位置（目标位置）
        lift_right = 0.00715  # 右臂的提升高度

        # 定义左机械臂的轨迹
        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # 初始位置（休眠）
            {"t": 120, "xyz": socket_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # 靠近插槽
            {"t": 170, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # 下降到插槽
            {"t": 220, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # 夹持物体
            {"t": 285, "xyz": meet_xyz + np.array([-0.1, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # 靠近会合位置
            {"t": 340, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements,"gripper": 0},  # 插入物体
            {"t": 400, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0},  # 完成插入
        ]

        # 定义右机械臂的轨迹
        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # 初始位置（休眠）
            {"t": 120, "xyz": peg_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # 靠近peg物体
            {"t": 170, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # 下降到peg物体
            {"t": 220, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # 夹持物体
            {"t": 285, "xyz": meet_xyz + np.array([0.1, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # 靠近会合位置
            {"t": 340, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # 插入物体
            {"t": 400, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # 完成插入
        ]

class InsertionRevPolicy(BasePolicy):
    """
    InsertionPolicy 类继承自 BasePolicy，用于生成一个用于机械臂插入任务的轨迹。
    这个轨迹包含左右臂的位置、姿态和夹持器控制，模拟机械臂完成插入任务。
    """
    
    def generate_trajectory(self, ts_first):
        """
        生成插入任务的轨迹。
        根据任务的初始状态，生成左右机械臂的轨迹，执行插入任务。

        参数:
        ts_first: 当前时间步的状态，通常包含环境和机械臂的初始状态信息。

        返回:
        None
        """
        # 获取初始的左右机械臂的位姿（位置 + 姿态四元数）
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        # 获取插入任务的环境状态，提取标靶物体（peg）和插槽（socket）的位置信息
        peg_info = np.array(ts_first.observation['env_state'])[:7]  # 取出前7个数据：peg的位置和姿态
        peg_xyz = peg_info[:3]  # 获取peg物体的位置
        peg_quat = peg_info[3:]  # 获取peg物体的姿态（四元数）

        socket_info = np.array(ts_first.observation['env_state'])[7:]  # 取出后面数据：socket的位置和姿态
        socket_xyz = socket_info[:3]  # 获取socket位置
        socket_quat = socket_info[3:]  # 获取socket姿态（四元数）

        # 计算机械臂夹持器的目标姿态，针对右臂和左臂分别定义夹持器的姿态
        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])  # 右臂夹持器的初始姿态
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)  # 旋转60度

        gripper_pick_quat_left = Quaternion(init_mocap_pose_left[3:])  # 左臂夹持器的初始姿态
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)  # 旋转60度

        # 设定插入任务的目标位置和右臂的提升高度
        meet_xyz = np.array([0, 0.5, 0.15])  # 两个机械臂会合的位置（目标位置）
        lift_right = 0.012  # 右臂的提升高度

        # 定义左机械臂的轨迹
        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # 初始位置（休眠）
            {"t": 70, "xyz": socket_xyz + np.array([0, 0, 0.10]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # 靠近插槽
            {"t": 140, "xyz": socket_xyz + np.array([0, 0, 0.02]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # 下降到插槽
            {"t": 180, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # 下降到插槽
            {"t": 210, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # 夹持物体
            {"t": 220, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # 夹持物体
            {"t": 230, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # 夹持物体

            {"t": 280, "xyz": meet_xyz + np.array([-0.1, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # 靠近会合位置
            {"t": 320, "xyz": meet_xyz + np.array([-0.08, 0, 0]), "quat": gripper_pick_quat_left.elements,"gripper": 0},  # 插入物体
            {"t": 360, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements,"gripper": 0},  # 插入物体
            {"t": 400, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0},  # 完成插入
        ]

        # 定义右机械臂的轨迹
        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # 初始位置（休眠）
            {"t": 100, "xyz": peg_xyz + np.array([0.01, 0, 0.10]), "quat": gripper_pick_quat_right.elements, "gripper": 0.7}, # 靠近peg物体
            {"t": 140, "xyz": peg_xyz + np.array([0.01, 0, 0.02]), "quat": gripper_pick_quat_right.elements, "gripper": 0.7}, # 下降到peg物体
            {"t": 180, "xyz": peg_xyz + np.array([0.01, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 0.7}, # 夹持物体
            {"t": 210, "xyz": peg_xyz + np.array([0.01, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 0.7}, # 夹持物体
            {"t": 220, "xyz": peg_xyz + np.array([0.01, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # 夹持物体
            {"t": 230, "xyz": peg_xyz + np.array([0.01, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # 夹持物体

            {"t": 280, "xyz": meet_xyz + np.array([0.1, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # 靠近会合位置
            {"t": 320, "xyz": meet_xyz + np.array([0.08, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # 插入物体
            {"t": 360, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements,"gripper": 0},  # 插入物体
            {"t": 400, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # 完成插入
        ]


class PickAndStackPolicy(BasePolicy):
    def generate_trajectory(self, ts_first):
        """
        该策略用于抓取两个小物体并堆叠起来
        ts_first：初始时间步的数据
        """
        # 获取机械臂和物体信息
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        # 随机获取两个物体的位置信息
        object_info = np.array(ts_first.observation['env_state'])[:7]  # 物体1
        object_xyz_1 = object_info[:3]
        object_quat_1 = object_info[3:]
        
        object_info_2 = np.array(ts_first.observation['env_state'])[7:]  # 物体2
        object_xyz_2 = object_info_2[:3]
        object_quat_2 = object_info_2[3:]

        # 计算抓取物体的姿态
        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        gripper_pick_quat_left = Quaternion(init_mocap_pose_left[3:])
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)

        # 目标位置（中央位置）
        central_position = np.array([0, 0.5, 0.05])  # 放置物体的中心位置

        # 轨迹：左机械臂抓取物体1，右机械臂抓取物体2，堆叠物体
        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # 初始位置（休眠）
            {"t": 90, "xyz": object_xyz_2 + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # 靠近插槽
            {"t": 150, "xyz": object_xyz_2 + np.array([0, 0, -0.015]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # 下降到插槽
            {"t": 160, "xyz": object_xyz_2 + np.array([0, 0, -0.015]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # 夹持物体

            {"t": 210, "xyz": central_position + np.array([0, 0, 0.1]), "quat": gripper_pick_quat_left.elements,"gripper": 0},  # 插入物体
            {"t": 260, "xyz": central_position, "quat": gripper_pick_quat_left.elements,"gripper": 0},
            {"t": 270, "xyz": central_position, "quat": gripper_pick_quat_left.elements,"gripper": 1},

            {"t": 300, "xyz": central_position + np.array([-0.15, 0, 0.15]), "quat": gripper_pick_quat_left.elements,"gripper": 1},
            {"t": 400, "xyz": central_position + np.array([-0.15, 0, 0.15]), "quat": gripper_pick_quat_left.elements, "gripper": 1},  # 完成插入
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # 初始位置（休眠）
            {"t": 90, "xyz": object_xyz_1 + np.array([0, 0, 0.12]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # 靠近插槽
            {"t": 130, "xyz": object_xyz_1 + np.array([0, 0, -0.015]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # 下降到插槽
            {"t": 170, "xyz": object_xyz_1 + np.array([0, 0, -0.015]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # 夹持物体
            
            {"t": 200, "xyz": central_position + np.array([0.15, 0, 0.1]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # 靠近会合位置
            {"t": 270, "xyz": central_position + np.array([0.15, 0, 0.1]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # 靠近会合位置
            
            {"t": 320, "xyz": central_position + np.array([0, 0, 0.1]), "quat": gripper_pick_quat_right.elements,"gripper": 0},  # 插入物体
            {"t": 360, "xyz": central_position + np.array([0, 0, 0.05]) , "quat": gripper_pick_quat_right.elements,"gripper": 0},  # 插入物体
            {"t": 370, "xyz": central_position + np.array([0, 0, 0.05]) , "quat": gripper_pick_quat_right.elements,"gripper": 1},  # 插入物体
            
            {"t": 400, "xyz": central_position + np.array([0.15, 0, 0.15]), "quat": gripper_pick_quat_right.elements, "gripper": 1},  # 完成插入
        ]

class StoragePolicy(BasePolicy):
    def generate_trajectory(self, ts_first):
        """
        该策略用于抓取两个小物体并堆叠起来
        ts_first：初始时间步的数据
        """
        # 获取机械臂和物体信息
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        # 随机获取两个物体的位置信息
        object_info = np.array(ts_first.observation['env_state'])[:7]  # 物体1
        object_xyz_1 = object_info[:3]
        object_quat_1 = object_info[3:]
        
        object_info_2 = np.array(ts_first.observation['env_state'])[7:]  # 物体2
        object_xyz_2 = object_info_2[:3]
        object_quat_2 = object_info_2[3:]

        # 计算抓取物体的姿态
        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        gripper_pick_quat_left = Quaternion(init_mocap_pose_left[3:])
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)

        # 目标位置（中央位置）
        central_position = np.array([0, 0.5, 0.1])  # 放置物体的中心位置

        # 轨迹：左机械臂抓取物体1，右机械臂抓取物体2，堆叠物体
        # self.left_trajectory = [
        #     {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # 初始位置（休眠）
        #     {"t": 90, "xyz": object_xyz_2 + np.array([0, 0.06, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # 靠近插槽
        #     {"t": 150, "xyz": object_xyz_2 + np.array([0, 0.06, -0.015]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # 下降到插槽
        #     {"t": 170, "xyz": object_xyz_2 + np.array([0, 0.06, -0.015]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # 夹持物体

        #     {"t": 210, "xyz": central_position + np.array([0, 0.06, 0.15]), "quat": gripper_pick_quat_left.elements,"gripper": 0},  # 插入物体
        #     {"t": 260, "xyz": central_position + np.array([0, 0.06, 0.05]), "quat": gripper_pick_quat_left.elements,"gripper": 0},
        #     {"t": 270, "xyz": central_position + np.array([0, 0.06, 0.05]), "quat": gripper_pick_quat_left.elements,"gripper": 1},

        #     {"t": 300, "xyz": central_position + np.array([-0.15, 0, 0.15]), "quat": gripper_pick_quat_left.elements,"gripper": 1},
        #     {"t": 400, "xyz": central_position + np.array([-0.15, 0, 0.15]), "quat": gripper_pick_quat_left.elements, "gripper": 1},  # 完成插入
        # ]

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # 初始位置（休眠）
            {"t": 90, "xyz": object_xyz_2 + np.array([0, -0.06, 0.15]), "quat": gripper_pick_quat_left.elements, "gripper": 0.7}, # 靠近插槽
            {"t": 150, "xyz": object_xyz_2 + np.array([0, -0.06, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0.7}, # 下降到插槽
            {"t": 170, "xyz": object_xyz_2 + np.array([0, -0.06, 0.05]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # 夹持物体

            {"t": 230, "xyz": central_position + np.array([0, -0.06, 0.05]), "quat": gripper_pick_quat_left.elements,"gripper": 0},  # 插入物体
            {"t": 260, "xyz": central_position+ np.array([0, -0.06, 0]), "quat": gripper_pick_quat_left.elements,"gripper": 0},
            {"t": 270, "xyz": central_position+ np.array([0, -0.06, 0]), "quat": gripper_pick_quat_left.elements,"gripper": 1},

            {"t": 300, "xyz": central_position + np.array([-0.15, 0, 0.15]), "quat": gripper_pick_quat_left.elements,"gripper": 1},
            {"t": 400, "xyz": central_position + np.array([-0.15, 0, 0.15]), "quat": gripper_pick_quat_left.elements, "gripper": 1},  # 完成插入
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # 初始位置（休眠）
            {"t": 90, "xyz": object_xyz_1 + np.array([0, 0, 0.12]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # 靠近插槽
            {"t": 130, "xyz": object_xyz_1 + np.array([0, 0, -0.015]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # 下降到插槽
            {"t": 170, "xyz": object_xyz_1 + np.array([0, 0, -0.015]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # 夹持物体
            
            {"t": 200, "xyz": central_position + np.array([0.15, 0, 0.1]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # 靠近会合位置
            {"t": 270, "xyz": central_position + np.array([0.15, 0, 0.1]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # 靠近会合位置
            
            {"t": 320, "xyz": central_position + np.array([0, 0, 0.05]), "quat": gripper_pick_quat_right.elements,"gripper": 0},  # 插入物体
            {"t": 360, "xyz": central_position + np.array([0, 0, 0.02]) , "quat": gripper_pick_quat_right.elements,"gripper": 0},  # 插入物体
            {"t": 370, "xyz": central_position + np.array([0, 0, 0.02]) , "quat": gripper_pick_quat_right.elements,"gripper": 1},  # 插入物体
            
            {"t": 400, "xyz": central_position + np.array([0.15, 0, 0.15]), "quat": gripper_pick_quat_right.elements, "gripper": 1},  # 完成插入
        ]

def test_policy(task_name):
    # example rolling out pick_and_transfer policy
    onscreen_render = True
    inject_noise = False

    # setup the environment
    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    if 'sim_transfer_cube' in task_name:
        env = make_ee_sim_env('sim_transfer_cube')
    elif 'sim_insertion' in task_name:
        env = make_ee_sim_env('sim_insertion')
    else:
        raise NotImplementedError

    for episode_idx in range(2):
        ts = env.reset()
        episode = [ts]
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images']['angle'])
            plt.ion()

        policy = PickAndTransferPolicy(inject_noise)
        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images']['angle'])
                plt.pause(0.02)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        if episode_return > 0:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")


if __name__ == '__main__':
    test_task_name = 'sim_transfer_cube_scripted'
    test_policy(test_task_name)

