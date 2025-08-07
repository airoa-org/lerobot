#!/home/openpi/.venv/bin/python3
import os
import sys
import json
import cv2

#!/usr/bin/env python3

from typing import List, Dict, Any
from collections import deque

import numpy as np

# ros関連
import rospy
from actionlib import SimpleActionClient
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import CompressedImage, JointState
from geometry_msgs.msg import Twist
from tmc_control_msgs.msg import GripperApplyEffortAction, GripperApplyEffortActionGoal
from hsr_data_msgs.srv import StringTrigger, StringTriggerResponse

import torch

# gr00t関連
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.dataset import ModalityConfig
from gr00t.experiment.data_config import DATA_CONFIG_MAP

from gr00t.model.policy import Gr00tPolicy

class HSREnv:
    """
    ROS経由でHSRロボットのセンサ情報の取得やアクションの実行を行う環境クラス．
    """

    GRIPPER_OPEN = 1
    GRIPPER_CLOSE = 0
    GRIPPER_CLOSE_THRESHOLD = 0.5 # グリッパーを閉じる閾値

    def __init__(self, update_freq=10):
        self.update_freq = update_freq
        self.rate = rospy.Rate(self.update_freq)

        # センサ情報の初期化
        self.head_rgb = None
        self.hand_rgb = None
        self.joint_state = None
        self.gripper_state = 0
        self.control_mode = None
        self.instruction = rospy.get_param("~instruction", "Grasp the apple.")

        self.joint_state_names: List[str] = [
            "arm_lift_joint",
            "arm_flex_joint",
            "arm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
            "hand_motor_joint",
            "head_pan_joint",
            "head_tilt_joint",
        ]

        self.arm_action_names: List[str] = [
            "arm_lift_joint",
            "arm_flex_joint",
            "arm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
        ]
        self.head_action_names: List[str] = ["head_pan_joint", "head_tilt_joint"]
        self.base_action_names: List[str] = ["base_x", "base_y", "base_theta"]

        # パブリッシャーの初期化
        self.arm_pub = rospy.Publisher("/hsrb/arm_trajectory_controller/command", JointTrajectory, queue_size=1)
        self.head_pub = rospy.Publisher("/hsrb/head_trajectory_controller/command", JointTrajectory, queue_size=1)
        self.gripper_pub = rospy.Publisher("/hsrb/gripper_controller/command", JointTrajectory, queue_size=1)
        self.base_pub = rospy.Publisher("/hsrb/command_velocity", Twist, queue_size=1)
        self.gripper_close_client = SimpleActionClient("/hsrb/gripper_controller/grasp", GripperApplyEffortAction)

        # サービス登録（instruction更新用）
        rospy.Service("/hsr_openpi/update_instruction", StringTrigger, self.update_instruction_srv)

        # サブスクライバーの初期化
        rospy.Subscriber(
            "/hsrb/head_rgbd_sensor/rgb/image_rect_color/compressed", CompressedImage, self.head_image_callback, queue_size=1
        )
        rospy.Subscriber("/hsrb/hand_camera/image_raw/compressed", CompressedImage, self.hand_image_callback, queue_size=1)
        rospy.Subscriber("/hsrb/joint_states", JointState, self.joint_state_callback, queue_size=1)
        rospy.Subscriber("/hsrb/gripper_controller/command", JointTrajectory, self.gripper_open_callback, queue_size=1)
        rospy.Subscriber(
            "/hsrb/gripper_controller/grasp/goal", GripperApplyEffortActionGoal, self.gripper_close_callback, queue_size=1
        )
        rospy.Subscriber("/control_mode", String, self.control_mode_callback, queue_size=1)

    def head_image_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)[:, :, :]  # bgr -> rgb
        self.head_rgb = np.array(image)

    def hand_image_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)[:, :, :]  # bgr -> rgb
        self.hand_rgb = np.array(image)

    def joint_state_callback(self, msg: JointState):
        joints = [msg.position[msg.name.index(name)] for name in self.joint_state_names]
        self.joint_state = np.asarray(joints, dtype=np.float32)

    def gripper_open_callback(self, msg: JointTrajectory):
        self.gripper_state = self.GRIPPER_OPEN

    def gripper_close_callback(self, msg: GripperApplyEffortActionGoal):
        self.gripper_state = self.GRIPPER_CLOSE

    def control_mode_callback(self, msg: String):
        self.control_mode = msg.data

    def update_instruction_srv(self, req: StringTrigger):
        self.instruction = req.message
        rospy.loginfo("Instruction updated: %s", self.instruction)
        return StringTriggerResponse(success=True)

    def reset_observation(self):
        """
        センサ情報をリセットする関数．
        """
        self.head_rgb = None
        self.hand_rgb = None
        self.joint_state = None

    def get_observations(self):
        """
        ロボットのセンサ情報をまとめた辞書を返す関数．
        全ての必要な情報がそろっていなければNone．
        """
        if self.head_rgb is None or self.hand_rgb is None or self.joint_state is None:
            return None
        return {
            "head_rgb": self.head_rgb,
            "hand_rgb": self.hand_rgb,
            "joint_state": self.joint_state,
            "instruction": self.instruction,
            "gripper_state": self.gripper_state,
            "control_mode": self.control_mode,
        }

    def execute_actions(self, action: np.ndarray) -> bool:
        """
        acitonをロボットに反映

        Parameters
        ----------
        action : np.ndarray
            ロボットに反映するアクション：
            [
                "arm_lift_joint",
                "arm_flex_joint",
                "arm_roll_joint",
                "wrist_flex_joint",
                "wrist_roll_joint",
                "hand_motor_joint",
                "head_pan_joint",
                "head_tilt_joint",
                "base_x",
                "base_y",
                "base_t",
            ]

        Returns
        -------
        bool
            実行できた場合はTrue, できなかった場合はFalse
        """
        # control_modeが"auto"の場合のみ実行
        if self.control_mode != "auto":
            return False  # 実行できない場合はFalseを返す

        # アーム制御
        arm_traj = JointTrajectory()
        arm_traj.joint_names = self.arm_action_names
        arm_point = JointTrajectoryPoint()
        arm_point.positions = action[:5]
        arm_point.velocities = []
        arm_point.time_from_start = rospy.Duration(1 / self.update_freq/2)
        arm_traj.points = [arm_point]

        # ヘッド制御
        head_traj = JointTrajectory()
        head_traj.joint_names = self.head_action_names
        arm_point = JointTrajectoryPoint()
        arm_point.positions = action[6:8]
        arm_point.velocities = []
        arm_point.time_from_start = rospy.Duration(1 / self.update_freq/2)

        head_traj.points = [arm_point]

        # ベース制御
        twist = Twist()
        twist.linear.x = action[8]
        twist.linear.y = action[9]
        twist.angular.z = action[10]

        # グリッパー制御
        gripper_action = None
        if action[5] < self.GRIPPER_CLOSE_THRESHOLD:  # グリッパーを閉じるかどうか 1: 閉じる, 0: 開く
            gripper_action = self.GRIPPER_CLOSE
        else:
            gripper_action = self.GRIPPER_OPEN

        if self.gripper_state != gripper_action:
            if gripper_action == self.GRIPPER_CLOSE:  # グリッパーを閉じる
                goal = GripperApplyEffortActionGoal()
                goal.goal.effort = -0.018
                self.gripper_close_client.send_goal(goal.goal)
            else:  # グリッパーを開く
                arm_traj = JointTrajectory()
                arm_traj.joint_names = ["hand_motor_joint"]
                arm_point = JointTrajectoryPoint()
                arm_point.positions = [1.239183768915874]
                arm_point.velocities = []
                arm_point.time_from_start = rospy.Duration(1)
                arm_traj.points = [arm_point]
                self.gripper_pub.publish(arm_traj)
            self.gripper_state = gripper_action

        self.arm_pub.publish(arm_traj)
        self.head_pub.publish(head_traj)
        self.base_pub.publish(twist)

        return True

    def sleep(self):
        self.rate.sleep()


class Gr00tHSRPolicy:
    """ Gr00tPolicyをHSRロボットに適用するためのクラスです．
    """
    def __init__(self,model_path: str = "/home/kohei/codes/matuolab/checkpoint-40000", adapted_action_chunks: int = 15):
        self.dagtconfig = data_config = DATA_CONFIG_MAP["hsr"]
        self.modality_config = data_config.modality_config()
        transforms = data_config.transform()
        self.policy = Gr00tPolicy(
            model_path=model_path,
            modality_config=self.modality_config,
            modality_transform=transforms,
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,  # HSRのembodiment tag
            device="cuda",
        )
        self.adopted_action_chunks = adapted_action_chunks
        self.action_queue = {
            "action.arm": deque(maxlen=self.adopted_action_chunks),
            "action.wrist": deque(maxlen=self.adopted_action_chunks),
            "action.hand": deque(maxlen=self.adopted_action_chunks),
            "action.head": deque(maxlen=self.adopted_action_chunks),
            "action.base": deque(maxlen=self.adopted_action_chunks),   
        }

    def act(self,obs):
        """
        obs: Dict[str, Any]
            センサ情報
            {
                "head_rgb": <np.ndarray shape (H, W, 3)>,
                "hand_rgb": <np.ndarray shape (H, W, 3)>,
                "joint_state": <np.ndarray shape (8,)>, # ["arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint","hand_motor_joint(gripper)", "head_pan_joint", "head_tilt_joint"]
                "instruction": <str>,
            }
        return: np.ndarray : shape (11,)
            アクション
            [
                "arm_lift_joint",
                "arm_flex_joint",
                "arm_roll_joint",
                "wrist_flex_joint",
                "wrist_roll_joint",
                "gripper",
                "head_pan_joint",
                "head_tilt_joint",
                "base_x",
                "base_y",
                "base_t",
            ]
        """

        if len(self.action_queue["action.arm"]) > 0:
            action_arm = self.action_queue["action.arm"].popleft()
            action_wrist = self.action_queue["action.wrist"].popleft()
            action_hand = self.action_queue["action.hand"].popleft()
            action_head = self.action_queue["action.head"].popleft()
            action_base = self.action_queue["action.base"].popleft()
            action = np.concatenate(
                [
                    action_arm,
                    action_wrist,
                    [action_hand],
                    action_head,
                    action_base,
                ]
            )
            action = action + np.concatenate(
                [obs["joint_state"][:5], np.array([0]), obs["joint_state"][6:8], np.array([0, 0, 0])]
            )
            return action
        print("=== Gr00tHSRPolicy: Getting action from policy ===")
        # image shapeは(480, 640, 3) → (1, 480, 640, 3)
        video_head = np.expand_dims(obs["head_rgb"], axis=0)
        video_hand = np.expand_dims(obs["hand_rgb"], axis=0)
        state_arm = np.expand_dims(obs["joint_state"][:3], axis=0)  # armの状態
        state_wrist = np.expand_dims(obs["joint_state"][3:5], axis=0)  # wristの状態
        state_hand = np.expand_dims(obs["joint_state"][5:6], axis=0)  # handの状態
        state_head = np.expand_dims(obs["joint_state"][6:8], axis=0)  # headの状態
        instruction = [obs["instruction"]]  # タスクの説明
        policy_input = {
            "video.head": video_head,
            "video.hand": video_hand,
            "state.arm" : state_arm,  # armの状態
            "state.wrist": state_wrist,  # wristの状態
            "state.hand": state_hand,  # handの状態
            "state.head": state_head,
            "annotation.human.task_description": instruction,  # タスクの説明
        }

        action_chunk = self.policy.get_action(policy_input)
        
        self.action_queue["action.arm"].extend(action_chunk["action.arm"][1:self.adopted_action_chunks])
        self.action_queue["action.wrist"].extend(action_chunk["action.wrist"][1:self.adopted_action_chunks])
        self.action_queue["action.hand"].extend(action_chunk["action.hand"][1:self.adopted_action_chunks])
        self.action_queue["action.head"].extend(action_chunk["action.head"][1:self.adopted_action_chunks])
        self.action_queue["action.base"].extend(action_chunk["action.base"][1:self.adopted_action_chunks])
        
        
        action_arm = action_chunk["action.arm"][0]  # 最初のアクションだけを使用
        action_wrist = action_chunk["action.wrist"][0] # 最初のアクションだけを使用
        action_hand = action_chunk["action.hand"][0]  # 最初のアクションだけを使用
        action_head = action_chunk["action.head"][0]  # 最初のアクションだけを使用
        action_base = action_chunk["action.base"][0]  # 最初のアクションだけを使用
        action = np.concatenate(
            [
                action_arm,
                action_wrist,
                [action_hand],
                action_head,
                action_base,
            ]
        )
        
        # 差分になっている行動を元に戻す
        action = action + np.concatenate(
            [obs["joint_state"][:5], np.array([0]), obs["joint_state"][6:8], np.array([0, 0, 0])]
        )
        print("=== Gr00tHSRPolicy: Action generated ===")
        print("action" , action)
        return action


def main():
    print("Start hsr_openpi")

    rospy.init_node("hsr_openpi")

    checkpoint_dir: str = rospy.get_param(
        "~checkpoint_dir", "/home/openpi/checkpoints/pi0_hsr_low_mem_finetune/hsr_tmc_new/5000"
    )
    adopted_action_chunks = rospy.get_param("~adopted_action_chunks", 1)
    update_freq: int = rospy.get_param("~update_freq", 5)

    rospy.loginfo("checkpoint_dir: %s", checkpoint_dir)
    rospy.loginfo("adopted_action_chunks: %s", adopted_action_chunks)
    rospy.loginfo("update_freq: %s", update_freq)

    env = HSREnv(update_freq=update_freq)
    policy = Gr00tHSRPolicy(model_path=checkpoint_dir)

    while not rospy.is_shutdown():
        obs = env.get_observations()
        if obs is not None:
            action = policy.act(obs)
            is_executed = env.execute_actions(action)
            
            # # テストで画像を出力
            # cv2.imwrite("/root/catkin_ws/head_rgb.png", obs["head_rgb"])
            # cv2.imwrite("/root/catkin_ws/hand_rgb.png", obs["hand_rgb"])
            # # cv2.imshow("hand_rgb", obs["hand_rgb"])
            # break
            
            if is_executed:
                rospy.loginfo("Action executed.")
            else:
                rospy.loginfo("Action not executed.")
            rospy.loginfo("Language instruction: %s", obs["instruction"])
            rospy.loginfo("Action: %s", action)
            env.reset_observation()
        else:
            rospy.loginfo("Observations are not ready.")
        env.sleep()


if __name__ == "__main__":
    main()
