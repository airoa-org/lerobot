#!/home/openpi/.venv/bin/python3
from collections import deque

#!/usr/bin/env python3
from typing import Any

import cv2
import numpy as np

# gr00t関連
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

import lcm
from lcm_msgs import RequestMsg, ResponseMsg


def compressedimage_to_array_lcm(msg):
    # Convert signed bytes (-128~127) to unsigned (0~255)
    signed = np.array(msg.data[: msg.data_length], dtype=np.int8)
    unsigned = signed.astype(np.uint8)

    # TODO: 元コードにバグがあり，rgbに変換されずに使われていたのでそのままにしてある
    # 学習はバグありのままされていた？
    # そうならそのまま使って，そうでないなら以下の正しいコードを使う
    image = cv2.imdecode(unsigned, cv2.IMREAD_COLOR)[:, :, :]  # bgr -> rgb
    # image = cv2.imdecode(unsigned, cv2.IMREAD_COLOR)[:, :, ::-1]  # bgr -> rgb
    return image


class HSRLcmServer:
    GRIPPER_OPEN = 1
    GRIPPER_CLOSE = 0
    GRIPPER_CLOSE_THRESHOLD = 0.5  # グリッパーを閉じる閾値

    def __init__(self, policy, traj_hz=10.0):
        self.traj_hz = float(traj_hz)

        self._lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=1")
        self._lc.subscribe("REQUEST_CHANNEL", self._handle_request)

        self.policy = policy

        self.joint_state_names: list[str] = [
            "arm_lift_joint",
            "arm_flex_joint",
            "arm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
            "hand_motor_joint",
            "head_pan_joint",
            "head_tilt_joint",
        ]

    def handle(self):
        self._lc.handle()

    def _handle_request(self, channel, data):
        request = RequestMsg.decode(data)
        head_rgb = compressedimage_to_array_lcm(request.head_rgb)
        hand_rgb = compressedimage_to_array_lcm(request.hand_rgb)

        try:
            # self.joint_state_namesの順に並び替え
            joint_msg = request.joint_state
            name_list = joint_msg.name[: joint_msg.num_joints]
            position_list = joint_msg.position[: joint_msg.num_joints]
            joint_state = [position_list[name_list.index(name)] for name in self.joint_state_names]

            observation = {
                "head_rgb": head_rgb,
                "hand_rgb": hand_rgb,
                "joint_state": np.array(joint_state, dtype=np.float32),
                "instruction": request.instruction,
            }

            action = self.policy.act(observation)

            action_joint_names = self.joint_state_names + [
                "base_x",
                "base_y",
                "base_t",
            ]

            if action.shape[1] != len(action_joint_names):
                raise RuntimeError(f"Expected (T, {len(action_joint_names)}), got {action.shape}")

        except Exception as e:
            print(f"Error processing data: {e}")
            # 現在位置 + 台車速度なし（仮）で対応
            action = np.array([joint_state + [0.0, 0.0, 0.0]])

        rows, cols = action.shape
        response = ResponseMsg()
        response.hz = self.traj_hz
        response.num_joints = cols
        response.joint_names = action_joint_names
        response.rows = rows
        response.cols = cols
        response.result = action.tolist()

        self._lc.publish("ACTION_RESPONSE", response.encode())


class SmolvlaHSRPolicy:
    """
    Gr00tPolicyをHSRロボットに適用するためのクラスです.
    HSREnvからセンサ情報を取得し, policyの計算後にアクションを環境に反映させます.
    """

    def __init__(
        self,
        model_path: str = "/home/kohei/codes/matuolab/checkpoint-40000", 
        adopted_action_chunks: int = 15,
        num_traj: int = 1
    ):
        assert num_traj <= adopted_action_chunks, "num_traj must be <= adopted_action_chunks"
        # self.dagtconfig = data_config = DATA_CONFIG_MAP["hsr"]
        # self.modality_config = data_config.modality_config()
        # transforms = data_config.transform()
        # self.policy = SmolVLAPolicy(
        #     model_path=model_path,
        #     modality_config=self.modality_config,
        #     modality_transform=transforms,
        #     embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,  # HSRのembodiment tag
        #     device="cuda",
        # )
        self.policy = SmolVLAPolicy.from_pretrained(
            pretrained_name_or_path = './pretrained_model',
        )
        self.num_traj = num_traj
        policy_input = {
            'observation.image.head' : torch.zeros(1, 3, 480, 640, device=device, dtype=torch.float32), # uint8 / 255
            'observation.image.hand' : torch.zeros(1, 3, 480, 640, device=device, dtype=torch.float32), # 
            'observation.state'      : torch.zeros(1, 8          , device=device, dtype=torch.float32),
            'task' : ['test'],
        }
        self.reset_buffer()
    
    def reset_buffer(self):
        self.policy.reset()

    def act(self, obs: dict[str, Any]) -> np.ndarray:
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

        print("=== SmolvlaHSRPolicy: Getting action from policy ===")
        # image shapeは(480, 640, 3) → (1, 480, 640, 3)
        video_head = np.expand_dims(obs["head_rgb"], axis=0)
        video_hand = np.expand_dims(obs["hand_rgb"], axis=0)
        state = np.expand_dims(obs["joint_state"], axis=0)  # armの状態
        instruction = [obs["instruction"]]  # タスクの説明
        policy_input = {
            'observation.image.head': video_head,
            'observation.image.hand': video_hand,
            'observation.state' : state, 
            'task': instruction,  # タスクの説明
        }

        action = self.policy.select_action(policy_input)

        action = torch.squeeze(action)

        policy_action = action.cpu().detach().numpy().astype(np.float64)

        
        actions = []
        action = np.concatenate(
            [
                policy_action[0:5],
                policy_action[5],
                policy_action[7:9],
                policy_action[9:12],
            ]
        )

        # 差分になっている行動を元に戻す
        action = action + np.concatenate(
            [obs["joint_state"][:5], np.array([0]), obs["joint_state"][6:8], np.array([0, 0, 0])]
        )
        actions.append(action)
        return np.array(actions)


def main():
    print("Start Smolvla")

    # TODO: 引数でいい感じに処理するようにする
    checkpoint_dir = "/home/veluga-g3/airoa/ckpts"
    adopted_action_chunks = 15

    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"adopted_action_chunks: {adopted_action_chunks}")

    policy = SmolvlaHSRPolicy(model_path=checkpoint_dir,adopted_action_chunks=adopted_action_chunks)
    lcm_hsr_server = HSRLcmServer(policy)

    print("start server...")
    try:
        while True:
            lcm_hsr_server.handle()
    except KeyboardInterrupt:
        pass

    return


if __name__ == "__main__":
    main()
