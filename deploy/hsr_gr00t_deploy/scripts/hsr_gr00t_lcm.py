#!/home/openpi/.venv/bin/python3
from collections import deque

#!/usr/bin/env python3
from typing import Any

import cv2
import numpy as np

# gr00t関連
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.dataset import ModalityConfig
from gr00t.experiment.data_config import DATA_CONFIG_MAP

from gr00t.model.policy import Gr00tPolicy

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


class Gr00tHSRPolicy:
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
        self.adopted_action_chunks = adopted_action_chunks
        self.action_queue = {
            "action.arm": deque(maxlen=self.adopted_action_chunks),
            "action.hand": deque(maxlen=self.adopted_action_chunks),
            "action.head": deque(maxlen=self.adopted_action_chunks),
            "action.base": deque(maxlen=self.adopted_action_chunks),   
        }
        self.num_traj = num_traj
        rand_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        policy_input = {
            "head_rgb": rand_img,
            "hand_rgb": rand_img,
            "state.arm": np.array([0.0 for _ in range(5)]),
            "state.hand": np.array([0.0 for _ in range(1)]),
            "state.head": np.array([0.0 for _ in range(2)]),
            "prompt": "Test prompt. Do not move.",
        }
        self.reset_buffer()
    
    def reset_buffer(self):
        self.action_queue.clear()

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

        if len(self.action_queue["action.arm"]) >= self.num_traj:
            actions = []
            for _ in range(self.num_traj):
                action_arm = self.action_queue["action.arm"].popleft()
                action_hand = self.action_queue["action.hand"].popleft()
                action_head = self.action_queue["action.head"].popleft()
                action_base = self.action_queue["action.base"].popleft()
                action = np.concatenate(
                    [
                        action_arm,
                        [action_hand],
                        action_head,
                        action_base,
                    ]
                )
                action = action + np.concatenate(
                    [obs["joint_state"][:5], np.array([0]), obs["joint_state"][6:8], np.array([0, 0, 0])]
                )
                actions.append(action)
            return np.stack(actions)

        print("=== Gr00tHSRPolicy: Getting action from policy ===")
        # image shapeは(480, 640, 3) → (1, 480, 640, 3)
        video_head = np.expand_dims(obs["head_rgb"], axis=0)
        video_hand = np.expand_dims(obs["hand_rgb"], axis=0)
        state_arm = np.expand_dims(obs["joint_state"][:5], axis=0)  # armの状態
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
        
        self.action_queue["action.arm"].extend(action_chunk["action.arm"][self.num_traj:self.adopted_action_chunks])
        self.action_queue["action.hand"].extend(action_chunk["action.hand"][self.num_traj:self.adopted_action_chunks])
        self.action_queue["action.head"].extend(action_chunk["action.head"][self.num_traj:self.adopted_action_chunks])
        self.action_queue["action.base"].extend(action_chunk["action.base"][self.num_traj:self.adopted_action_chunks])
        
        actions = []
        for i in range(self.num_traj):
            action_arm = action_chunk["action.arm"][i]  # 最初のアクションだけを使用
            action_hand = action_chunk["action.hand"][i]  # 最初のアクションだけを使用
            action_head = action_chunk["action.head"][i]  # 最初のアクションだけを使用
            action_base = action_chunk["action.base"][i]  # 最初のアクションだけを使用
            action = np.concatenate(
                [
                    action_arm,
                    [action_hand],
                    action_head,
                    action_base,
                ]
            )


        
            # 差分になっている行動を元に戻す
            action = action + np.concatenate(
                [obs["joint_state"][:5], np.array([0]), obs["joint_state"][6:8], np.array([0, 0, 0])]
            )
            actions.append(action)
        return np.array(actions)


def main():
    print("Start Issac-GR00T")

    # TODO: 引数でいい感じに処理するようにする
    checkpoint_dir = "/home/veluga-g3/airoa/ckpts"
    adopted_action_chunks = 15

    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"adopted_action_chunks: {adopted_action_chunks}")

    policy = Gr00tHSRPolicy(model_path=checkpoint_dir,adopted_action_chunks=adopted_action_chunks)
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
