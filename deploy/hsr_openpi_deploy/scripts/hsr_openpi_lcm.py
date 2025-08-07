#!/home/openpi/.venv/bin/python3
from collections import deque

#!/usr/bin/env python3
from typing import Any

import cv2
import numpy as np

# openpi関連
import torch

from openpi.policies import policy_config
from openpi.training import config

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


class OpenpiPolicy:
    """
    PiZeroというpolicyの推論・実行を担当するクラスです.
    HSREnvからセンサ情報を取得し, policyの計算後にアクションを環境に反映させます.
    """

    def __init__(
        self,
        config_name: str,
        checkpoint_dir: str,
        adopted_action_chunks: int = 15,  # 一度の推論で得られるaction_chunkのうち、最初何個を使うか
        num_traj: int = 1,  # 返すアクションの次元
    ):
        assert num_traj <= adopted_action_chunks, "num_traj must be <= adopted_action_chunks"
        # openpiのpolicyのロード
        self.config: config.TrainConfig = config.get_config(config_name)
        self.policy = policy_config.create_trained_policy(self.config, checkpoint_dir)

        self.adopted_action_chunks: int = adopted_action_chunks
        self.action_queue: deque = deque(maxlen=adopted_action_chunks)
        self.num_traj = num_traj

        # 呼ばれたときすぐ使えるよう，policyを一回動かしておく
        rand_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        policy_input = {
            "head_rgb": rand_img,
            "hand_rgb": rand_img,
            "state": np.array([0.0 for _ in range(8)]),
            "prompt": "Test prompt. Do not move.",
        }

        _ = self.policy.infer(policy_input)["actions"]
        # dequeも空にしておく
        self.reset_buffer()

    def reset_buffer(self):
        self.action_queue.clear()

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        """
        センサ情報を受け取り、アクションを返す関数
        obs: Dict[str, Any]
            センサ情報
            {
                "head_rgb": <np.ndarray shape (H, W, 3)>,
                "hand_rgb": <np.ndarray shape (H, W, 3)>,
                "joint_state": <np.ndarray shape (8,)>, # ["arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint","hand_motor_joint(gripper)", "head_pan_joint", "head_tilt_joint"]
                "instruction": <str>,
            }
        return: np.ndarray : shape (self.num_traj, 11)
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

        if len(self.action_queue) >= self.num_traj:
            actions = []
            for _ in range(self.num_traj):
                action = self.action_queue.popleft()
                base_action = np.concatenate(
                    [obs["joint_state"][:5], np.array([0]), obs["joint_state"][6:8], np.array([0, 0, 0])]
                )
                actions.append(action + base_action)
            return np.stack(actions)

        # Policy への入力辞書を作成
        policy_input = {
            "head_rgb": obs["head_rgb"],
            "hand_rgb": obs["hand_rgb"],
            "state": obs["joint_state"],
            "prompt": obs["instruction"],
        }

        action_chunk = self.policy.infer(policy_input)["actions"]
        self.action_queue.extend(action_chunk[self.num_traj : self.adopted_action_chunks])

        actions = []
        for i in range(self.num_traj):
            action = action_chunk[i]
            base_action = np.concatenate(
                [obs["joint_state"][:5], np.array([0]), obs["joint_state"][6:8], np.array([0, 0, 0])]
            )
            actions.append(action + base_action)
        return np.array(actions)


def main():
    print("Start hsr_openpi")

    # TODO: 引数でいい感じに処理するようにする
    config_name = "pi0_hsr_tmc_weblab"
    checkpoint_dir = "/home/veluga-g3/pi0_hsr_low_mem_finetune_grp6/my_experiment/39999"
    adopted_action_chunks = 15

    print(f"config_name: {config_name}")
    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"adopted_action_chunks: {adopted_action_chunks}")

    policy = OpenpiPolicy(config_name, checkpoint_dir, adopted_action_chunks)
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
