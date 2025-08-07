from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.dataset import ModalityConfig
from gr00t.experiment.data_config import DATA_CONFIG_MAP

from gr00t.model.policy import Gr00tPolicy

# get the data config
data_config = DATA_CONFIG_MAP["hsr"]

# get the modality configs and transforms
modality_config = data_config.modality_config()
transforms = data_config.transform()

# This is a LeRobotSingleDataset object that loads the data from the given dataset path.
dataset = LeRobotSingleDataset(
    dataset_path="/home/kohei/codes/matuolab/tmc_new_gr00t",
    modality_configs=modality_config,
    transforms=None,  # we can choose to not apply any transforms
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT, # the embodiment to use
    video_backend="torchvision_av"
)

policy = Gr00tPolicy(
    model_path="/home/kohei/codes/matuolab/checkpoint-40000",
    modality_config=modality_config,
    modality_transform=transforms,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,  # the embodiment to
    device="cuda",
)

action_chunk = policy.get_action(dataset[0])

print("=== Action chunk values === ")

print(action_chunk)