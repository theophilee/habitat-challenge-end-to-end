import numpy as np
import random
import torch

from gym.spaces import Box
from gym.spaces import Dict as SpaceDict
from gym.spaces import Discrete
from habitat.config import Config
from habitat.core.logging import logger
from habitat.core.agent import Agent
from habitat.core.env import Env

from src import POLICY_CLASSES
from src.default import get_config
from src.models.common import batch_obs
from src.models.rednet import load_rednet
from src.segmentation.coco_segmentation_model import COCOSegmentationModel


class RLSegFTAgent(Agent):
    def __init__(self, config: Config):
        if not config.MODEL_PATH:
            raise Exception(
                "Model checkpoint wasn't provided, quitting."
            )
        if config.TORCH_GPU_ID >= 0:
            self.device = torch.device("cuda:{}".format(config.TORCH_GPU_ID))
        else:
            self.device = torch.device("cpu")

        ckpt_dict = torch.load(config.MODEL_PATH, map_location=self.device)["state_dict"]
        ckpt_dict = {
            k.replace("actor_critic.", ""): v
            for k, v in ckpt_dict.items()
        }
        ckpt_dict = {
            k.replace("module.", ""): v
            for k, v in ckpt_dict.items()
        }

        # Config
        self.config = config
        config = self.config.clone()
        self.model_cfg = config.MODEL
        il_cfg = config.IL.BehaviorCloning
        task_cfg = config.TASK_CONFIG.TASK

        # Load spaces (manually)
        spaces = {
            "objectgoal": Box(
                low=0, high=20,  # From matterport dataset
                shape=(1,),
                dtype=np.int64
            ),
            "depth": Box(
                low=0,
                high=1,
                shape=(config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT,
                        config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH, 1),
                dtype=np.float32,
            ),
            "rgb": Box(
                low=0,
                high=255,
                shape=(config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT,
                        config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH, 3),
                dtype=np.uint8,
            ),
            "gps": Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,),  # Spoof for model to be shaped correctly
                dtype=np.float32,
            ),
            "compass": Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float)
        }

        observation_spaces = SpaceDict(spaces)
        if 'action_distribution.linear.bias' in ckpt_dict:
            num_acts = ckpt_dict['action_distribution.linear.bias'].size(0)
        action_spaces = Discrete(num_acts)

        is_objectnav = "ObjectNav" in task_cfg.TYPE
        additional_sensors = []
        if is_objectnav:
            additional_sensors = ["gps", "compass"]

        policy_class = POLICY_CLASSES[il_cfg.POLICY.name]
        self.model = policy_class(
            observation_space=observation_spaces,
            action_space=action_spaces,
            model_config=self.model_cfg,
            device=self.device,
            goal_sensor_uuid=task_cfg.GOAL_SENSOR_UUID,
            additional_sensors=additional_sensors,
        ).to(self.device)

        self.model.load_state_dict(ckpt_dict, strict=True)
        self.model.eval()

        self.semantic_predictor = None

        if self.model_cfg.USE_SEMANTICS:
            logger.info("setting up sem seg predictor")
            # self.semantic_predictor = load_rednet(
            #     self.device,
            #     ckpt=self.model_cfg.SEMANTIC_ENCODER.rednet_ckpt,
            #     resize=True,  # Since we train on half-vision
            #     num_classes=self.model_cfg.SEMANTIC_ENCODER.num_classes
            # )
            # self.semantic_predictor.eval()
            self.semantic_predictor = COCOSegmentationModel(
                sem_pred_prob_thr=0.9, sem_gpu_id=config.TORCH_GPU_ID, visualize=False
            )

        # Load other items
        self.test_recurrent_hidden_states = torch.zeros(
            self.model_cfg.STATE_ENCODER.num_recurrent_layers,
            1,  # num_processes
            self.model_cfg.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        self.not_done_masks = torch.zeros(1, 1, device=self.device, dtype=torch.bool)
        self.prev_actions = torch.zeros(
            1, 1, dtype=torch.long, device=self.device
        )

        self.ep = 0

    def reset(self):
        # We don't reset state because our rnn accounts for masks, and ignore actions because we don't use actions
        self.not_done_masks = torch.zeros(1, 1, device=self.device, dtype=torch.bool)
        self.prev_actions = torch.zeros(1, 1, dtype=torch.long, device=self.device)

        # Load other items
        self.test_recurrent_hidden_states = torch.zeros(
            self.model_cfg.STATE_ENCODER.num_recurrent_layers,
            1,  # num_processes
            self.model_cfg.STATE_ENCODER.hidden_size,
            device=self.device,
        )

        self.ep += 1
        logger.info("Episode done: {}".format(self.ep))

    @torch.no_grad()
    def act(self, observations):

        batch = batch_obs([observations], device=self.device)

        with torch.no_grad():
            if self.semantic_predictor is not None:
                # batch["semantic"] = self.semantic_predictor(batch["rgb"], batch["depth"])
                # if self.config.MODEL.SEMANTIC_ENCODER.is_thda:
                #     batch["semantic"] = batch["semantic"] - 1
                semantic, semantic_vis = self.semantic_predictor.get_prediction(
                    batch["rgb"].cpu().numpy(),
                    batch["depth"].cpu().numpy()
                )
                batch["semantic"] = torch.from_numpy(semantic).to(batch["rgb"].device)

            logits, self.test_recurrent_hidden_states = self.model(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
            )
            actions = torch.argmax(logits, dim=1)

            self.prev_actions.copy_(actions)

        # Reset called externally, we're not done until then
        self.not_done_masks = torch.ones(1, 1, device=self.device, dtype=torch.bool)
        return actions[0].item()


def main():
    challenge_config_file = "configs/challenge_objectnav2022.local.rgbd.yaml"
    agent_config_file = "configs/rl_objectnav_sem_seg_hm3d.yaml"
    model_path = "ckpt/model.pth"

    config = get_config(agent_config_file, ['BASE_TASK_CONFIG_PATH', challenge_config_file])
    config.defrost()
    config.TORCH_GPU_ID = 0
    config.MODEL_PATH = model_path
    seed = 7
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    config.RANDOM_SEED = seed
    config.freeze()

    agent = RLSegFTAgent(config)
    env = Env(config=config.TASK_CONFIG)

    obs = env.reset()
    agent.reset()

    t = 0
    while not env.episode_over:
        t += 1
        print(t)
        action = agent.act(obs)
        obs = env.step(action)


if __name__ == "__main__":
    main()
