FROM fairembodied/habitat-challenge:testing_2022_habitat_base_docker

RUN /bin/bash -c ". activate habitat; pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html; pip install pillow==6.1"
RUN /bin/bash -c ". activate habitat; python -m pip install git+https://github.com/facebookresearch/detectron2.git"

ADD rl_semseg_agent.py agent.py
ADD submission.sh submission.sh
ADD configs/challenge_objectnav2022.local.rgbd.yaml /challenge_objectnav2022.local.rgbd.yaml
ENV AGENT_EVALUATION_TYPE remote

# Add checkpoints and custom configs

ADD ckpts/objectnav_env/sem_seg_pred/rl_ft_hm3d/scratch_pt_77k_4node/ckpt.32.pth ckpt/model.pth

ADD ckpts/rednet/rednet_semmap_mp3d_tuned.pth ckpt/rednet_semmap_mp3d_tuned.pth

ADD ckpts/rednet/rednet_semmap_mp3d_40_v2_vince.pth ckpt/rednet_semmap_mp3d_40_v2_vince.pth

ADD ckpts/ddppo-models/gibson-2plus-resnet50.pth ckpt/gibson-2plus-resnet50.pth

ADD configs/rl_objectnav_sem_seg_hm3d.yaml configs/rl_objectnav_sem_seg.yaml

# Add src folder for models    parser.add_argument("--config-path", type=str, required=True, default="configs/aux_objectnav.yaml")

ADD src/ src/

ENV AGENT_CONFIG_FILE "configs/rl_objectnav_sem_seg.yaml"
ENV TRACK_CONFIG_FILE "/challenge_objectnav2022.local.rgbd.yaml"

CMD ["/bin/bash", "-c", "source activate habitat && export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && bash submission.sh --model-path ckpt/model.pth --config-path $AGENT_CONFIG_FILE"]
