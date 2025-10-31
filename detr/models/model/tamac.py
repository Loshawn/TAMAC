import torch
from torch import nn
from einops import rearrange
from .common import get_sinusoid_encoding_table, reparametrize

import IPython

e = IPython.embed


class TAMAC(nn.Module):
    """This is the DETR module that performs object detection"""

    def __init__(
        self,
        backbones,
        transformer,
        encoder,
        state_dim,
        num_queries,
        camera_names,
        history_size=50,
        use_full_seq=False,
    ):
        """Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.history_size = history_size
        self.cam_history_size = history_size // 10
        self.use_full_seq = use_full_seq

        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        if self.use_full_seq:
            self.proprio_head = nn.Linear(hidden_dim, state_dim)

        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32  # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
        self.encoder_action_proj = nn.Linear(14, hidden_dim)  # project action to embedding
        self.encoder_joint_proj = nn.Linear(14, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(
            hidden_dim, self.latent_dim * 2
        )  # project hidden state to latent std, var
        self.register_buffer(
            "pos_table", get_sinusoid_encoding_table(1 + 50 + num_queries, hidden_dim)
        )  # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)  # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(
            100, hidden_dim
        )  # learned position embedding for proprio and latent

    def forward(self, qpos, image, env_state, actions=None, is_pad=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        bs = qpos.shape[0]

        image_future = image[:, len(self.camera_names) * self.cam_history_size :].clone()
        image = image[:, : len(self.camera_names) * self.cam_history_size].clone()
        qpos_future = qpos[:, len(self.camera_names) * self.history_size :].clone()
        qpos = qpos[:, : len(self.camera_names) * self.history_size].clone()

        ### Obtain latent z from action sequence
        if actions is not None:
            # get latent_input
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, 50, hidden_dim)

            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1)  # (bs, 1, hidden_dim)
            encoder_input = torch.cat(
                [cls_embed, qpos_embed, action_embed], axis=1
            )  # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 51), False).to(qpos.device)  # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0]  # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, : self.latent_dim]
            logvar = latent_info[:, self.latent_dim :]
            latent_sample = reparametrize(mu, logvar).unsqueeze(1).repeat(1, 50, 1)
            latent_input = self.latent_out_proj(latent_sample)

        else:
            mu = logvar = None
            latent_sample = (
                torch.zeros([bs, self.latent_dim], dtype=torch.float32)
                .to(qpos.device)
                .unsqueeze(1)
                .repeat(1, 50, 1)
            )
            latent_input = self.latent_out_proj(latent_sample)

        # get src and pos
        features_list = []
        pos_list = []

        if self.training:
            image_total = torch.cat([image, image_future], axis=0)  # (2*bs, N, C, H, W)
            for i in range(image_total.shape[1]):  # N = 5
                single_image_total = image_total[:, i, :, :, :]  # (2*bs, C, H, W)

                features, pos = self.backbones[0](single_image_total)
                features = features[0]  # last layer
                pos = pos[0]

                features_list.append(features)
                pos_list.append(pos)

            features = torch.stack(features_list, dim=1)
            pos = torch.stack(pos_list, dim=1)

            project_feature = features[:bs, :]
            all_cam_features_future = features[bs:, :]

            project_feature = rearrange(project_feature, "b n c w h -> b c (w n) h")  # = src

        else:
            for i in range(image.shape[1]):  # (bs, N, C, H, W), N = 5
                single_image = image[:, i, :, :, :]

                features, pos = self.backbones[0](
                    single_image.reshape([-1, 3, single_image.shape[-2], single_image.shape[-1]])
                )
                features = features[0]
                pos = pos[0]

                features_list.append(features)
                pos_list.append(pos)

            features = torch.stack(features_list, dim=1)
            pos = torch.stack(pos_list, dim=1)

            project_feature = rearrange(features, "b n c w h -> b c (w n) h")  # = src

        src = project_feature
        pos = rearrange(pos, "b n c w h -> b c (w n) h")
        # proprioception features
        proprio_input = self.input_proj_robot_state(qpos)

        hs = self.transformer(
            src,
            None,
            self.query_embed.weight,
            pos,
            latent_input,
            proprio_input,
            self.additional_pos_embed.weight,
        )[0]

        if self.use_full_seq:
            hs_proprio = hs[:, : self.num_queries, :].clone()
            hs_img = hs[:, self.num_queries : -1 * self.num_queries, :].clone()
            hs_action = hs[:, -1 * self.num_queries :, :].clone()

            a_hat = self.action_head(hs_action)
            a_proprio = self.proprio_head(hs_proprio)

            if self.training:
                src_future = all_cam_features_future.flatten(3)
                src_future = rearrange(src_future, "b n c hw -> b (hw n) c")
                hs_img = {"hs_img": hs_img, "src_future": src_future}

            return a_hat, a_proprio, hs_img, qpos_future, [mu, logvar]

        else:
            a_hat = self.action_head(hs)
            is_pad_hat = self.is_pad_head(hs)
            return a_hat, is_pad_hat, [mu, logvar]
