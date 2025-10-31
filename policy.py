import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.main import (
    build_ACT_model_and_optimizer,
    build_CNNMLP_model_and_optimizer,
    build_TAMAC_model_and_optimizer,
)
import IPython

e = IPython.embed


class TAMACPolicy(nn.Module):

    def __init__(self, args_override: dict):
        super().__init__()
        model, optimizer = build_TAMAC_model_and_optimizer(args_override)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override["kl_weight"]
        self.use_full_seq = args_override["use_full_seq"]
        self.feature_loss_weight = args_override.get("feature_loss_weight", 0.0)
        print(f"KL Weight {self.kl_weight}")

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, : self.model.num_queries]
            is_pad = is_pad[:, : self.model.num_queries]

            output = self.model(qpos, image, env_state, actions, is_pad)

            loss_dict = dict()

            if self.use_full_seq:
                a_hat, a_proprio, hs_img_dict, qpos_future, (mu, logvar) = output

                # l1 loss -> action
                all_l1 = F.l1_loss(actions, a_hat, reduction="none")
                l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
                loss_dict["l1_action"] = l1

                # kl loss -> latent (z)
                total_kld, _, _ = kl_divergence(mu, logvar)
                loss_dict["kl"] = total_kld[0] * self.kl_weight

                if self.model.training:
                    # l1 loss -> pos
                    loss_dict["l1_pos"] = 0.05 * F.l1_loss(
                        qpos_future[:, :-1, :], a_proprio[:, :-1, :]
                    ) + 0.1 * F.l1_loss(qpos_future[:, -1, :], a_proprio[:, -1, :])

                    # mse loss -> image
                    loss_dict["l2_feature"] = (
                        self.feature_loss_weight
                        * F.mse_loss(
                            hs_img_dict["hs_img"][:, 0:1200, :], hs_img_dict["src_future"][:, 0:1200, :]
                        ).mean()
                        + 0.1
                        * F.mse_loss(
                            hs_img_dict["hs_img"][:, 1200:1500, :], hs_img_dict["src_future"][:, 1200:1500, :]
                        ).mean()
                    )

                    loss_dict["loss"] = (
                        loss_dict["l1_action"]
                        + loss_dict["kl"]
                        + loss_dict["l2_feature"]
                        + loss_dict["l1_pos"]
                    )
                else:
                    loss_dict["loss"] = loss_dict["l1_action"] + loss_dict["kl"]

            else:
                a_hat, is_pad_hat, (mu, logvar) = output

                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

                all_l1 = F.l1_loss(actions, a_hat, reduction="none")
                l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
                loss_dict["l1"] = l1
                loss_dict["kl"] = total_kld[0] * self.kl_weight
                loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"]

            return loss_dict

        else:  # inference time
            a_hat = self.model(qpos, image, env_state)[0]  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class ACTPolicy(nn.Module):

    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override["kl_weight"]
        print(f"KL Weight {self.kl_weight}")

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, : self.model.num_queries]
            is_pad = is_pad[:, : self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            return loss_dict
        else:  # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):

    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model  # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None  # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict["mse"] = mse
            loss_dict["loss"] = loss_dict["mse"]
            return loss_dict
        else:  # inference time
            a_hat = self.model(qpos, image, env_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
