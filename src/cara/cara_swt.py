"""Implement CaRA."""

from typing import Any, Dict
import numpy as np
import tensorly as tl
import timm
import torch as th
import torch.nn as nn
import torch

tl.set_backend("pytorch")

global_model: th.nn.Module


def cp_attn(self, x: th.Tensor, mask=None, attn_mask=None) -> th.Tensor:
    """Forward pass.

        Args:
            x: Input features with shape of (num_windows*B, N, C).
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None.

        Returns:
            Output features with shape of (num_windows*B, N, C).
        """
    B_, N, C = x.shape
    qkv = self.qkv(x)#.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    #print(self._embedding_dim_unit, self.act_heads)
    f1 = global_model.CP_A1[self.attn_idx : self.attn_idx + 3, :]
    f2 = global_model.CP_A2[:self._embedding_dim_unit * self.act_heads, :]
    f3 = global_model.CP_A3[:self.act_heads, :]
    #print(f1.shape, f2.shape, f3.shape, global_model.CP_A4.shape)
    tensor_attn = tl.cp_to_tensor(
        (
            global_model.CP_R1,
            (f1, f2, f3, global_model.CP_A4),
        )
    )
    K, E, H, D = tensor_attn.shape
    tensor_attn = tensor_attn.reshape((K, E, H * D))
    qkv_delta = th.einsum("bnd, kde->kbne", x, self.dp(tensor_attn))
    qkv_delta = qkv_delta.reshape(
        3, B_, N, self.num_heads, C // self.num_heads
    ).permute(0, 1, 3, 2, 4)
    qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(
        2, 0, 3, 1, 4
    )
    qkv += qkv_delta * self.s
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn + self._get_rel_pos_bias()
    if mask is not None:
        num_win = mask.shape[0]
        attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
    attn = self.softmax(attn)
    attn = self.attn_drop(attn)
    x = attn @ v

    x = x.transpose(1, 2).reshape(B_, N, -1)

    p1 = global_model.CP_P1[self.idx : self.idx + 1, :]
    p2 = global_model.CP_P2[:self._embedding_dim_unit * self.act_heads, :]
    p3 = global_model.CP_P3[:self._embedding_dim_unit * self.act_heads, :]
    tensor_proj = tl.cp_to_tensor(
        (global_model.CP_R2, (p1, p2, p3))
    )
    AA, AB, AC = tensor_proj.shape
    tensor_proj = tensor_proj.reshape((AA * AB, AC))
    proj_delta = x @ self.dp(tensor_proj.T)
    #print(tensor_proj.shape)
    proj_delta += global_model.CP_bias1[:self._embedding_dim_unit * self.act_heads]
    proj = self.proj(x)
    proj += proj_delta * self.s
    x = self.proj_drop(x)
    self.cara_info = f'Attn qkv delta shape:{qkv_delta.shape}, Attn delta shape:{proj_delta.shape}'
    return x


def cp_mlp(self, x: th.Tensor) -> th.Tensor:
    """Mlp with CP parameters.

    Args:
        x (th.Tensor): Input tensor.

    Returns:
        th.Tensor: Mlp projected output.
    """
    p1_up = global_model.CP_P1[self.idx : self.idx + 4, :]
    p1_down = global_model.CP_P1[self.idx + 4 : self.idx + 8, :]
    p2 = global_model.CP_P2[:self._embedding_dim_unit * self.act_heads, :]
    p3 = global_model.CP_P3[:self._embedding_dim_unit * self.act_heads, :]
    #print(p1_up.shape, p1_down.shape, p2.shape, p3.shape)

    up = self.fc1(x)
    tensor_up = tl.cp_to_tensor(
        (global_model.CP_R2, (p1_up, p2, p3))
    )
    AA, AB, AC = tensor_up.shape
    tensor_up = tensor_up.reshape((AA * AB, AC))
    #print(tensor_up.shape)
    up_delta = x @ self.dp(tensor_up.T) + global_model.CP_bias2[:self._embedding_dim_unit * self.act_heads*4]
    up += up_delta * self.s

    x = self.act(up)
    x = self.drop1(x)

    down = self.fc2(x)
    tensor_down = tl.cp_to_tensor(
        (global_model.CP_R2, (p1_down, p2, p3))
    )
    tensor_down = tensor_down.reshape((AA * AB, AC))
    down_delta = x @ self.dp(tensor_down) + global_model.CP_bias3[:self._embedding_dim_unit * self.act_heads]
    down += down_delta * self.s
    x = self.drop2(down)
    self.cara_info = f'FC1 delta shape:{up_delta.shape}, FC2 delta shape:{down_delta.shape}'
    return x

def get_swt_num_heads(block_idx, layer_config):
    s = 0
    for i, v in enumerate(layer_config):
        if s <= block_idx  < s + v:
            return i
        s += v
    return None

def set_cara_swt(
    model: nn.Module, rank: int, scale: float, l_mu: float, l_std: float
    , model_config = {
                'embedding_dim': 96,  #depend on the model setting
                'depths': [2, 2, 6, 2],
                'num_heads': [3, 6, 12, 24],}
) -> None:
    """Cara setup.

    Args:
        model (nn.Module): ViT model.
        rank (int): FT Rank.
        scale (float): FT scale.
        l_mu (float): Init lambda_mu.
        l_std (float): Init lambda_std.
    """
    if type(model) is timm.models.swin_transformer.SwinTransformer:
        # Declare CaRA parameters
        model_config['num_heads'] = np.array(model_config['num_heads']) // 3
        model_config['num_head'] = model_config['num_heads'][-1]
        model_config['num_layers'] = sum(model_config['depths'])
        model_config['_embedding_dim_unit'] = model_config['embedding_dim']
        #update the embedding dim to maximum one, swt defines each q,k,v as one head
        # the q,k,v splitted inside the attention module are incorperated into the layer dimension
        model_config['embedding_dim'] = model_config['embedding_dim']*model_config['num_head']
        global_model.num_heads_cara = model_config['num_heads']
        global_model._embedding_dim_unit_cara = model_config['_embedding_dim_unit']
        global_model.depths_cara = model_config['depths']
        print(model_config)

        model.CP_A1 = nn.Parameter(th.empty([3*model_config['num_layers'], rank]), requires_grad=True)
        model.CP_A2 = nn.Parameter(th.empty([model_config['embedding_dim'], rank]), requires_grad=True)
        model.CP_A3 = nn.Parameter(th.empty([model_config['num_head'], rank]), requires_grad=True)
        head_dim = model_config['_embedding_dim_unit']
        model.CP_A4 = nn.Parameter(
            th.empty([head_dim, rank]), requires_grad=True
        )
        model.CP_P1 = nn.Parameter(th.empty([9*model_config['num_layers'], rank]), requires_grad=True)
        model.CP_P2 = nn.Parameter(th.empty([model_config['embedding_dim'], rank]), requires_grad=True)
        model.CP_P3 = nn.Parameter(th.empty([model_config['embedding_dim'], rank]), requires_grad=True)
        model.CP_R1 = nn.Parameter(th.empty([rank]), requires_grad=True)
        model.CP_R2 = nn.Parameter(th.empty([rank]), requires_grad=True)
        model.CP_bias1 = nn.Parameter(th.empty([model_config['embedding_dim']]), requires_grad=True)
        model.CP_bias2 = nn.Parameter(th.empty([model_config['embedding_dim'] * 4]), requires_grad=True)
        model.CP_bias3 = nn.Parameter(th.empty([model_config['embedding_dim']]), requires_grad=True)
        # Initialise CaRA parameters
        nn.init.xavier_normal_(model.CP_A1)
        nn.init.zeros_(model.CP_A2)
        nn.init.orthogonal_(model.CP_A3)
        nn.init.orthogonal_(model.CP_A4)
        nn.init.xavier_normal_(model.CP_P1)
        nn.init.zeros_(model.CP_P2)
        nn.init.orthogonal_(model.CP_P3)
        if l_std != 0.0:
            nn.init.normal_(model.CP_R1, mean=l_mu, std=l_std)
            nn.init.normal_(model.CP_R2, mean=l_mu, std=l_std)
        elif l_mu == 1.0 and l_std == 0.0:
            nn.init.ones_(model.CP_R1)
            nn.init.ones_(model.CP_R2)
        nn.init.zeros_(model.CP_bias1)
        nn.init.zeros_(model.CP_bias2)
        nn.init.zeros_(model.CP_bias3)
        # CaRA indexing
        model.idx = 0
        model.attn_idx = 0
    for child in model.children():
        if type(child) is timm.models.swin_transformer.WindowAttention:
            child.dp = nn.Dropout(0.1)
            child.s = scale
            child.dim = rank
            child.idx = global_model.idx
            child.attn_idx = global_model.attn_idx
            child.act_heads = global_model.num_heads_cara[get_swt_num_heads(global_model.attn_idx//3, global_model.depths_cara)]
            child._embedding_dim_unit = global_model._embedding_dim_unit_cara

            global_model.idx += 1
            global_model.attn_idx += 3
            bound_method = cp_attn.__get__(child, child.__class__)
            setattr(child, "forward", bound_method)  # noqa: B010
        elif type(child) is timm.layers.mlp.Mlp:    #timm.models.layers.mlp.Mlp:
            child.dp = nn.Dropout(0.1)
            child.s = scale
            child.dim = rank
            child.idx = global_model.idx
            child.act_heads = global_model.num_heads_cara[get_swt_num_heads((global_model.attn_idx - 3)//3, global_model.depths_cara)]
            child._embedding_dim_unit = global_model._embedding_dim_unit_cara

            global_model.idx += 8
            bound_method = cp_mlp.__get__(child, child.__class__)
            setattr(child, "forward", bound_method)  # noqa: B010
        elif len(list(child.children())) != 0:
            set_cara_swt(child, rank, scale, l_mu, l_std)


def cara_swt(config: Dict[str, Any]) -> th.nn.Module:
    """Set CaRA for the given configuration.

    Args:
        config (Dict[str, Any]): Dictionary containing CaRA configuration.

    Returns:
        th.nn.Module: CaRA model.
    """
    # CaRA parameters
    model = config["model"]
    rank = config["rank"]
    scale = config["scale"]
    l_mu = config["l_mu"]
    l_std = config["l_std"]

    global global_model
    global_model = model
    set_cara_swt(model, rank, scale, l_mu, l_std)
    return global_model