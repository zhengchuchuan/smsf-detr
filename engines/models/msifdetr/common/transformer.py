# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""
Transformer class
"""
import math
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from engines.models.msifdetr.common.ops.modules import MSDeformAttn

class ScalarGate(nn.Module):
    """
    标量门控参数容器。

    BaseTrainer 会识别名称中包含 ".fusion_alpha." 的参数并禁用 weight decay，
    因此用 fusion_alpha.value 的命名方式，避免 AdamW 把 gate 系统性压到 0。
    """

    def __init__(self, init: float = 0.0):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(init)))

    def forward(self) -> Tensor:
        return self.value

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def gen_sineembed_for_position(pos_tensor, dim=128):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(dim, dtype=pos_tensor.dtype, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / dim)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


def gen_encoder_output_proposals(memory, memory_padding_mask, spatial_shapes, unsigmoid=True):
    """
    Input:
        - memory: bs, \sum{hw}, d_model
        - memory_padding_mask: bs, \sum{hw}
        - spatial_shapes: nlevel, 2
    Output:
        - output_memory: bs, \sum{hw}, d_model
        - output_proposals: bs, \sum{hw}, 4
    """
    N_, S_, C_ = memory.shape
    base_scale = 4.0
    proposals = []
    _cur = 0
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        if memory_padding_mask is not None:
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
        else:
            valid_H = torch.tensor([H_ for _ in range(N_)], device=memory.device)
            valid_W = torch.tensor([W_ for _ in range(N_)], device=memory.device)

        grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                        torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1) # H_, W_, 2

        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale

        wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)

        proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
        proposals.append(proposal)
        _cur += (H_ * W_)

    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)

    if unsigmoid:
        output_proposals = torch.log(output_proposals / (1 - output_proposals)) # unsigmoid
        if memory_padding_mask is not None:
            output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))
    else:
        if memory_padding_mask is not None:
            output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float(0))

    output_memory = memory
    if memory_padding_mask is not None:
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))

    return output_memory.to(memory.dtype), output_proposals.to(memory.dtype)


class Transformer(nn.Module):

    def __init__(self, d_model=512, sa_nhead=8, ca_nhead=8, num_queries=300,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, group_detr=1, 
                 two_stage=False,
                 num_feature_levels=4, dec_n_points=4,
                 lite_refpoint_refine=False,
                 decoder_norm_type='LN',
                 bbox_reparam=False,
                 # decoder 侧 ms 融合（Query 级语义对齐）
                 decoder_ms_fusion: bool = False,
                 decoder_ms_fusion_alpha_init: float = 0.0,
                 decoder_ms_fusion_alpha_scale: float = 1.0,
                 freeze_decoder_ms_fusion_alpha: bool = False,
                 # decoder 侧 ms 融合：对 MSI 分支的 reference_points 做可学习平移（补偿残余错位）
                 decoder_ms_fusion_ref_shift: bool = False,
                 decoder_ms_fusion_ref_shift_init=(0.0, 0.0),
                 decoder_ms_fusion_ref_shift_scale: float = 0.02,
                 freeze_decoder_ms_fusion_ref_shift: bool = False):
        super().__init__()
        self.encoder = None

        decoder_layer = TransformerDecoderLayer(d_model, sa_nhead, ca_nhead, dim_feedforward,
                                                dropout, activation, normalize_before, 
                                                group_detr=group_detr,
                                                num_feature_levels=num_feature_levels,
                                                dec_n_points=dec_n_points,
                                                skip_self_attn=False,
                                                decoder_ms_fusion=decoder_ms_fusion,
                                                decoder_ms_fusion_alpha_init=decoder_ms_fusion_alpha_init,
                                                decoder_ms_fusion_alpha_scale=decoder_ms_fusion_alpha_scale,
                                                freeze_decoder_ms_fusion_alpha=freeze_decoder_ms_fusion_alpha,
                                                decoder_ms_fusion_ref_shift=decoder_ms_fusion_ref_shift,
                                                decoder_ms_fusion_ref_shift_init=decoder_ms_fusion_ref_shift_init,
                                                decoder_ms_fusion_ref_shift_scale=decoder_ms_fusion_ref_shift_scale,
                                                freeze_decoder_ms_fusion_ref_shift=freeze_decoder_ms_fusion_ref_shift,)
        assert decoder_norm_type in ['LN', 'Identity']
        norm = { 
            "LN": lambda channels: nn.LayerNorm(channels),
            "Identity": lambda channels: nn.Identity(),
        }
        decoder_norm = norm[decoder_norm_type](d_model)

        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model,
                                          lite_refpoint_refine=lite_refpoint_refine,
                                          bbox_reparam=bbox_reparam)
        
        
        self.two_stage = two_stage
        if two_stage:
            self.enc_output = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(group_detr)])
            self.enc_output_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(group_detr)])

        self._reset_parameters()

        self.num_queries = num_queries
        self.d_model = d_model
        self.dec_layers = num_decoder_layers
        self.group_detr = group_detr
        self.num_feature_levels = num_feature_levels
        self.bbox_reparam = bbox_reparam

        self._export = False
    
    def export(self):
        self._export = True

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
    
    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(
        self,
        srcs,
        masks,
        pos_embeds,
        refpoint_embed,
        query_feat,
        *,
        ms_srcs=None,
        ms_masks=None,
        ms_pos_embeds=None,
    ):
        # 原实现依赖于 for-loop 中的 `bs` 变量在函数作用域内可见；
        # 引入 _flatten_inputs 后需显式获取 batch_size，避免后续 group_detr 的 split/cat 逻辑报错。
        bs = int(srcs[0].shape[0]) if len(srcs) > 0 else int(query_feat.shape[0])

        def _flatten_inputs(_srcs, _masks, _pos_embeds):
            src_flatten = []
            mask_flatten = [] if _masks is not None else None
            lvl_pos_embed_flatten = []
            spatial_shapes = []
            valid_ratios = [] if _masks is not None else None
            for lvl, (src, pos_embed) in enumerate(zip(_srcs, _pos_embeds)):
                bs, c, h, w = src.shape
                spatial_shapes.append((h, w))
                src = src.flatten(2).transpose(1, 2)                # bs, hw, c
                pos_embed = pos_embed.flatten(2).transpose(1, 2)    # bs, hw, c
                lvl_pos_embed_flatten.append(pos_embed)
                src_flatten.append(src)
                if _masks is not None:
                    mask = _masks[lvl].flatten(1)                    # bs, hw
                    mask_flatten.append(mask)
            memory = torch.cat(src_flatten, 1)    # bs, \sum{hxw}, c
            if _masks is not None:
                mask_flatten = torch.cat(mask_flatten, 1)   # bs, \sum{hxw}
                valid_ratios = torch.stack([self.get_valid_ratio(m) for m in _masks], 1)
            lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1) # bs, \sum{hxw}, c
            spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=memory.device)
            level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
            return memory, mask_flatten, lvl_pos_embed_flatten, spatial_shapes, level_start_index, valid_ratios

        memory, mask_flatten, lvl_pos_embed_flatten, spatial_shapes, level_start_index, valid_ratios = _flatten_inputs(
            srcs, masks, pos_embeds
        )

        ms_memory = None
        ms_mask_flatten = None
        if ms_srcs is not None:
            ms_memory, ms_mask_flatten, ms_lvl_pos, ms_spatial_shapes, ms_level_start_index, ms_valid_ratios = _flatten_inputs(
                ms_srcs, ms_masks, ms_pos_embeds
            )
            if not torch.equal(ms_spatial_shapes, spatial_shapes):
                raise ValueError(
                    f"Decoder MS fusion 需要 RGB/MS spatial_shapes 一致，但收到 rgb={spatial_shapes.tolist()} ms={ms_spatial_shapes.tolist()}"
                )
            if not torch.equal(ms_level_start_index, level_start_index):
                raise ValueError("Decoder MS fusion 需要 RGB/MS level_start_index 一致。")
            if (valid_ratios is None) != (ms_valid_ratios is None):
                raise ValueError("Decoder MS fusion 需要 RGB/MS valid_ratios 同时存在或同时为 None。")
        
        if self.two_stage:
            output_memory, output_proposals = gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes, unsigmoid=not self.bbox_reparam)
            # group detr for first stage
            refpoint_embed_ts, memory_ts, boxes_ts = [], [], []
            group_detr = self.group_detr if self.training else 1
            for g_idx in range(group_detr):
                output_memory_gidx = self.enc_output_norm[g_idx](self.enc_output[g_idx](output_memory))
    
                enc_outputs_class_unselected_gidx = self.enc_out_class_embed[g_idx](output_memory_gidx)
                if self.bbox_reparam:
                    enc_outputs_coord_delta_gidx = self.enc_out_bbox_embed[g_idx](output_memory_gidx)
                    enc_outputs_coord_cxcy_gidx = enc_outputs_coord_delta_gidx[...,
                        :2] * output_proposals[..., 2:] + output_proposals[..., :2]
                    enc_outputs_coord_wh_gidx = enc_outputs_coord_delta_gidx[..., 2:].exp() * output_proposals[..., 2:]
                    enc_outputs_coord_unselected_gidx = torch.concat(
                        [enc_outputs_coord_cxcy_gidx, enc_outputs_coord_wh_gidx], dim=-1)
                else:
                    enc_outputs_coord_unselected_gidx = self.enc_out_bbox_embed[g_idx](
                        output_memory_gidx) + output_proposals # (bs, \sum{hw}, 4) unsigmoid

                topk = min(self.num_queries, enc_outputs_class_unselected_gidx.shape[-2])
                topk_proposals_gidx = torch.topk(enc_outputs_class_unselected_gidx.max(-1)[0], topk, dim=1)[1] # bs, nq
                
                refpoint_embed_gidx_undetach = torch.gather(
                    enc_outputs_coord_unselected_gidx, 1, topk_proposals_gidx.unsqueeze(-1).repeat(1, 1, 4)) # unsigmoid
                # for decoder layer, detached as initial ones, (bs, nq, 4)
                refpoint_embed_gidx = refpoint_embed_gidx_undetach.detach()
                
                # get memory tgt
                tgt_undetach_gidx = torch.gather(
                    output_memory_gidx, 1, topk_proposals_gidx.unsqueeze(-1).repeat(1, 1, self.d_model))
                
                refpoint_embed_ts.append(refpoint_embed_gidx)
                memory_ts.append(tgt_undetach_gidx)
                boxes_ts.append(refpoint_embed_gidx_undetach)
            # concat on dim=1, the nq dimension, (bs, nq, d) --> (bs, nq, d)
            refpoint_embed_ts = torch.cat(refpoint_embed_ts, dim=1)
            # (bs, nq, d)
            memory_ts = torch.cat(memory_ts, dim=1)#.transpose(0, 1)
            boxes_ts = torch.cat(boxes_ts, dim=1)#.transpose(0, 1)
        
        if self.dec_layers > 0:
            tgt = query_feat.unsqueeze(0).repeat(bs, 1, 1)
            refpoint_embed = refpoint_embed.unsqueeze(0).repeat(bs, 1, 1)
            if self.two_stage:
                ts_len = refpoint_embed_ts.shape[-2]
                refpoint_embed_ts_subset = refpoint_embed[..., :ts_len, :]
                refpoint_embed_subset = refpoint_embed[..., ts_len:, :]

                if self.bbox_reparam:
                    refpoint_embed_cxcy = refpoint_embed_ts_subset[..., :2] * refpoint_embed_ts[..., 2:]
                    refpoint_embed_cxcy = refpoint_embed_cxcy + refpoint_embed_ts[..., :2]
                    refpoint_embed_wh = refpoint_embed_ts_subset[..., 2:].exp() * refpoint_embed_ts[..., 2:]
                    refpoint_embed_ts_subset = torch.concat(
                        [refpoint_embed_cxcy, refpoint_embed_wh], dim=-1
                    )
                else:
                    refpoint_embed_ts_subset = refpoint_embed_ts_subset + refpoint_embed_ts
                
                refpoint_embed = torch.concat(
                    [refpoint_embed_ts_subset, refpoint_embed_subset], dim=-2)

            hs, references = self.decoder(tgt, memory, memory_key_padding_mask=mask_flatten,
                            pos=lvl_pos_embed_flatten, refpoints_unsigmoid=refpoint_embed,
                            level_start_index=level_start_index, 
                            spatial_shapes=spatial_shapes,
                            valid_ratios=valid_ratios.to(memory.dtype) if valid_ratios is not None else valid_ratios,
                            ms_memory=ms_memory,
                            ms_memory_key_padding_mask=ms_mask_flatten)
        else:
            assert self.two_stage, "if not using decoder, two_stage must be True"
            hs = None
            references = None
        
        if self.two_stage:
            if self.bbox_reparam:
                return hs, references, memory_ts, boxes_ts
            else:
                return hs, references, memory_ts, boxes_ts.sigmoid()
        return hs, references, None, None


class TransformerDecoder(nn.Module):

    def __init__(self,
                 decoder_layer,
                 num_layers,
                 norm=None, 
                 return_intermediate=False,
                 d_model=256,
                 lite_refpoint_refine=False,
                 bbox_reparam=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.d_model = d_model
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.lite_refpoint_refine = lite_refpoint_refine
        self.bbox_reparam = bbox_reparam
        # 由 LWDETR 在初始化时按 lite_refpoint_refine/两阶段策略注入；
        # 这里提供默认值，便于独立构造 Transformer 做单元自检。
        self.bbox_embed = None

        self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)

        self._export = False
    
    def export(self):
        self._export = True

    def refpoints_refine(self, refpoints_unsigmoid, new_refpoints_delta):
        if self.bbox_reparam:
            new_refpoints_cxcy = new_refpoints_delta[..., :2] * refpoints_unsigmoid[..., 2:] + refpoints_unsigmoid[..., :2]
            new_refpoints_wh = new_refpoints_delta[..., 2:].exp() * refpoints_unsigmoid[..., 2:]
            new_refpoints_unsigmoid = torch.concat(
                [new_refpoints_cxcy, new_refpoints_wh], dim=-1
            )
        else:
            new_refpoints_unsigmoid = refpoints_unsigmoid + new_refpoints_delta
        return new_refpoints_unsigmoid

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None,
                # for memory
                level_start_index: Optional[Tensor] = None, # num_levels
                spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                valid_ratios: Optional[Tensor] = None,
                # ms memory（可选）
                ms_memory: Optional[Tensor] = None,
                ms_memory_key_padding_mask: Optional[Tensor] = None):
        output = tgt

        intermediate = []
        hs_refpoints_unsigmoid = [refpoints_unsigmoid]
        
        def get_reference(refpoints):
            # [num_queries, batch_size, 4]
            obj_center = refpoints[..., :4]
            
            if self._export:
                query_sine_embed = gen_sineembed_for_position(obj_center, self.d_model / 2) # bs, nq, 256*2 
                refpoints_input = obj_center[:, :, None] # bs, nq, 1, 4
            else:
                refpoints_input = obj_center[:, :, None] \
                                        * torch.cat([valid_ratios, valid_ratios], -1)[:, None] # bs, nq, nlevel, 4
                query_sine_embed = gen_sineembed_for_position(
                    refpoints_input[:, :, 0, :], self.d_model / 2) # bs, nq, 256*2 
            query_pos = self.ref_point_head(query_sine_embed)
            return obj_center, refpoints_input, query_pos, query_sine_embed
        
        # always use init refpoints
        if self.lite_refpoint_refine:
            if self.bbox_reparam:
                obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(refpoints_unsigmoid)
            else:
                obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(refpoints_unsigmoid.sigmoid())

        for layer_id, layer in enumerate(self.layers):
            # iter refine each layer
            if not self.lite_refpoint_refine:
                if self.bbox_reparam:
                    obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(refpoints_unsigmoid)
                else:
                    obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(refpoints_unsigmoid.sigmoid())

            # For the first decoder layer, we do not apply transformation over p_s
            pos_transformation = 1

            query_pos = query_pos * pos_transformation
            
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed, 
                           is_first=(layer_id == 0),
                           reference_points=refpoints_input,
                           spatial_shapes=spatial_shapes,
                           level_start_index=level_start_index,
                           ms_memory=ms_memory,
                           ms_memory_key_padding_mask=ms_memory_key_padding_mask)

            if not self.lite_refpoint_refine:
                # box iterative update
                new_refpoints_delta = self.bbox_embed(output)
                new_refpoints_unsigmoid = self.refpoints_refine(refpoints_unsigmoid, new_refpoints_delta)
                if layer_id != self.num_layers - 1:
                    hs_refpoints_unsigmoid.append(new_refpoints_unsigmoid)
                refpoints_unsigmoid = new_refpoints_unsigmoid.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if self._export:
                # to shape: B, N, C
                hs = intermediate[-1]
                if self.bbox_embed is not None:
                    ref = hs_refpoints_unsigmoid[-1]
                else:
                    ref = refpoints_unsigmoid
                return hs, ref
            # box iterative update
            if self.bbox_embed is not None:
                return [
                    torch.stack(intermediate),
                    torch.stack(hs_refpoints_unsigmoid),
                ]
            else:
                return [
                    torch.stack(intermediate), 
                    refpoints_unsigmoid.unsqueeze(0)
                ]

        return output.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, sa_nhead, ca_nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, group_detr=1, 
                 num_feature_levels=4, dec_n_points=4, 
                 skip_self_attn=False,
                 decoder_ms_fusion: bool = False,
                 decoder_ms_fusion_alpha_init: float = 0.0,
                 decoder_ms_fusion_alpha_scale: float = 1.0,
                 freeze_decoder_ms_fusion_alpha: bool = False,
                 decoder_ms_fusion_ref_shift: bool = False,
                 decoder_ms_fusion_ref_shift_init=(0.0, 0.0),
                 decoder_ms_fusion_ref_shift_scale: float = 0.02,
                 freeze_decoder_ms_fusion_ref_shift: bool = False):
        super().__init__()
        # Decoder Self-Attention
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=sa_nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Decoder Cross-Attention
        self.cross_attn = MSDeformAttn(
            d_model, n_levels=num_feature_levels, n_heads=ca_nhead, n_points=dec_n_points)
        self.decoder_ms_fusion = bool(decoder_ms_fusion)
        if self.decoder_ms_fusion:
            self.ms_cross_attn = MSDeformAttn(
                d_model, n_levels=num_feature_levels, n_heads=ca_nhead, n_points=dec_n_points)
            self.fusion_alpha = ScalarGate(float(decoder_ms_fusion_alpha_init))
            self.fusion_alpha_scale = float(decoder_ms_fusion_alpha_scale)
            if bool(freeze_decoder_ms_fusion_alpha):
                self.fusion_alpha.value.requires_grad = False

            # 对 MSI 分支 reference_points 做可学习平移（dx, dy），用于补偿 RGB/MSI 的残余像素级错位。
            self.ms_ref_shift_enabled = bool(decoder_ms_fusion_ref_shift)
            if self.ms_ref_shift_enabled:
                init_xy = decoder_ms_fusion_ref_shift_init
                try:
                    init_x, init_y = float(init_xy[0]), float(init_xy[1])
                except Exception:
                    init_x, init_y = 0.0, 0.0
                self.ms_ref_shift = nn.Parameter(torch.tensor([init_x, init_y], dtype=torch.float32))
                self.ms_ref_shift_scale = float(decoder_ms_fusion_ref_shift_scale)
                if bool(freeze_decoder_ms_fusion_ref_shift):
                    self.ms_ref_shift.requires_grad = False
            else:
                self.ms_ref_shift = None
                self.ms_ref_shift_scale = 0.0
        else:
            self.ms_cross_attn = None
            self.fusion_alpha = None
            self.fusion_alpha_scale = 0.0
            self.ms_ref_shift_enabled = False
            self.ms_ref_shift = None
            self.ms_ref_shift_scale = 0.0

        # 运行期统计：用于判断 MSI 分支在 decoder cross-attn 中是否“有幅度/被抵消”。
        # persistent=False 避免进入 checkpoint；仅在训练阶段更新，按 epoch 汇总后由 trainer 读取。
        self.register_buffer("_ms_fusion_sum_rgb_abs", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("_ms_fusion_sum_ms_abs", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("_ms_fusion_sum_ratio", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("_ms_fusion_count", torch.zeros((), dtype=torch.float32), persistent=False)

        self.nhead = ca_nhead

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.group_detr = group_detr

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_sine_embed = None,
                     is_first = False,
                     reference_points = None,
                     spatial_shapes=None,
                     level_start_index=None,
                     ms_memory: Optional[Tensor] = None,
                     ms_memory_key_padding_mask: Optional[Tensor] = None,
                     ):
        bs, num_queries, _ = tgt.shape
        
        # ========== Begin of Self-Attention =============
        # Apply projections here
        # shape: batch_size x num_queries x 256
        q = k = tgt + query_pos
        v = tgt
        if self.training:
            q = torch.cat(q.split(num_queries // self.group_detr, dim=1), dim=0)
            k = torch.cat(k.split(num_queries // self.group_detr, dim=1), dim=0)
            v = torch.cat(v.split(num_queries // self.group_detr, dim=1), dim=0)

        tgt2 = self.self_attn(q, k, v, attn_mask=tgt_mask,
                            key_padding_mask=tgt_key_padding_mask,
                            need_weights=False)[0]
        
        if self.training:
            tgt2 = torch.cat(tgt2.split(bs, dim=0), dim=1)
        # ========== End of Self-Attention =============

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        tgt2_rgb = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            memory,
            spatial_shapes,
            level_start_index,
            memory_key_padding_mask
        )
        if self.decoder_ms_fusion and (ms_memory is not None) and (self.ms_cross_attn is not None) and (self.fusion_alpha is not None):
            ms_reference_points = reference_points
            if self.ms_ref_shift_enabled and (self.ms_ref_shift is not None) and (reference_points is not None):
                # reference_points: [bs, nq, nlevel, 2/4]
                # shift 仅作用于 (x,y) / (cx,cy)，并用 tanh+scale 限幅，避免训练不稳定。
                delta_xy = (self.ms_ref_shift_scale * torch.tanh(self.ms_ref_shift)).to(reference_points.dtype)
                # broadcast to [bs, nq, nlevel, 2]
                delta_xy = delta_xy.view(1, 1, 1, 2)
                ms_reference_points = reference_points.clone()
                if ms_reference_points.shape[-1] >= 2:
                    ms_reference_points[..., :2] = (ms_reference_points[..., :2] + delta_xy).clamp(0.0, 1.0)

            tgt2_ms = self.ms_cross_attn(
                self.with_pos_embed(tgt, query_pos),
                ms_reference_points,
                ms_memory,
                spatial_shapes,
                level_start_index,
                ms_memory_key_padding_mask,
            )
            alpha = (self.fusion_alpha_scale * torch.tanh(self.fusion_alpha())).to(tgt2_rgb.dtype)
            tgt2 = tgt2_rgb + alpha * tgt2_ms

            if self.training:
                rgb_abs = tgt2_rgb.detach().abs().mean().to(torch.float32)
                ms_abs = tgt2_ms.detach().abs().mean().to(torch.float32)
                ratio = (ms_abs / (rgb_abs + 1e-6)).to(torch.float32)
                self._ms_fusion_sum_rgb_abs += rgb_abs
                self._ms_fusion_sum_ms_abs += ms_abs
                self._ms_fusion_sum_ratio += ratio
                self._ms_fusion_count += 1.0
        else:
            tgt2 = tgt2_rgb
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = (tgt + self.dropout3(tgt2))
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_sine_embed = None,
                is_first = False,
                reference_points = None,
                spatial_shapes=None,
                level_start_index=None,
                ms_memory: Optional[Tensor] = None,
                ms_memory_key_padding_mask: Optional[Tensor] = None):
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, 
                                 query_sine_embed, is_first,
                                 reference_points, spatial_shapes, level_start_index,
                                 ms_memory=ms_memory,
                                 ms_memory_key_padding_mask=ms_memory_key_padding_mask)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    
    try:
        two_stage = args.two_stage
    except:
        two_stage = False

    return Transformer(
        d_model=args.hidden_dim,
        sa_nhead=args.sa_nheads,
        ca_nhead=args.ca_nheads,
        num_queries=args.num_queries,
        dropout=args.dropout,
        dim_feedforward=args.dim_feedforward,
        num_decoder_layers=args.dec_layers,
        return_intermediate_dec=True,
        group_detr=args.group_detr,
        two_stage=two_stage,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        lite_refpoint_refine=args.lite_refpoint_refine,
        decoder_norm_type=args.decoder_norm,
        bbox_reparam=args.bbox_reparam,
        decoder_ms_fusion=bool(getattr(args, "decoder_ms_fusion", False)),
        decoder_ms_fusion_alpha_init=float(getattr(args, "decoder_ms_fusion_alpha_init", 0.0)),
        decoder_ms_fusion_alpha_scale=float(getattr(args, "decoder_ms_fusion_alpha_scale", 1.0)),
        freeze_decoder_ms_fusion_alpha=bool(getattr(args, "freeze_decoder_ms_fusion_alpha", False)),
        decoder_ms_fusion_ref_shift=bool(getattr(args, "decoder_ms_fusion_ref_shift", False)),
        decoder_ms_fusion_ref_shift_init=getattr(args, "decoder_ms_fusion_ref_shift_init", (0.0, 0.0)),
        decoder_ms_fusion_ref_shift_scale=float(getattr(args, "decoder_ms_fusion_ref_shift_scale", 0.02)),
        freeze_decoder_ms_fusion_ref_shift=bool(getattr(args, "freeze_decoder_ms_fusion_ref_shift", False)),
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
