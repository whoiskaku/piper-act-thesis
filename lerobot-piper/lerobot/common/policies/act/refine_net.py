"""RefineNet — context-aware residual refiner on top of frozen ACT.

Per RESEARCH_PLAN_v4.3 §4 (T3 final spec) and Principle 2 (ACT frozen).

Inputs (Piper 7-dim convention):
    A_target / ACT_output   (B, chunk_size=100, action_dim=7)
    A_prev / history_action (B, history_max_length=600, action_dim=7)
    I_prev / history_imu    (B, history_max_length=600, imu_dim=6)
    current_state            (B, state_dim=7)  — z-scored externally

Output:
    delta   (B, 100, 7)  — residual modification (initialized near 0)
    refined = ACT_output + delta

Architecture:
    - A_target tokenizer:   per-step Linear (preserves 100 output alignment)
    - history tokenizers:   1D-CNN patchify 0.5s/token (kernel=stride=15)
    - state tokenizer:      Linear, 1 token
    - per-branch position embedding + 4-class type embedding
    - 2-layer transformer encoder, 1-layer decoder (dim=256, 8 heads, ffn=512)
    - 100 learnable query embeddings
    - delta head: Linear(256, 7), zero-initialized so refined ≈ ACT_output at init

Approx 2M parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RefineNet(nn.Module):
    def __init__(
        self,
        chunk_size: int = 100,
        action_dim: int = 7,
        imu_dim: int = 6,
        state_dim: int = 7,
        history_max_length: int = 600,
        patch_size: int = 15,
        dim: int = 256,
        n_encoder_layers: int = 2,
        n_decoder_layers: int = 1,
        nhead: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.0,
        use_state: bool = True,
        use_history_action: bool = True,
        use_history_imu: bool = True,
    ) -> None:
        super().__init__()

        assert history_max_length % patch_size == 0, (
            f"history_max_length ({history_max_length}) must be divisible by "
            f"patch_size ({patch_size})"
        )
        n_patch_tokens = history_max_length // patch_size
        # 输入开关(S0–S3 消融):A_target 永远在；其余三路可关。
        self.use_state = use_state
        self.use_history_action = use_history_action
        self.use_history_imu = use_history_imu

        # ---- Tokenizers (条件构建) ----
        # A_target: per-step Linear so output alignment is per timestep. (always on)
        self.act_proj = nn.Linear(action_dim, dim)

        if use_history_action:
            self.hist_action_tokenizer = nn.Sequential(
                nn.Conv1d(action_dim, 64, kernel_size=patch_size, stride=patch_size),
                nn.GELU(),
                nn.Conv1d(64, dim, kernel_size=1),
            )
            self.pe_a_prev = nn.Embedding(n_patch_tokens, dim)
        if use_history_imu:
            self.hist_imu_tokenizer = nn.Sequential(
                nn.Conv1d(imu_dim, 64, kernel_size=patch_size, stride=patch_size),
                nn.GELU(),
                nn.Conv1d(64, dim, kernel_size=1),
            )
            self.pe_i_prev = nn.Embedding(n_patch_tokens, dim)
        if use_state:
            self.state_proj = nn.Linear(state_dim, dim)
            self.pe_state = nn.Embedding(1, dim)

        # ---- Position + type embeddings ----
        self.pe_a_target = nn.Embedding(chunk_size, dim)
        self.type_embed  = nn.Embedding(4, dim)  # 0:A_target 1:A_prev 2:I_prev 3:state

        # ---- Transformer ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)

        self.query_embed = nn.Embedding(chunk_size, dim)

        # ---- Output head: delta (residual) ----
        self.delta_head = nn.Linear(dim, action_dim)
        # Zero-init so at training start refined ≈ ACT_output (safe identity).
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

        # ---- Cached metadata ----
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.imu_dim = imu_dim
        self.history_max_length = history_max_length
        self.patch_size = patch_size
        self.n_patch_tokens = n_patch_tokens
        self.dim = dim

    @property
    def n_encoder_tokens(self) -> int:
        n = self.chunk_size
        if self.use_history_action: n += self.n_patch_tokens
        if self.use_history_imu:    n += self.n_patch_tokens
        if self.use_state:          n += 1
        return n

    def forward(
        self,
        act_output: torch.Tensor,
        history_action: torch.Tensor,
        history_imu: torch.Tensor,
        current_state: torch.Tensor,
        history_pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            act_output:       (B, chunk_size, action_dim)
            history_action:   (B, history_max_length, action_dim)
            history_imu:      (B, history_max_length, imu_dim)
            current_state:    (B, state_dim) — already z-scored externally
            history_pad_mask: (B, history_max_length) bool, True = pad

        Returns:
            refined: (B, chunk_size, action_dim) = act_output + delta
            delta:   (B, chunk_size, action_dim) residual
        """
        B = act_output.shape[0]
        device = act_output.device

        # ---- Tokenize each branch (按开关条件组装) ----
        a_tgt = self.act_proj(act_output)
        a_tgt = a_tgt + self.pe_a_target.weight.unsqueeze(0) + self.type_embed.weight[0]
        toks = [a_tgt]                                  # A_target 永远在
        mask_segs = [torch.zeros(B, self.chunk_size, dtype=torch.bool, device=device)]

        patch_mask = None
        if (self.use_history_action or self.use_history_imu) and history_pad_mask is not None:
            # 600 帧 mask 下采样到 patch:patch 内 15 帧全 pad 才算 pad。
            patch_mask = history_pad_mask.view(
                B, self.n_patch_tokens, self.patch_size).all(dim=-1)

        if self.use_history_action:
            ha = self.hist_action_tokenizer(history_action.transpose(1, 2)).transpose(1, 2)
            ha = ha + self.pe_a_prev.weight.unsqueeze(0) + self.type_embed.weight[1]
            toks.append(ha)
            mask_segs.append(patch_mask if patch_mask is not None
                             else torch.zeros(B, self.n_patch_tokens, dtype=torch.bool, device=device))
        if self.use_history_imu:
            hi = self.hist_imu_tokenizer(history_imu.transpose(1, 2)).transpose(1, 2)
            hi = hi + self.pe_i_prev.weight.unsqueeze(0) + self.type_embed.weight[2]
            toks.append(hi)
            mask_segs.append(patch_mask if patch_mask is not None
                             else torch.zeros(B, self.n_patch_tokens, dtype=torch.bool, device=device))
        if self.use_state:
            s = self.state_proj(current_state).unsqueeze(1)
            s = s + self.pe_state.weight.unsqueeze(0) + self.type_embed.weight[3]
            toks.append(s)
            mask_segs.append(torch.zeros(B, 1, dtype=torch.bool, device=device))

        memory_tokens = torch.cat(toks, dim=1)
        # 只有存在 history 且给了 pad mask 才需要 key_padding_mask
        kpm = torch.cat(mask_segs, dim=1) if patch_mask is not None else None

        # ---- Encoder ----
        memory = self.encoder(memory_tokens, src_key_padding_mask=kpm)

        # ---- Decoder ----
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        out = self.decoder(queries, memory, memory_key_padding_mask=kpm)

        # ---- Delta head + residual ----
        delta = self.delta_head(out)
        refined = act_output + delta
        return refined, delta

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
