import torch
import torch.nn.functional as F
import copy
import comfy
from comfy.ldm.modules.attention import optimized_attention


def get_masks_from_q(masks, q, original_shape):
    if original_shape[2] * original_shape[3] == q.shape[1]:
        down_sample_rate = 1
    elif (original_shape[2] // 2) * (original_shape[3] // 2) == q.shape[1]:
        down_sample_rate = 2
    elif (original_shape[2] // 4) * (original_shape[3] // 4) == q.shape[1]:
        down_sample_rate = 4
    else:
        down_sample_rate = 8

    ret_masks = []
    for mask in masks:
        if isinstance(mask, torch.Tensor):
            size = (
                original_shape[2] // down_sample_rate,
                original_shape[3] // down_sample_rate,
            )
            mask_downsample = F.interpolate(mask.unsqueeze(0), size=size, mode="nearest")
            mask_downsample = mask_downsample.view(1, -1, 1).repeat(
                q.shape[0], 1, q.shape[2]
            )
            ret_masks.append(mask_downsample)
        else:  # no coupling
            ret_masks.append(torch.ones_like(q))

    ret_masks = torch.cat(ret_masks, dim=0)
    return ret_masks


def set_model_patch_replace(model, patch, key):
    to = model.model_options["transformer_options"]
    to.setdefault("patches_replace", {}).setdefault("attn2", {})[key] = patch


class AttentionCouple:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "mode": (["Attention", "Latent"],),
                "isolation_factor": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    FUNCTION = "attention_couple"
    CATEGORY = "loaders"

    # --------------------------------------------------------------------- #
    # Small utility: repeat or truncate a prompt so all prompts share the
    # same token length.  This fixes the cat-mismatch bug.
    # --------------------------------------------------------------------- #
    @staticmethod
    def _match_length(cond: torch.Tensor, target_len: int) -> torch.Tensor:
        """Return `cond` with `seq_len == target_len` by repeating and/or slice."""
        cur_len = cond.shape[1]
        if cur_len == target_len:
            return cond
        # How many full repeats do we need?
        repeat_factor = target_len // cur_len
        if target_len % cur_len == 0:
            return cond.repeat(1, repeat_factor, 1)
        # Repeat one extra time then slice
        return cond.repeat(1, repeat_factor + 1, 1)[:, :target_len, :]

    # --------------------------------------------------------------------- #

    def attention_couple(self, model, positive, negative, mode, isolation_factor):
        if mode == "Latent":
            return model, positive, negative  # nothing to do

        self.negative_positive_masks = []
        self.negative_positive_conds = []
        self.isolation_factor = isolation_factor

        new_positive = copy.deepcopy(positive)
        new_negative = copy.deepcopy(negative)

        dtype = model.model.diffusion_model.dtype
        device = comfy.model_management.get_torch_device()

        # Collect masks and conds
        for conditions in [new_negative, new_positive]:
            conditions_masks = []
            conditions_conds = []
            if len(conditions) != 1:
                mask_norm = torch.stack(
                    [
                        cond[1]["mask"].to(device, dtype=dtype) * cond[1]["mask_strength"]
                        for cond in conditions
                    ]
                )
                mask_norm = mask_norm / mask_norm.sum(dim=0)
                conditions_masks.extend([mask_norm[i] for i in range(mask_norm.shape[0])])
                conditions_conds.extend([cond[0].to(device, dtype=dtype) for cond in conditions])

                # remove mask info for latent-couple fallback
                del conditions[0][1]["mask"]
                del conditions[0][1]["mask_strength"]
            else:
                conditions_masks = [False]
                conditions_conds = [conditions[0][0].to(device, dtype=dtype)]

            self.negative_positive_masks.append(conditions_masks)
            self.negative_positive_conds.append(conditions_conds)

        self.conditioning_length = (len(new_negative), len(new_positive))

        # ----------------------------------------------------------------- #
        # Patch cross-attn blocks
        # ----------------------------------------------------------------- #
        new_model = model.clone()
        self.sdxl = hasattr(new_model.model.diffusion_model, "label_emb")

        if not self.sdxl:
            for idx in [1, 2, 4, 5, 7, 8]:  # input_blocks with cross attention
                set_model_patch_replace(
                    new_model,
                    self.make_patch(
                        new_model.model.diffusion_model.input_blocks[idx][1].transformer_blocks[
                            0
                        ].attn2
                    ),
                    ("input", idx),
                )
            set_model_patch_replace(
                new_model,
                self.make_patch(
                    new_model.model.diffusion_model.middle_block[1].transformer_blocks[0].attn2
                ),
                ("middle", 0),
            )
            for idx in [3, 4, 5, 6, 7, 8, 9, 10, 11]:
                set_model_patch_replace(
                    new_model,
                    self.make_patch(
                        new_model.model.diffusion_model.output_blocks[idx][1].transformer_blocks[
                            0
                        ].attn2
                    ),
                    ("output", idx),
                )
        else:  # SDXL-type model
            for idx in [4, 5, 7, 8]:
                depth_range = range(2) if idx in (4, 5) else range(10)
                for j in depth_range:
                    set_model_patch_replace(
                        new_model,
                        self.make_patch(
                            new_model.model.diffusion_model.input_blocks[idx][1]
                            .transformer_blocks[j]
                            .attn2
                        ),
                        ("input", idx, j),
                    )
            for j in range(10):
                set_model_patch_replace(
                    new_model,
                    self.make_patch(
                        new_model.model.diffusion_model.middle_block[1].transformer_blocks[j].attn2
                    ),
                    ("middle", 0, j),
                )
            for idx in range(6):
                depth_range = range(2) if idx in (3, 4, 5) else range(10)
                for j in depth_range:
                    set_model_patch_replace(
                        new_model,
                        self.make_patch(
                            new_model.model.diffusion_model.output_blocks[idx][1]
                            .transformer_blocks[j]
                            .attn2
                        ),
                        ("output", idx, j),
                    )

        return new_model, [new_positive[0]], [new_negative[0]]

    # --------------------------------------------------------------------- #
    # Patch function injected into each cross-attn block
    # --------------------------------------------------------------------- #
    def make_patch(self, module):
        def patch(q, k, v, extra_opts):
            len_neg, len_pos = self.conditioning_length
            cond_flags = extra_opts["cond_or_uncond"]  # 0→cond, 1→uncond
            q_list = q.chunk(len(cond_flags), dim=0)
            batch  = q_list[0].shape[0]

            masks_uncond = get_masks_from_q(
                self.negative_positive_masks[0], q_list[0], extra_opts["original_shape"]
            )
            masks_cond = get_masks_from_q(
                self.negative_positive_masks[1], q_list[0], extra_opts["original_shape"]
            )

            # -------- FIX: equalise prompt lengths per group -------- #
            max_len_uncond = max(c.shape[1] for c in self.negative_positive_conds[0])
            context_uncond = torch.cat(
                [self._match_length(c, max_len_uncond) for c in self.negative_positive_conds[0]],
                dim=0,
            )

            max_len_cond = max(c.shape[1] for c in self.negative_positive_conds[1])
            context_cond = torch.cat(
                [self._match_length(c, max_len_cond) for c in self.negative_positive_conds[1]],
                dim=0,
            )
            # --------------------------------------------------------- #

            k_uncond = module.to_k(context_uncond)
            v_uncond = module.to_v(context_uncond)
            k_cond   = module.to_k(context_cond)
            v_cond   = module.to_v(context_cond)

            outputs = []
            for idx, flag in enumerate(cond_flags):
                if flag == 0:
                    masks = masks_cond
                    k_src = k_cond
                    v_src = v_cond
                    grp_len = len_pos
                else:
                    masks = masks_uncond
                    k_src = k_uncond
                    v_src = v_uncond
                    grp_len = len_neg

                q_tgt = q_list[idx].repeat(grp_len, 1, 1)
                k_rep = torch.cat(
                    [k_src[i].unsqueeze(0).repeat(batch, 1, 1) for i in range(grp_len)],
                    dim=0,
                )
                v_rep = torch.cat(
                    [v_src[i].unsqueeze(0).repeat(batch, 1, 1) for i in range(grp_len)],
                    dim=0,
                )

                k_rep, v_rep, masks = (
                    k_rep.to(q_tgt.dtype),
                    v_rep.to(q_tgt.dtype),
                    masks.to(q_tgt.dtype),
                )

                sharpened = self.sharpen_masks(masks, self.isolation_factor)
                attn_out  = optimized_attention(q_tgt, k_rep, v_rep, extra_opts["n_heads"])
                attn_out  = attn_out * sharpened
                attn_out  = attn_out.view(
                    grp_len, batch, -1, module.heads * module.dim_head
                ).sum(dim=0)

                outputs.append(attn_out)

            return torch.cat(outputs, dim=0)

        return patch

    # --------------------------------------------------------------------- #
    # Mask sharpening (unchanged)
    # --------------------------------------------------------------------- #
    @staticmethod
    def sharpen_masks(masks, isolation_factor):
        iso = torch.tensor(isolation_factor, device=masks.device, dtype=masks.dtype)
        masks = torch.pow(masks, torch.exp(iso))
        return masks / (masks.sum(dim=0, keepdim=True) + 1e-6)


NODE_CLASS_MAPPINGS = {"Attention couple": AttentionCouple}
NODE_DISPLAY_NAME_MAPPINGS = {"Attention couple": "Load Attention couple"}
