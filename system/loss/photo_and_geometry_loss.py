import torch

from system.loss.ssim import SSIM
from system.utils import inverse_warp

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
compute_ssim_loss = SSIM().to(device)

def photo_and_geometry_loss(inputs, outputs,hparams):
    diff_img_list = []
    diff_color_list = []
    diff_depth_list = []
    valid_mask_list = []
    for f_i in hparams.frame_ids[1:]:
        diff_img_tmp1, diff_color_tmp1, diff_depth_tmp1, valid_mask_tmp1 = compute_pairwise_loss(
            inputs['color',0],inputs['color',f_i],
            outputs['depth',0],outputs['depth',f_i],
            pose=outputs['pose',0,f_i],intrinsic=inputs['K'],
            hparams=hparams
        )
        diff_img_tmp2, diff_color_tmp2, diff_depth_tmp2, valid_mask_tmp2 = compute_pairwise_loss(
            inputs['color', f_i], inputs['color', 0],
            outputs['depth', f_i], outputs['depth', 0],
            pose=outputs['pose',  f_i, 0], intrinsic=inputs['K'],
            hparams=hparams
        )
        diff_img_list += [diff_img_tmp1, diff_img_tmp2]
        diff_color_list += [diff_color_tmp1, diff_color_tmp2]
        diff_depth_list += [diff_depth_tmp1, diff_depth_tmp2]
        valid_mask_list += [valid_mask_tmp1, valid_mask_tmp2]

    diff_img = torch.cat(diff_img_list, dim=1)
    diff_color = torch.cat(diff_color_list, dim=1)
    diff_depth = torch.cat(diff_depth_list, dim=1)
    valid_mask = torch.cat(valid_mask_list, dim=1)
    if not hparams.no_min_optimize:
        indices = torch.argmin(diff_color, dim=1, keepdim=True)

        diff_img = torch.gather(diff_img, 1, indices)
        diff_depth = torch.gather(diff_depth, 1, indices)
        valid_mask = torch.gather(valid_mask, 1, indices)
    photo_loss = mean_on_mask(diff_img, valid_mask)
    geometry_loss = mean_on_mask(diff_depth, valid_mask)
    return photo_loss, geometry_loss

def compute_pairwise_loss(tgt_img,ref_img,tgt_depth,ref_depth,pose,intrinsic,hparams):
    ref_img_warped,projected_depth,computed_depth = inverse_warp(
        ref_img,tgt_depth,ref_depth,pose,intrinsic,padding_mode='zeros'
    )
    diff_depth = (computed_depth-projected_depth).abs() / \
        (computed_depth+projected_depth)

    valid_mask_ref = (
            ref_img_warped.abs().mean(
                dim=1, keepdim=True) > 1e-3).float()
    valid_mask_tgt = (tgt_img.abs().mean(dim=1, keepdim=True) > 1e-3).float()

    valid_mask = valid_mask_tgt * valid_mask_ref

    diff_color = (tgt_img - ref_img_warped).abs().mean(dim=1, keepdim=True)
    if not hparams.no_auto_mask:
        identity_warp_err = (tgt_img - ref_img).abs().mean(dim=1, keepdim=True)
        auto_mask = (diff_color < identity_warp_err).float()
        valid_mask = auto_mask * valid_mask

    diff_img = (tgt_img - ref_img_warped).abs().clamp(0, 1)
    if not hparams.no_ssim:
        ssim_map = compute_ssim_loss(tgt_img, ref_img_warped)
        diff_img = (0.15 * diff_img + 0.85 * ssim_map)
    diff_img = torch.mean(diff_img, dim=1, keepdim=True)

    # reduce photometric loss weight for dynamic regions
    if not hparams.no_dynamic_mask:
        weight_mask = (1 - diff_depth).detach()
        diff_img = diff_img * weight_mask

    return diff_img, diff_color, diff_depth, valid_mask

# compute mean value on a binary mask
def mean_on_mask(diff, valid_mask):
    mask = valid_mask.expand_as(diff)
    if mask.sum() > 100:
        mean_value = (diff * mask).sum() / mask.sum()
    else:
        mean_value = torch.tensor(0).float().to(device)
    return mean_value