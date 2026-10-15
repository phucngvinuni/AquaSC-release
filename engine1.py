# engine1.py
import torch
import math
import torch.nn as nn
import sys
import numpy as np
from typing import Iterable, Dict, Tuple, List, Optional
import os

from timm.utils import AverageMeter
import utils 
from torchvision.utils import save_image

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False


try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


try:
    from torchmetrics.detection import MeanAveragePrecision
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    # print("Warning: torchmetrics not found. mAP calculation will be skipped. `pip install torchmetrics`")


# --- QUANTIZER DIAGNOSTICS PRINTING (flat single-codebook and per-stage RVQ) ---
def _print_quantizer_diagnostics(quantizer, qname: str, prefix: str = ""):
    """Print codebook utilization/perplexity/dead/entropy, one line per RVQ stage."""
    if hasattr(quantizer, 'num_stages'):
        d = quantizer.get_diagnostics()
        eff_rate = 0.0
        for st in d['stages']:
            print(f"  {prefix} codebook[{qname}/stage{st['stage'] + 1}]: "
                  f"utilization={st['utilization']:.4f}, perplexity={st['perplexity']:.2f}, "
                  f"dead={st['dead_codewords']}, entropy={st['entropy_bits']:.2f} bits")
            eff_rate += st['entropy_bits']
        print(f"  {prefix} codebook[{qname}]: effective_rate_entropy = {eff_rate:.2f} bits/token")
    else:
        d = quantizer.get_diagnostics()
        print(f"  {prefix} codebook[{qname}]: "
              f"utilization={d['utilization']:.4f}, perplexity={d['perplexity']:.2f}, "
              f"dead={d['dead_codewords']}, entropy={d['entropy_bits']:.2f} bits")


def _effective_box_weights(args):
    """[ABLATION] Reconstruction-loss box weights.

    When --disable_bbw is set, returns (1.0, 1.0) so the reconstruction loss is
    plain per-pixel L1 (no box weighting). Otherwise returns the configured
    (inside_box_loss_weight, outside_box_loss_weight)."""
    if getattr(args, 'disable_bbw', False):
        return 1.0, 1.0
    return args.inside_box_loss_weight, args.outside_box_loss_weight


# --- EVALUATE FUNCTION ---
@torch.no_grad()
def evaluate_semcom_with_yolo(
    semcom_net: torch.nn.Module,
    yolo_model: Optional[torch.nn.Module],
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    reconstruction_criterion: torch.nn.Module,
    fim_criterion: Optional[torch.nn.Module], 
    lpips_criterion: Optional[nn.Module], 
    args,
    current_epoch_num: any,
    viz_output_dir: Optional[str] = None,
    print_freq=20,
    visualize_batches=1,
    visualize_images_per_batch=1,
    save_recons_dir: Optional[str] = None
):
    semcom_net.eval()
    if yolo_model: yolo_model.eval()
    if hasattr(semcom_net, 'quantizer_high'):
        semcom_net.quantizer_high.reset_diagnostics()
        semcom_net.quantizer_low.reset_diagnostics()

    main_rec_loss_meter_eval = AverageMeter()
    vq_loss_meter_eval = AverageMeter()
    fim_loss_meter_eval = AverageMeter()
    lpips_loss_meter_eval = AverageMeter() 
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    map_calculator = None
    if yolo_model and TORCHMETRICS_AVAILABLE:
        try:
          
            map_calculator = MeanAveragePrecision(iou_type="bbox", class_metrics=True) 
        except Exception as e_map:
            print(f"Error initializing MeanAveragePrecision: {e_map}. mAP calculation will be skipped.")
            map_calculator = None
        if map_calculator: map_calculator = map_calculator.to(device)
    
    if viz_output_dir: os.makedirs(viz_output_dir, exist_ok=True)
    print(f"\n--- Starting Evaluation (Epoch/Run: {current_epoch_num}, Eval SNR: {args.snr_db_eval:.1f} dB) ---")

    digital_ldpc = getattr(args, 'digital_ldpc', False)
    digital_csi_mode = getattr(args, 'digital_csi_mode', 'perfect')
    digital_bits_per_symbol = getattr(args, 'digital_bits_per_symbol', 4)
    digital_payload_rate = getattr(args, 'digital_payload_rate', 0.5)
    prior_floor = getattr(args, 'prior_floor_eval', False)
    digital_stat_agg = {'blocks': 0, 'failed': 0, 'erased_tokens': 0, 'bit_accuracy': 0.0,
                        'ber': 0.0, 'iterations_mean': 0.0, 'n_images': 0,
                        'channel_uses_per_px': 0.0, 'n_uses_batches': 0}
    per_image_ap_out = getattr(args, 'per_image_ap_out', None)
    per_image_ap_list = [] if per_image_ap_out else None
    per_image_npred_list = [] if per_image_ap_out else None

    for batch_idx, (semcom_data_input, targets_tuple) in enumerate(dataloader):
        (original_imgs_tensor, bm_pos) = semcom_data_input
        (semcom_recon_target_gt, yolo_gt_targets_list_of_dicts, fim_target_map_batch_eval) = targets_tuple

        original_imgs_tensor = original_imgs_tensor.to(device, non_blocking=True)
        if bm_pos is not None: bm_pos = bm_pos.to(device, non_blocking=True)
        semcom_recon_target_gt = semcom_recon_target_gt.to(device, non_blocking=True)
        fim_target_map_batch_eval = fim_target_map_batch_eval.to(device, non_blocking=True)

        yolo_gt_for_metric_device = []
        if yolo_model and map_calculator:
            for gt_dict in yolo_gt_targets_list_of_dicts:
                yolo_gt_for_metric_device.append({
                    "boxes": gt_dict["boxes"].to(device), "labels": gt_dict["labels"].to(device)
                })
        
        # Plan v7.4 (bug-class sweep): digital_seed must be batch-size-invariant.
        # This was previously args.seed + batch_idx, which is only a per-image-
        # unique seed because run_class_main.py's eval dataloader hardcodes
        # batch_size=1 (run_class_main.py:221) -- correct by coincidence, not by
        # formula. transfer_eval.py made the identical mistake with its own
        # batch_size=8 loader and produced a real, measured defect (RESULTS_OF_RECORD.md
        # §6d-3). Using original_imgs_tensor.shape[0] (the true batch size actually
        # in hand here, not args.batch_size, which the batch_size=1-hardcoded caller
        # never even sets to match) makes this a no-op today (batch_size=1: 1*batch_idx
        # == batch_idx) and permanently removes the bug class regardless of what any
        # future caller's batch size is.
        digital_seed = args.seed + batch_idx * original_imgs_tensor.shape[0]
        outputs_dict = semcom_net(
            img=original_imgs_tensor, bm_pos=bm_pos, _eval=True, # Removed targets=None
            eval_snr_db=args.snr_db_eval,
            digital_ldpc=digital_ldpc, digital_csi_mode=digital_csi_mode,
            digital_seed=digital_seed,
            digital_bits_per_symbol=digital_bits_per_symbol,
            digital_payload_rate=digital_payload_rate,
            digital_joint_frame=getattr(args, 'digital_joint_frame', False),
            prior_floor=prior_floor,
        )
        reconstructed_image_batch = outputs_dict['reconstructed_image']

        if digital_ldpc and 'digital_stats' in outputs_dict:
            st = outputs_dict['digital_stats']
            digital_stat_agg['blocks'] += int(st.get('blocks', 0))
            digital_stat_agg['failed'] += int(st.get('failed', 0))
            digital_stat_agg['erased_tokens'] += int(st.get('erased_tokens', 0))
            digital_stat_agg['bit_accuracy'] += float(st.get('bit_accuracy', 0.0))
            digital_stat_agg['ber'] += float(st.get('ber', 0.0))
            digital_stat_agg['iterations_mean'] += float(st.get('iterations_mean', 0.0)) * int(st.get('blocks', 0))
            digital_stat_agg['n_images'] += 1
            if 'channel_uses_per_px' in st:
                digital_stat_agg['channel_uses_per_px'] += float(st['channel_uses_per_px'])
                digital_stat_agg['n_uses_batches'] += 1

        batch_size_eval = reconstructed_image_batch.size(0)

        if save_recons_dir:
            os.makedirs(save_recons_dir, exist_ok=True)
            for i in range(batch_size_eval):
                torch.save({
                    'orig': semcom_recon_target_gt[i].detach().cpu(),
                    'recon': reconstructed_image_batch[i].detach().cpu(),
                    'boxes': yolo_gt_targets_list_of_dicts[i]['boxes'].detach().cpu(),
                    'labels': yolo_gt_targets_list_of_dicts[i]['labels'].detach().cpu(),
                }, os.path.join(save_recons_dir,
                                f'img_{batch_idx * batch_size_eval + i:05d}.pt'))

        current_device_eval = reconstructed_image_batch.device
        total_weighted_rec_loss_eval = torch.tensor(0.0, device=current_device_eval)
        in_w_eval, out_w_eval = _effective_box_weights(args)
        for i in range(batch_size_eval):
            rec_img_single = reconstructed_image_batch[i]
            orig_img_single = semcom_recon_target_gt[i]
            gt_boxes_single = yolo_gt_targets_list_of_dicts[i]['boxes']
            
            weight_map = utils.create_bbox_weight_map(
                rec_img_single.shape[1:], gt_boxes_single.to(current_device_eval),
                in_w_eval, out_w_eval, current_device_eval
            )
            pixel_loss = reconstruction_criterion(rec_img_single, orig_img_single)
            weighted_loss = (pixel_loss * weight_map.unsqueeze(0)).mean()
            total_weighted_rec_loss_eval += weighted_loss
        
        avg_weighted_rec_loss_eval = total_weighted_rec_loss_eval / batch_size_eval if batch_size_eval > 0 else torch.tensor(0.0)
        main_rec_loss_meter_eval.update(avg_weighted_rec_loss_eval.item(), batch_size_eval)

        current_eval_vq_loss = outputs_dict.get('vq_loss', torch.tensor(0.0)).item()
        vq_loss_meter_eval.update(current_eval_vq_loss, batch_size_eval)
        
        current_eval_fim_loss = 0.0
        if fim_criterion and 'fim_importance_scores' in outputs_dict:
            fim_preds_eval_logits = outputs_dict['fim_importance_scores']
    
            if args.mask_ratio == 0.0: 
                target_fim_map = fim_target_map_batch_eval
            else: 
                if bm_pos is not None:
                    flat_fim_targets = fim_target_map_batch_eval.reshape(-1, 1)
                    flat_bm_pos_encoder_mask = bm_pos.reshape(-1)
                    target_fim_map = flat_fim_targets[~flat_bm_pos_encoder_mask].reshape(batch_size_eval, -1, 1)
                else: 
                    target_fim_map = fim_target_map_batch_eval 

            if fim_preds_eval_logits.shape == target_fim_map.shape:
                 with torch.amp.autocast(device_type=args.device, enabled=False):
                    fim_loss_eval = fim_criterion(fim_preds_eval_logits.float(), target_fim_map.to(fim_preds_eval_logits.dtype).float())
                    current_eval_fim_loss = fim_loss_eval.item()


        fim_loss_meter_eval.update(current_eval_fim_loss, batch_size_eval)

        current_eval_lpips_loss = 0.0
        if lpips_criterion and hasattr(args, 'lpips_loss_weight') and args.lpips_loss_weight > 0:
            with torch.amp.autocast(device_type=args.device, enabled=False): # LPIPS often prefers FP32
                recon_for_lpips_eval = (reconstructed_image_batch.float() * 2.0) - 1.0
                orig_for_lpips_eval = (semcom_recon_target_gt.float() * 2.0) - 1.0
                lpips_val_eval = lpips_criterion(recon_for_lpips_eval, orig_for_lpips_eval).mean()
                current_eval_lpips_loss = lpips_val_eval.item()
        lpips_loss_meter_eval.update(current_eval_lpips_loss, batch_size_eval)


        batch_psnr = utils.calc_psnr(reconstructed_image_batch.detach().cpu(), semcom_recon_target_gt.detach().cpu())
        batch_ssim = utils.calc_ssim(reconstructed_image_batch.detach().cpu(), semcom_recon_target_gt.detach().cpu())
        psnr_meter.update(np.mean(batch_psnr) if batch_psnr else 0.0, batch_size_eval)
        ssim_meter.update(np.mean(batch_ssim) if batch_ssim else 0.0, batch_size_eval)

        if batch_idx < visualize_batches and viz_output_dir:

            for i in range(min(batch_size_eval, visualize_images_per_batch)):
                save_image(reconstructed_image_batch[i].cpu(), os.path.join(viz_output_dir, f"ep{current_epoch_num}_b{batch_idx}_i{i}_snr{args.snr_db_eval:.0f}dB_RECON.png"))
                save_image(original_imgs_tensor[i].cpu(), os.path.join(viz_output_dir, f"ep{current_epoch_num}_b{batch_idx}_i{i}_ORIG.png"))
                if 'fim_importance_scores' in outputs_dict and CV2_AVAILABLE:
                    try:
                        fim_logits_viz = outputs_dict['fim_importance_scores'][i].squeeze().detach()
                        fim_scores_viz = torch.sigmoid(fim_logits_viz).cpu().numpy() # Apply sigmoid for visualization
                        num_h_patches = args.input_size // args.patch_size
                        num_w_patches = args.input_size // args.patch_size
                        if fim_scores_viz.size == num_h_patches * num_w_patches :
                            fim_map_2d_viz = fim_scores_viz.reshape(num_h_patches, num_w_patches)
                            fim_map_resized_viz = cv2.resize((fim_map_2d_viz * 255).astype(np.uint8), (args.input_size, args.input_size), interpolation=cv2.INTER_NEAREST)
                            cv2.imwrite(os.path.join(viz_output_dir, f"ep{current_epoch_num}_b{batch_idx}_i{i}_FIM_PRED_MAP.png"), fim_map_resized_viz)
                        
             
                        target_fim_map_viz_item = fim_target_map_batch_eval[i].squeeze().cpu().numpy()
                        if target_fim_map_viz_item.size == num_h_patches * num_w_patches:
                            target_fim_map_2d_viz = target_fim_map_viz_item.reshape(num_h_patches, num_w_patches)
                            target_fim_map_resized_viz = cv2.resize((target_fim_map_2d_viz * 255).astype(np.uint8), (args.input_size, args.input_size), interpolation=cv2.INTER_NEAREST)
                            cv2.imwrite(os.path.join(viz_output_dir, f"ep{current_epoch_num}_b{batch_idx}_i{i}_FIM_TARGET_MAP.png"), target_fim_map_resized_viz)
                    except Exception as e_viz_fim: print(f"Error during FIM map visualization: {e_viz_fim}")


        if yolo_model and map_calculator:
            if batch_idx < visualize_batches and viz_output_dir and CV2_AVAILABLE:
          
                try:
                    yolo_input_orig_eval = original_imgs_tensor.detach()
                    yolo_results_orig_eval = yolo_model(yolo_input_orig_eval, verbose=False, conf=args.yolo_conf_thres, iou=args.yolo_iou_thres)
                    for i_vo in range(min(len(yolo_results_orig_eval), visualize_images_per_batch)):
                        if hasattr(yolo_results_orig_eval[i_vo], 'plot'):
                            cv2.imwrite(os.path.join(viz_output_dir, f"ep{current_epoch_num}_b{batch_idx}_i{i_vo}_YOLO_ON_ORIG.png"), yolo_results_orig_eval[i_vo].plot())
                except Exception as e_yolo_orig: print(f"    Eval Viz Error (YOLO on Original): {e_yolo_orig}")
            
            yolo_preds_for_metric = []
            try:
                yolo_input_recon_eval = reconstructed_image_batch.detach() 

                yolo_results_list_recon_eval = yolo_model(yolo_input_recon_eval, verbose=False, conf=args.map_conf_thres, iou=args.yolo_iou_thres)
                for i_vr, result_r_eval in enumerate(yolo_results_list_recon_eval):
                    if result_r_eval.boxes is not None and len(result_r_eval.boxes.xyxy) > 0:
                        yolo_preds_for_metric.append({
                            "boxes": result_r_eval.boxes.xyxy.to(device), "scores": result_r_eval.boxes.conf.to(device), "labels": result_r_eval.boxes.cls.to(torch.int64).to(device)
                        })
                    else:
                        yolo_preds_for_metric.append({"boxes": torch.empty((0,4), dtype=torch.float32, device=device),"scores": torch.empty(0, dtype=torch.float32, device=device),"labels": torch.empty(0, dtype=torch.int64, device=device)})

                if batch_idx < visualize_batches and viz_output_dir and CV2_AVAILABLE:
                    try:
                        yolo_viz_recon_eval = yolo_model(yolo_input_recon_eval, verbose=False, conf=args.yolo_conf_thres, iou=args.yolo_iou_thres)
                        for i_vr in range(min(len(yolo_viz_recon_eval), visualize_images_per_batch)):
                            if hasattr(yolo_viz_recon_eval[i_vr], 'plot'):
                                cv2.imwrite(os.path.join(viz_output_dir, f"ep{current_epoch_num}_b{batch_idx}_i{i_vr}_snr{args.snr_db_eval:.0f}dB_YOLO_ON_RECON.png"), yolo_viz_recon_eval[i_vr].plot())
                    except Exception as e_yolo_viz: print(f"    Eval Viz Error (YOLO on Recon): {e_yolo_viz}")
            except Exception as e_yolo_recon:
                print(f"    Error during YOLO inference on reconstructed for mAP: {e_yolo_recon}")
                yolo_preds_for_metric = [{"boxes": torch.empty((0,4), dtype=torch.float32, device=device),"scores": torch.empty(0, dtype=torch.float32, device=device),"labels": torch.empty(0, dtype=torch.int64, device=device)} for _ in range(batch_size_eval)]

            if yolo_preds_for_metric and len(yolo_preds_for_metric) == len(yolo_gt_for_metric_device):
                try: map_calculator.update(yolo_preds_for_metric, yolo_gt_for_metric_device)
                except Exception as e_map_update: print(f"    Error updating mAP calculator: {e_map_update}")
                if per_image_ap_list is not None:
                    import stats as _stats
                    for _i in range(len(yolo_preds_for_metric)):
                        per_image_ap_list.append(_stats.per_image_ap50(
                            yolo_preds_for_metric[_i]["boxes"].detach().cpu().numpy(),
                            yolo_preds_for_metric[_i]["scores"].detach().cpu().numpy(),
                            yolo_gt_for_metric_device[_i]["boxes"].detach().cpu().numpy(),
                            iou_thr=0.5))
                        per_image_npred_list.append(int(yolo_preds_for_metric[_i]["boxes"].shape[0]))
            elif yolo_preds_for_metric : print(f"    Warning: Mismatch num YOLO preds ({len(yolo_preds_for_metric)}) vs GTs ({len(yolo_gt_for_metric_device)}). Skipping mAP update batch {batch_idx}.")

        if (batch_idx + 1) % print_freq == 0 or (batch_idx + 1) == len(dataloader):
            print_str = f'Eval Batch {batch_idx+1}/{len(dataloader)}: ' \
                        f'[RecL: {main_rec_loss_meter_eval.avg:.3f}] [VQL: {vq_loss_meter_eval.avg:.3f}] ' \
                        f'[FIM_L: {fim_loss_meter_eval.avg:.3f}] '
            if lpips_criterion: print_str += f'[LPIPS: {lpips_loss_meter_eval.avg:.3f}] '
            print_str += f'[PSNR: {psnr_meter.avg:.2f}] [SSIM: {ssim_meter.avg:.3f}]'
            print(print_str)
            sys.stdout.flush()

    final_stats = {
        'rec_loss': main_rec_loss_meter_eval.avg, 'vq_loss': vq_loss_meter_eval.avg,
        'fim_loss': fim_loss_meter_eval.avg, 'lpips_loss': lpips_loss_meter_eval.avg,
        'psnr': psnr_meter.avg, 'ssim': ssim_meter.avg,
    }
    if digital_ldpc and digital_stat_agg['n_images'] > 0:
        n_imgs = digital_stat_agg['n_images']
        n_blocks = digital_stat_agg['blocks']
        final_stats['digital_blocks'] = n_blocks
        final_stats['digital_failed'] = digital_stat_agg['failed']
        final_stats['digital_bler'] = digital_stat_agg['failed'] / n_blocks if n_blocks else 0.0
        final_stats['digital_ber'] = digital_stat_agg['ber'] / n_imgs
        final_stats['digital_bit_accuracy'] = digital_stat_agg['bit_accuracy'] / n_imgs
        final_stats['digital_erased_tokens'] = digital_stat_agg['erased_tokens']
        final_stats['digital_erased_token_fraction'] = digital_stat_agg['erased_tokens'] / (n_imgs * 784.0)
        final_stats['digital_iterations_mean'] = digital_stat_agg['iterations_mean'] / n_blocks if n_blocks else 0.0
        if digital_stat_agg['n_uses_batches'] > 0:
            final_stats['digital_channel_uses_per_px'] = (
                digital_stat_agg['channel_uses_per_px'] / digital_stat_agg['n_uses_batches'])
    if map_calculator:
        try:
            final_map_results = map_calculator.compute()
            print(f"\nFinal mAP Results (Eval SNR: {args.snr_db_eval:.1f} dB, Epoch/Run: {current_epoch_num}):")
            for k, v_tensor in final_map_results.items():
                v_item = v_tensor.item() if isinstance(v_tensor, torch.Tensor) else float(v_tensor)
                print(f"  {k}: {v_item:.4f}")
                safe_key = str(k).replace("map_per_class","map_cls").replace("(","_").replace(")","").replace("[","").replace("]","").replace(" ","_").replace("'","").replace(":","_")
                final_stats[safe_key] = v_item
            map_calculator.reset()
        except Exception as e:
            import traceback
            print(f"Error computing final mAP: {e}")
            traceback.print_exc()
            final_stats['map'] = 0.0

    if per_image_ap_out and per_image_ap_list is not None:
        np.save(per_image_ap_out, np.array(per_image_ap_list, dtype=np.float32))
        npred_out = per_image_ap_out.rsplit('.npy', 1)[0] + '.npred.npy'
        np.save(npred_out, np.array(per_image_npred_list, dtype=np.int32))
        n_blind = int(np.sum(np.array(per_image_npred_list) == 0))
        print(f"  Saved {len(per_image_ap_list)} per-image AP@0.5 values to {per_image_ap_out}")
        print(f"  Saved {len(per_image_npred_list)} per-image npred values to {npred_out} "
              f"(npred=0 on {n_blind}/{len(per_image_npred_list)} = {100*n_blind/len(per_image_npred_list):.1f}%)")

    if hasattr(semcom_net, 'quantizer_high'):
        for qname, quantizer in (('high', semcom_net.quantizer_high), ('low', semcom_net.quantizer_low)):
            _print_quantizer_diagnostics(quantizer, qname, prefix="(eval)")
            
    print("--- Finished Evaluation ---")
    sys.stdout.flush()
    return final_stats


def train_semcom_reconstruction_batch(
    model: torch.nn.Module,
    input_samples_for_semcom: torch.Tensor,
    original_images_for_loss: torch.Tensor,
    yolo_gt_for_this_batch: List[Dict[str, torch.Tensor]],
    fim_target_importance_map_batch: torch.Tensor,
    bm_pos: Optional[torch.Tensor],
    base_reconstruction_criterion: torch.nn.Module,
    fim_criterion: torch.nn.Module,
    lpips_criterion: Optional[nn.Module],
    args,
    digital_seed: int = 42,
    teacher_model: Optional[torch.nn.Module] = None,
    distill_weight_feat: float = 0.0,
    distill_weight_cos: float = 0.0,
    distill_weight_quant: float = 0.0,
    distill_weight_idx: float = 0.0,
    distill_weight_fim: float = 0.0,
    distill_task_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, float, float, float, float, dict]:
    outputs_dict = model(
        img=input_samples_for_semcom, bm_pos=bm_pos, _eval=False,
        train_snr_db_min=args.snr_db_train_min, train_snr_db_max=args.snr_db_train_max,
        index_flip=getattr(args, 'index_flip', False),
        index_flip_csv=getattr(args, 'index_flip_csv', None),
        train_channel_mode=getattr(args, 'train_channel_mode', 'analog'),
        erasure_bler_csv=getattr(args, 'erasure_bler_csv', None),
        digital_seed=digital_seed
    )
    reconstructed_image_batch = outputs_dict['reconstructed_image']
    fim_predicted_logits_batch = outputs_dict.get('fim_importance_scores')

    batch_size = reconstructed_image_batch.size(0)
    current_device = reconstructed_image_batch.device
    

    total_weighted_rec_loss_tensor = torch.tensor(0.0, device=current_device)
    in_w_train, out_w_train = _effective_box_weights(args)
    for i in range(batch_size):
        reconstructed_img_single = reconstructed_image_batch[i]
        original_img_single = original_images_for_loss[i]
        gt_boxes_single_img_abs = yolo_gt_for_this_batch[i]['boxes']
        weight_map_2d = utils.create_bbox_weight_map(
            reconstructed_img_single.shape[1:], gt_boxes_single_img_abs,
            in_w_train, out_w_train, current_device
        )
        pixel_wise_loss_single_image = base_reconstruction_criterion(reconstructed_img_single, original_img_single)
        weighted_pixel_loss_single_image = (pixel_wise_loss_single_image * weight_map_2d.unsqueeze(0)).mean()
        total_weighted_rec_loss_tensor += weighted_pixel_loss_single_image
    final_reconstruction_loss = total_weighted_rec_loss_tensor / batch_size if batch_size > 0 else torch.tensor(0.0, device=current_device)
    current_rec_loss_val = final_reconstruction_loss.item()

 
    vq_loss_component = torch.tensor(0.0, device=current_device)
    current_vq_loss_val = 0.0
    if 'vq_loss' in outputs_dict and outputs_dict['vq_loss'] is not None:
        vq_loss_tensor = outputs_dict['vq_loss'] 
        if isinstance(vq_loss_tensor, torch.Tensor):
         
            vq_loss_component = args.vq_loss_weight * vq_loss_tensor
            current_vq_loss_val = vq_loss_tensor.item() 

 
    fim_loss_component = torch.tensor(0.0, device=current_device)
    current_fim_loss_val = 0.0
    if fim_predicted_logits_batch is not None:
        target_fim_map_for_loss = fim_target_importance_map_batch # Assuming mask_ratio=0 for now
        if args.mask_ratio > 0 and bm_pos is not None:
             flat_fim_targets = fim_target_importance_map_batch.reshape(-1, 1)
             flat_bm_pos_encoder_mask = bm_pos.reshape(-1)
             target_fim_map_for_loss = flat_fim_targets[~flat_bm_pos_encoder_mask].reshape(batch_size, -1, 1)

        if fim_predicted_logits_batch.shape == target_fim_map_for_loss.shape:
            fim_loss = fim_criterion(fim_predicted_logits_batch, target_fim_map_for_loss.to(fim_predicted_logits_batch.dtype))
            current_fim_loss_val = fim_loss.item()
            if not getattr(args, 'disable_fim', False):
                fim_loss_component = args.fim_loss_weight * fim_loss
     
    perceptual_loss_component = torch.tensor(0.0, device=current_device)
    current_lpips_loss_val = 0.0
    if lpips_criterion is not None and args.lpips_loss_weight > 0:
       
        recon_for_lpips = (reconstructed_image_batch * 2.0) - 1.0
        orig_for_lpips = (original_images_for_loss * 2.0) - 1.0

        with torch.amp.autocast(device_type=args.device, enabled=False): 
             lpips_val = lpips_criterion(recon_for_lpips.float(), orig_for_lpips.float()).mean()
        perceptual_loss_component = args.lpips_loss_weight * lpips_val
        current_lpips_loss_val = lpips_val.item()
            
    # 2026-09-17 encoder-distillation (small-encoder experiment; design per
    # author review): a frozen, larger-encoder teacher forces the student
    # toward the SAME representation the codebook/decoder actually consume,
    # not the raw ViT output -- z = norm(proj(encoder(x))), i.e.
    # x_proj_normalized. Four terms:
    #   L_feat  = MSE(z_S, sg(z_T))                          -- scale/geometry
    #   L_cos   = 1 - cos_sim(z_S, z_T)                       -- direction
    #   L_quant = MSE(q(z_S), sg(q(z_T)))                     -- same codeword,
    #             q = the model's OWN (frozen, warm-started-identical) RVQ
    #             high-path quantizer applied to EVERY token, not just
    #             FIM-routed-high ones -- small feature MSE does not imply
    #             the same nearest-codeword assignment near a Voronoi
    #             boundary, and the decoder only ever sees q(z), never z.
    #   L_idx   = sum_stage CrossEntropy(student_stage_logits, teacher_stage_idx)
    #             -- 2026-09-18 pivot: L_feat/L_quant are only PROXIES for "same
    #             codeword" (a small MSE does not guarantee the same argmin near
    #             a Voronoi boundary -- confirmed empirically: idx_agree stayed
    #             at 3-24% even as feat/quant fell steadily under the MSE-only
    #             loss). L_idx trains the actual discrete objective directly:
    #             per RVQ stage, treat the teacher's own argmin index as a hard
    #             label and cross-entropy the student's own (naturally computed,
    #             not teacher-forced) negative-squared-distance logits against
    #             it. This is what should dominate once index agreement, not
    #             feature proximity, is the goal.
    #   L_FIM_KD = MSE(fim_logits_S, sg(fim_logits_T))        -- keep routing
    #             aligned as the encoder underneath FIM changes.
    # Teacher runs encoder+FIM+quantizer_high under no_grad (no decode needed).
    # NOTE: this calls model.quantizer_high an extra time beyond the model's
    # own forward (which only quantizes FIM-routed-high tokens through it) --
    # usage_counts/last_assigned get a harmless extra bump (diagnostics only;
    # no dead-code restart or EMA update happens while --freeze_codebooks
    # keeps quantizer_high in .eval(), which this experiment always sets).
    distill_loss_component = torch.tensor(0.0, device=current_device)
    distill_diag = {}
    have_teacher = teacher_model is not None
    any_kd_weight = (distill_weight_feat > 0 or distill_weight_cos > 0
                      or distill_weight_quant > 0 or distill_weight_idx > 0
                      or distill_weight_fim > 0)
    if have_teacher and any_kd_weight:
        student_proj = outputs_dict.get('x_proj_normalized')
        student_fim_logits = outputs_dict.get('fim_importance_scores')
        if student_proj is None:
            raise ValueError("teacher_model given but model's forward() did not return "
                              "'x_proj_normalized' -- this quantizer_type is not wired for "
                              "distillation yet (only the RVQ/flat main path returns it).")
        with torch.no_grad():
            B_t = input_samples_for_semcom.shape[0]
            Np_t = teacher_model.img_encoder.patch_embed.num_patches
            teacher_mask = torch.zeros(B_t, Np_t, dtype=torch.bool, device=current_device)
            teacher_tokens = teacher_model.img_encoder(input_samples_for_semcom, teacher_mask)
            teacher_proj = teacher_model.norm_before_quantizer(
                teacher_model.encoder_to_channel_proj(teacher_tokens))
            teacher_fim_logits = teacher_model.fim_module(teacher_tokens)
            teacher_q, _, _, teacher_idx = teacher_model.quantizer_high(
                teacher_proj.reshape(-1, teacher_proj.size(-1)), return_per_token_loss=True)

        if distill_weight_feat > 0:
            l_feat = torch.nn.functional.mse_loss(student_proj, teacher_proj)
            distill_loss_component = distill_loss_component + distill_weight_feat * l_feat
            distill_diag['feat'] = l_feat.item()

        if distill_weight_cos > 0:
            cos_sim = torch.nn.functional.cosine_similarity(
                student_proj.reshape(-1, student_proj.size(-1)),
                teacher_proj.reshape(-1, teacher_proj.size(-1)), dim=-1)
            l_cos = (1.0 - cos_sim).mean()
            distill_loss_component = distill_loss_component + distill_weight_cos * l_cos
            distill_diag['cos'] = l_cos.item()

        if distill_weight_quant > 0 or True:  # index agreement is a free diagnostic either way
            student_q, _, _, student_idx = model.quantizer_high(
                student_proj.reshape(-1, student_proj.size(-1)), return_per_token_loss=True)
            if distill_weight_quant > 0:
                l_quant = torch.nn.functional.mse_loss(student_q, teacher_q)
                distill_loss_component = distill_loss_component + distill_weight_quant * l_quant
                distill_diag['quant'] = l_quant.item()
            with torch.no_grad():
                num_stages = model.quantizer_high.num_stages
                bits = model.quantizer_high.bits_per_stage
                s_stage_idx = model.quantizer_high.unpack_indices(student_idx, num_stages, bits)
                t_stage_idx = model.quantizer_high.unpack_indices(teacher_idx, num_stages, bits)
                for s in range(num_stages):
                    distill_diag[f'idx_agree_stage{s+1}'] = (
                        s_stage_idx[s] == t_stage_idx[s]).float().mean().item()

        if distill_weight_idx > 0:
            # t_stage_idx/num_stages come from the diagnostic block just above
            # (unconditional -- "or True"), so they're already available here.
            student_stage_logits = model.quantizer_high.stage_logits(
                student_proj.reshape(-1, student_proj.size(-1)))
            l_idx = sum(
                torch.nn.functional.cross_entropy(student_stage_logits[s], t_stage_idx[s])
                for s in range(num_stages)
            ) / num_stages
            distill_loss_component = distill_loss_component + distill_weight_idx * l_idx
            distill_diag['idx_ce'] = l_idx.item()

        if distill_weight_fim > 0 and student_fim_logits is not None:
            l_fim_kd = torch.nn.functional.mse_loss(student_fim_logits, teacher_fim_logits)
            distill_loss_component = distill_loss_component + distill_weight_fim * l_fim_kd
            distill_diag['fim_kd'] = l_fim_kd.item()
            with torch.no_grad():
                k = min(args.routing_topk, student_fim_logits.shape[1])
                s_top = torch.sigmoid(student_fim_logits.squeeze(-1)).topk(k, dim=-1).indices
                t_top = torch.sigmoid(teacher_fim_logits.squeeze(-1)).topk(k, dim=-1).indices
                overlap = 0.0
                for b in range(s_top.shape[0]):
                    overlap += len(set(s_top[b].tolist()) & set(t_top[b].tolist()))
                distill_diag['fim_topk_overlap'] = overlap / (s_top.shape[0] * k)

    task_loss = (final_reconstruction_loss + vq_loss_component + fim_loss_component
                 + perceptual_loss_component)
    total_loss = distill_task_weight * task_loss + distill_loss_component

    return (total_loss, reconstructed_image_batch, current_vq_loss_val, current_rec_loss_val,
            current_fim_loss_val, current_lpips_loss_val, distill_diag)


def train_epoch_semcom_reconstruction(
    model: torch.nn.Module,
    base_reconstruction_criterion: torch.nn.Module,
    fim_criterion: torch.nn.Module,
    lpips_criterion: Optional[nn.Module],
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler: Optional[torch.cuda.amp.GradScaler], 
    args,
    max_norm: Optional[float] = None,           
    start_steps: int = 0,
    lr_schedule_values: Optional[np.ndarray] = None,
    wd_schedule_values: Optional[np.ndarray] = None,
    update_freq: int = 1,
    print_freq: int = 50,
    teacher_model: Optional[torch.nn.Module] = None,
    distill_weight_feat: float = 0.0,
    distill_weight_cos: float = 0.0,
    distill_weight_quant: float = 0.0,
    distill_weight_idx: float = 0.0,
    distill_weight_fim: float = 0.0,
    distill_task_weight: float = 1.0,
):
    model.train(True)
    if hasattr(model, 'quantizer_high'):
        model.quantizer_high.reset_diagnostics()
        model.quantizer_low.reset_diagnostics()
    print(f"  [Loss config] lpips_loss_weight={args.lpips_loss_weight:.4g} (LPIPS active: {lpips_criterion is not None and args.lpips_loss_weight > 0})")
    _in_w, _out_w = _effective_box_weights(args)
    _eff_fim = 0.0 if getattr(args, 'disable_fim', False) else args.fim_loss_weight
    print(f"  [Loss config] disable_fim={getattr(args, 'disable_fim', False)}, "
          f"disable_bbw={getattr(args, 'disable_bbw', False)} -> inside/outside box weights "
          f"({_in_w:.4g}/{_out_w:.4g}); fim_loss_weight={args.fim_loss_weight:.4g} "
          f"(effective {_eff_fim:.4g}); vq_loss_weight={args.vq_loss_weight:.4g}")
    
    main_rec_loss_meter = AverageMeter()
    vq_loss_meter_train = AverageMeter()
    fim_loss_meter_train = AverageMeter()
    lpips_loss_meter_train = AverageMeter()
    distill_meters = {}
    total_loss_meter = AverageMeter()

    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    if teacher_model is not None:
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        print(f"  [DISTILL] teacher_model active: w_feat={distill_weight_feat:.4g} "
              f"w_cos={distill_weight_cos:.4g} w_quant={distill_weight_quant:.4g} "
              f"w_idx={distill_weight_idx:.4g} w_fim={distill_weight_fim:.4g} "
              f"task_weight={distill_task_weight:.4g}")

    if update_freq > 1 : optimizer.zero_grad()
    print_prefix = f'Ep:[{epoch}]'

    for data_iter_step, (semcom_data_input, targets_tuple) in enumerate(data_loader):
        effective_iter = start_steps + (data_iter_step // update_freq)
        if (data_iter_step % update_freq == 0):
            if lr_schedule_values is not None and effective_iter < len(lr_schedule_values):
                for param_group in optimizer.param_groups: param_group["lr"] = lr_schedule_values[effective_iter] * param_group.get("lr_scale", 1.0)
            if wd_schedule_values is not None and effective_iter < len(wd_schedule_values) and args.weight_decay > 0:
                for param_group in optimizer.param_groups:
                    if param_group.get("weight_decay", 0.0) > 0: param_group["weight_decay"] = wd_schedule_values[effective_iter]

        (original_images, bm_pos) = semcom_data_input
        (original_images_for_loss, yolo_gt_targets_list, fim_target_map_batch) = targets_tuple
        
        original_images = original_images.to(device, non_blocking=True)
        original_images_for_loss = original_images_for_loss.to(device, non_blocking=True)
        if bm_pos is not None: bm_pos = bm_pos.to(device, non_blocking=True)
        fim_target_map_batch = fim_target_map_batch.to(device, non_blocking=True)
        
        yolo_gt_on_device = [{"boxes": gt_dict["boxes"].to(device), "labels": gt_dict["labels"].to(device)} for gt_dict in yolo_gt_targets_list]
        samples_for_semcom_input = original_images
        
        with torch.amp.autocast(device_type=args.device, enabled=(loss_scaler is not None)):
            (total_loss, reconstructed_batch, vq_loss_val, rec_loss_val, fim_loss_val,
             lpips_loss_val, distill_diag) = train_semcom_reconstruction_batch(
                model=model, input_samples_for_semcom=samples_for_semcom_input,
                original_images_for_loss=original_images_for_loss, yolo_gt_for_this_batch=yolo_gt_on_device,
                fim_target_importance_map_batch=fim_target_map_batch, bm_pos=bm_pos,
                base_reconstruction_criterion=base_reconstruction_criterion,
                fim_criterion=fim_criterion, lpips_criterion=lpips_criterion, args=args, # Pass lpips_criterion
                digital_seed=args.seed + epoch * 100003 + data_iter_step,
                teacher_model=teacher_model,
                distill_weight_feat=distill_weight_feat, distill_weight_cos=distill_weight_cos,
                distill_weight_quant=distill_weight_quant, distill_weight_idx=distill_weight_idx,
                distill_weight_fim=distill_weight_fim,
                distill_task_weight=distill_task_weight,
            )
        
        total_loss_value_item = total_loss.item()
        if not math.isfinite(total_loss_value_item): print(f"{print_prefix} Total Loss is {total_loss_value_item}, stopping."); sys.exit(1)

        loss_for_backward = total_loss / update_freq
        clip_grad_val_for_scaler = max_norm if max_norm is not None and max_norm > 0 else None
        if loss_scaler is not None:
            loss_scaler(loss_for_backward, optimizer, parameters=model.parameters(), clip_grad=clip_grad_val_for_scaler, update_grad=((data_iter_step + 1) % update_freq == 0))
            if ((data_iter_step + 1) % update_freq == 0) and update_freq > 1: optimizer.zero_grad()
        else:
            loss_for_backward.backward()
            if ((data_iter_step + 1) % update_freq == 0):
                if clip_grad_val_for_scaler: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_val_for_scaler)
                optimizer.step(); optimizer.zero_grad()

        if device == 'cuda': torch.cuda.synchronize()

        total_loss_meter.update(total_loss_value_item, original_images.size(0))
        main_rec_loss_meter.update(rec_loss_val, original_images.size(0))
        vq_loss_meter_train.update(vq_loss_val, original_images.size(0))
        fim_loss_meter_train.update(fim_loss_val, original_images.size(0))
        lpips_loss_meter_train.update(lpips_loss_val, original_images.size(0))
        for k, v in distill_diag.items():
            if k not in distill_meters:
                distill_meters[k] = AverageMeter()
            distill_meters[k].update(v, original_images.size(0))

        batch_psnr_train = utils.calc_psnr(reconstructed_batch.detach().cpu(), original_images_for_loss.detach().cpu())
        batch_ssim_train = utils.calc_ssim(reconstructed_batch.detach().cpu(), original_images_for_loss.detach().cpu())
        psnr_meter.update(np.mean(batch_psnr_train) if batch_psnr_train else 0.0, original_images.size(0))
        ssim_meter.update(np.mean(batch_ssim_train) if batch_ssim_train else 0.0, original_images.size(0))

        if (data_iter_step + 1) % print_freq == 0 or (data_iter_step + 1) == len(data_loader):
            lr = optimizer.param_groups[0]["lr"]
            print_str = f'{print_prefix} It:[{data_iter_step+1}/{len(data_loader)}] ' \
                        f'TotalL: {total_loss_meter.avg:.3f} (Rec: {main_rec_loss_meter.avg:.3f} ' \
                        f'VQ: {vq_loss_meter_train.avg:.3f} FIM: {fim_loss_meter_train.avg:.3f} '
            if lpips_criterion and args.lpips_loss_weight > 0: print_str += f'LPIPS: {lpips_loss_meter_train.avg:.3f} '
            print_str += f') PSNR: {psnr_meter.avg:.2f} SSIM: {ssim_meter.avg:.3f} LR: {lr:.2e}'
            if distill_meters:
                print_str += '  [DISTILL] ' + ' '.join(
                    f'{k}={m.avg:.4f}' for k, m in sorted(distill_meters.items()))
            print(print_str)
            sys.stdout.flush()

    train_stat = {
        'total_loss': total_loss_meter.avg, 'rec_loss': main_rec_loss_meter.avg,
        'vq_loss': vq_loss_meter_train.avg, 'fim_loss': fim_loss_meter_train.avg,
        'lpips_loss': lpips_loss_meter_train.avg,
        'psnr': psnr_meter.avg, 'ssim': ssim_meter.avg
    }
    if hasattr(model, 'quantizer_high'):
        for qname, quantizer in (('high', model.quantizer_high), ('low', model.quantizer_low)):
            _print_quantizer_diagnostics(quantizer, qname, prefix=print_prefix)
    # --- Dead-code restart (RVQFIX, plan 7.2): per-epoch usage counts; codewords
    # below the threshold are re-initialized from this epoch's assigned features. ---
    if (hasattr(model, 'quantizer_high') and hasattr(model.quantizer_high, 'num_stages')
            and getattr(args, 'rvq_restart_threshold', 0.0) > 0
            and not getattr(args, 'freeze_codebooks', False)):
        restarted = model.quantizer_high.restart_dead_codewords(
            threshold=args.rvq_restart_threshold, seed=args.rvq_kmeans_seed)
        if restarted:
            print(f"{print_prefix} [RVQFIX] dead-code restarts: {restarted}")
    print(f"--- Finished Training {print_prefix} ---")
    sys.stdout.flush()
    return train_stat
