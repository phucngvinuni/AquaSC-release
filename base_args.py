import argparse

IMGC_NUMCLASS = 1   # For the OBJECT DETECTION task (e.g., 1 for 'fish' or your single class)

def get_args():
    parser = argparse.ArgumentParser('SemCom for Reconstruction + YOLO Eval with Dual VQ', add_help=False)

    # ---------------- BASIC TRAINING & DATASET ----------------
    # ... (keep existing args like --model, --data_set, --data_path, --input_size, etc.) ...
    parser.add_argument('--model', default='ViT_Reconstruction_Model_Default', type=str, metavar='MODEL',
                        help='Name of model to train (ensure it is registered in model.py)')
    parser.add_argument('--data_set', default='fish', choices=['fish'],
                        type=str, help='Dataset type')
    parser.add_argument('--data_path', default='../yolo_fish_dataset_root/', type=str,
                        help='Root dataset path (must contain train/, valid/, test/ subdirs as per YOLODataset)')
    parser.add_argument('--input_size', default=224, type=int,
                        help='Image pixel H, W for SemCom input and reconstruction target')
    parser.add_argument('--num_object_classes', default=IMGC_NUMCLASS, type=int,
                        help='Number of object classes for detection task (should match your YOLO model)')
    parser.add_argument('--no_augment', action='store_true',
                        help='[EDGE STEP3] Disable the training-path data augmentation (random-resized-crop, '
                             'hflip/vflip, colour jitter, gaussian blur added in Wave-4 AUG). Val/test are '
                             'already un-augmented regardless of this flag.')
    parser.add_argument('--output_dir', default='outputs/semcom_dualvq_fish', type=str, # Adjusted output_dir
                        help='Path to save checkpoints and logs')
    parser.add_argument('--eval_viz_output_dir', default='outputs/semcom_dualvq_fish/visualizations', type=str,
                        help='Path to save evaluation visualizations')


    # ---------------- TRAINING HYPERPARAMETERS ----------------
    # ... (keep existing: --batch_size, --epochs, --update_freq, --lr, --min_lr, --warmup_epochs, --warmup_lr) ...
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--update_freq', default=1, type=int, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='Initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR', help='Lower LR bound for scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='Epochs to warmup LR')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR', help='Initial LR for warmup')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N', # <--- ADD THIS LINE
                        help='Number of warmup steps. Overrides warmup_epochs if > 0. Default -1 to use warmup_epochs.')

    # ---------------- OPTIMIZER ----------------
    # ... (keep existing: --opt, --weight_decay, --opt_betas, --opt_eps, --momentum, --clip_grad) ...
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', choices=['adamw', 'sgd'],
                        help='Optimizer (adamw, sgd)')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--opt_betas', default=[0.9, 0.999], type=float, nargs='+', help='AdamW betas')
    parser.add_argument('--opt_eps', default=1e-8, type=float, help='AdamW epsilon')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
    parser.add_argument('--clip_grad', type=float, default=1.0, metavar='NORM', help='Gradient clipping norm')
    parser.add_argument('--weight_decay_end', type=float, default=None, # <--- ADD OR UNCOMMENT THIS LINE
                        help="Final value of weight decay for cosine scheduler. If None, WD is constant.")
    parser.add_argument('--save_ckpt', action='store_true', # <--- ADD THIS LINE (or uncomment it)
                        help="Enable saving model checkpoints during training.")
    # ---------------- SEMCOM MODEL ARCHITECTURE ----------------
    parser.add_argument('--patch_size', default=8, type=int, help="Patch size for ViT encoder")
    parser.add_argument('--encoder_embed_dim', default=512, type=int)
    parser.add_argument('--encoder_depth', default=8, type=int)
    parser.add_argument('--encoder_num_heads', default=8, type=int)
    parser.add_argument('--decoder_embed_dim', default=512, type=int)
    parser.add_argument('--decoder_depth', default=6, type=int)
    parser.add_argument('--decoder_num_heads', default=8, type=int)
    parser.add_argument('--drop_rate', type=float, default=0.0, metavar='PCT')
    parser.add_argument('--drop_path_rate', type=float, default=0.1, metavar='PCT')

    # Quantizer Parameters (Common dimension, different bitrates/codebook sizes)
    parser.add_argument('--quantizer_dim', default=512, type=int,
                        help='Embedding dimension for VQ codebook vectors (same for high and low).')
    parser.add_argument('--bits_vq_high', default=10, type=int, # More bits for important
                        help='Number of bits for the high-importance VQ codebook size (2^bits). IGNORED when --quantizer_type=rvq.')
    parser.add_argument('--bits_vq_low', default=10, type=int,  # Fewer bits for less important
                        help='Number of bits for the low-importance VQ codebook size (2^bits).')
    parser.add_argument('--quantizer_type', default='flat', choices=['flat', 'rvq', 'analog'], type=str,
                        help="High-path quantizer: 'flat' = HierarchicalQuantizer (2^bits_vq_high, legacy), 'rvq' = ResidualQuantizer (H1: rvq_stages x rvq_bits_per_stage at the same effective rate), 'analog' = AquaSC-Analog (Wave 19 Part 2): no VQ/RVQ codebook, a standard DeepJSCC-style linear bottleneck projecting every token directly to complex symbols, uniformly (routing/FIM still computed and reported, but does not gate the analog projection -- isolates transport, not rate allocation).")
    parser.add_argument('--analog_symbols_per_token', default=9, type=int,
                        help="AquaSC-Analog, UNIFORM-RATE mode only (used when --analog_symbols_per_token_high/_low are both 0): COMPLEX symbols emitted per token by the analog bottleneck (nn.Linear(quantizer_dim, 2*S) -- S is complex symbols/token, the Linear's output width is 2*S REALS/token, not S reals/token). Corrected 2026-09-14: an earlier version of this help text said '9 reals/token x 784 = 3528 complex symbols = 0.0703 uses/px' -- that arithmetic divided by 2 for complex-pairing that the *2 in the Linear layer had already done, undercounting by exactly 2x. The real, measured rate for S=9 is 784 tokens x 9 COMPLEX symbols/token = 7056 complex symbols/image = 0.140625 uses/px (verified by a live forward pass, not by this formula -- see RESULTS_OF_RECORD.md 6q). This uniform mode also applies the SAME S to every token regardless of FIM routing, so routing does not gate rate here -- prefer --analog_symbols_per_token_high/_low (dual-rate) for a routing-preserving, rate-matched comparison.")
    parser.add_argument('--analog_symbols_per_token_high', default=0, type=int,
                        help="AquaSC-Analog, DUAL-RATE mode (Wave 19 Part 2 correction, 2026-09-14): COMPLEX symbols/token for the K=routing_topk FIM-high-routed tokens (separate nn.Linear(quantizer_dim, 2*S_high)). Must be set together with --analog_symbols_per_token_low (both >0) to enable dual-rate; restores routing as a real degree of freedom (high tokens get more symbols than low tokens), unlike the uniform mode where routing computed but unused. Recommended pair for near-exact flagship rate match: high=34, low=2 -> 63*34+721*2=3584 complex symbols/image = 0.071429 uses/px (flagship logged rate: 0.0714).")
    parser.add_argument('--analog_symbols_per_token_low', default=0, type=int,
                        help="AquaSC-Analog, DUAL-RATE mode: COMPLEX symbols/token for the (784-routing_topk) FIM-low-routed tokens (separate nn.Linear(quantizer_dim, 2*S_low)). See --analog_symbols_per_token_high.")
    parser.add_argument('--rvq_stages', default=3, type=int,
                        help='Number of RVQ stages for the high path (default 3).')
    parser.add_argument('--rvq_bits_per_stage', default=6, type=int,
                        help='Codebook bits per RVQ stage (2^bits entries per stage). Effective high rate = rvq_stages x rvq_bits_per_stage (default 3x6=18).')

    # ---------------- RVQFIX (plan 7.2: stage-1 collapse fixes) ----------------
    parser.add_argument('--rvq_kmeans_init', action=argparse.BooleanOptionalAction, default=True,
                        help='[RVQFIX] k-means initialize RVQ stage codebooks from real encoder features before training. Default ON (use --no-rvq_kmeans_init to disable).')
    parser.add_argument('--rvq_ema_decay', default=0.99, type=float,
                        help='[RVQFIX] EMA decay for codebook updates (0 disables EMA). Default 0.99.')
    parser.add_argument('--rvq_restart_threshold', default=1.0, type=float,
                        help='[RVQFIX] per-epoch usage count below which a codeword is re-initialized from assigned features (0 disables dead-code restart). Default 1.0.')
    parser.add_argument('--rvq_kmeans_images', default=800, type=int,
                        help='[RVQFIX] max training images used to collect encoder features for k-means init.')
    parser.add_argument('--rvq_kmeans_iters', default=20, type=int,
                        help='[RVQFIX] Lloyd iterations per k-means stage.')
    parser.add_argument('--rvq_kmeans_seed', default=42, type=int,
                        help='[RVQFIX] deterministic seed for feature collection and k-means.')
    # Remove old --bits_for_quantizer if it exists to avoid confusion

    # ---------------- FIM (Feature Importance Module) PARAMETERS ----------------
    parser.add_argument('--fim_embed_dim', default=256, type=int)
    parser.add_argument('--fim_depth', default=2, type=int)
    parser.add_argument('--fim_num_heads', default=4, type=int)
    parser.add_argument('--fim_drop_rate', default=0.1, type=float)
    parser.add_argument('--fim_routing_threshold', default=0.7, type=float, # Threshold to route to VQ_High
                        help='FIM score threshold to route tokens to high-importance VQ.')
    parser.add_argument('--routing_topk', default=63, type=int,
                        help='Fixed number of high-fidelity tokens (top-K of FIM scores). K=63 for 784 tokens (P_high=0.08). <=0 falls back to fim_routing_threshold.')
    parser.add_argument('--eps_stab', default=1e-6, type=float,
                        help='Stabilizer added to the denominator of the FIM-weighted VQ loss (paper Eq. 4).')

    # ---------------- LOSS WEIGHTS ----------------
    parser.add_argument('--inside_box_loss_weight', default=10, type=float)
    parser.add_argument('--outside_box_loss_weight', default=0.25, type=float)
    parser.add_argument('--vq_loss_weight', default=0.15, type=float, # Might need adjustment
                        help='Weight for the combined (FIM-weighted) VQ losses.')
    parser.add_argument('--fim_loss_weight', default=0.5, type=float, # Might need adjustment
                        help='Weight for the FIM training loss.')
    parser.add_argument('--lpips_loss_weight', default=0, type=float,
                        help='Weight for LPIPS perceptual loss (0 to disable).')

    # ---------------- ABLATION FLAGS (Wave 9b, plan §0.3 G2 / §11.3 cross-check) ----------------
    parser.add_argument('--disable_fim', action='store_true',
                        help='[ABLATION] Replace FIM routing with RANDOM top-K routing (seeded 42, '
                             'K=routing_topk, rate-identical), zero the FIM BCE loss weight, and make '
                             'the VQ loss UNWEIGHTED (no per-token FIM weighting). The FIM module still '
                             'exists in the architecture but is unused in routing and loss.')
    parser.add_argument('--disable_bbw', action='store_true',
                        help='[ABLATION] Disable box-weighted reconstruction: inside_box_loss_weight = '
                             'outside_box_loss_weight = 1.0 (plain reconstruction loss, no box weighting).')

    # ---------------- CHANNEL SIMULATION ----------------
    # ... (keep existing: --channel_type, --snr_db_train_min, --snr_db_train_max, --snr_db_eval) ...
    parser.add_argument('--channel_type', default='rayleigh', choices=['none', 'awgn', 'rayleigh', 'rician'], type=str)
    parser.add_argument('--snr_db_train_min', default=20, type=float)
    parser.add_argument('--snr_db_train_max', default=25, type=float)
    parser.add_argument('--snr_db_eval', default=22, type=float)

    # ---------------- DIGITAL PIPELINE (Wave 5 CHANNEL-cont, plan 5.3/5.4) ----------------
    parser.add_argument('--digital_ldpc', action='store_true',
                        help='[R3 #3] In --eval, transmit indices through the real digital chain '
                             '(LDPC + QAM + Rayleigh + soft demap + CRC + erasure + N4 concealment) '
                             'via the Sionna subprocess bridge. Off = analog-proxy hard-decision path.')
    parser.add_argument('--digital_csi_mode', default='perfect', choices=['perfect', 'pilot'], type=str,
                        help='CSI model for the digital chain.')
    parser.add_argument('--digital_joint_frame', action='store_true',
                        help='Opt-in matched5 profile: --eval --digital_ldpc, flagship RVQ 18/2, '
                             'perfect CSI; single BG2(3073,12288), QPSK, sum-product 40 iterations, '
                             '6144 symbols/image, per-image Rayleigh. Overrides legacy modulation '
                             'and payload-rate flags; does not use stage UEP.')
    parser.add_argument('--digital_bridge_python', default='__SIONNA_PY__', type=str,
                        help='Python interpreter that hosts Sionna/TensorFlow for the digital bridge.')
    parser.add_argument('--digital_bits_per_symbol', default=4, type=int, choices=[1, 2, 4],
                        help='[plan 16.9 Track F/E3] Modulation order for the coded digital chain: '
                             '1=BPSK, 2=QPSK, 4=16-QAM (current default). Evaluation-only MCS lever -- '
                             'the source coder and model are unaffected; only the channel-layer '
                             'modulation changes.')
    parser.add_argument('--digital_payload_rate', default=0.5, type=float, choices=[0.5, 1 / 3, 2 / 3],
                        help='[plan 16.9 Track F/E3] LDPC code rate for the payload segment '
                             '(stage2|stage3|low|payload_crc). header|mask|stage1 always stays at '
                             'rate 1/3 (maximum protection). Default 1/2 matches the existing E2 '
                             'checkpoint\'s trained/evaluated configuration. 2/3 added 2026-09-15 '
                             '(Wave 20 rate-matching): --digital_bits_per_symbol 2 --digital_payload_rate '
                             '2/3 gives exactly 6144 symbols/image = 0.122449 uses/px on the flagship '
                             '18/2 config (n_a=2, n_b=4 LDPC blocks, verified on a live forward pass, '
                             'not derived by hand -- see RESULTS_OF_RECORD.md section 6t), matching '
                             'the Wave 20 joint-frame profile\'s rate exactly for a rate-controlled '
                             'multi-block-vs-joint-frame comparison. get_coder(2/3) was already '
                             'implemented; only this CLI restriction was new.')
    parser.add_argument('--prior_floor_eval', action='store_true',
                        help='[plan 16.9 Track F/E3, mandatory companion] Bypass the encoder and FIM '
                             'entirely; feed the decoder purely random codebook indices (no image '
                             'input, no channel simulation) and evaluate mAP as normal. Establishes '
                             'the decoder\'s learned-prior floor that every low-SNR claim must exceed.')
    parser.add_argument('--index_flip', action='store_true',
                        help='[plan 5.4] training-time index-noise: flip index bits at the measured '
                             'digital error rate (BLER) while keeping the analog proxy.')
    parser.add_argument('--index_flip_csv', default=None, type=str,
                        help='CSV with (snr_db, bler) columns for the index-flip rate (default blers_vs_snr.csv).')
    parser.add_argument('--train_channel_mode', default='analog', choices=['analog', 'digital_uncoded', 'erasure'], type=str,
                        help='[plan 16.9 Track E] Training-time channel. analog (default, unchanged): '
                             'continuous Rayleigh proxy on the codeword vectors. digital_uncoded (E1): '
                             'train through the real uncoded chain (index -> Gray 16-QAM -> Rayleigh -> '
                             'demap), STE gradient to the encoder -- matches the uncoded eval path instead '
                             'of handicapping it with an analog-proxy-trained checkpoint. erasure (E2): '
                             'per-token Bernoulli erasure at the coded chain\'s measured BLER(SNR), '
                             'concealed with the same N4 diffusion the coded eval path uses -- the '
                             'tractable train-through-the-channel form for the non-differentiable LDPC code. '
                             'Mutually exclusive with --index_flip (ignored unless mode=analog).')
    parser.add_argument('--erasure_bler_csv', default=None, type=str,
                        help='CSV with (snr_db, bler) columns for --train_channel_mode erasure '
                             '(default: e2_bler_train.csv, the Track E measured coded-chain BLER curve).')
    parser.add_argument('--finetune_reset', action='store_true',
                        help='After --resume, reset start_epoch to 0 and rebuild a fresh optimizer/'
                             'scaler for a short fine-tune run.')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='S-2/S-2b (Wave 14, plan 16.13): freeze img_encoder after --resume.')
    parser.add_argument('--freeze_decoder', action='store_true',
                        help='Freeze img_decoder + channel_to_decoder_proj after --resume.')
    parser.add_argument('--freeze_fim', action='store_true',
                        help='Freeze fim_module after --resume.')
    parser.add_argument('--freeze_codebooks', action='store_true',
                        help='Freeze quantizer_high + quantizer_low after --resume. Note: RVQ codebooks '
                             'also update via EMA outside the optimizer (model_util.py); this flag only '
                             'stops the gradient/optimizer path, not the EMA path -- see run_class_main.py '
                             'for the matching --freeze_codebooks EMA-disable wiring.')
    parser.add_argument('--distill_teacher_ckpt', default=None, type=str,
                        help='2026-09-17 (encoder-distillation experiment): path to a frozen '
                             'teacher checkpoint whose img_encoder is (typically) deeper/wider '
                             'than this run\'s --encoder_*. Teacher forces the student toward its '
                             'own pre-quantization projected feature, quantized codeword, and FIM '
                             'routing scores via a composite KD loss (see --distill_weight_* below), '
                             'always in addition to the ordinary AquaSC task loss (scaled by '
                             '--distill_task_weight). Requires '
                             '--distill_teacher_encoder_embed_dim/depth/num_heads to describe the '
                             'teacher\'s architecture (defaults to the flagship\'s 512/8/8).')
    parser.add_argument('--distill_weight_feat', default=1.0, type=float,
                        help='Weight on L_feat = MSE(z_S, sg(z_T)) where z = post-projection, '
                             'post-norm token (x_proj_normalized) -- the representation the '
                             'codebook actually sees, not the raw ViT output. No effect unless '
                             '--distill_teacher_ckpt is set.')
    parser.add_argument('--distill_weight_cos', default=0.0, type=float,
                        help='Weight on L_cos = 1 - cosine_sim(z_S, z_T), averaged over tokens. '
                             'Matches direction; L_feat/MSE already matches scale/local geometry. '
                             'No effect unless --distill_teacher_ckpt is set.')
    parser.add_argument('--distill_weight_quant', default=0.0, type=float,
                        help='Weight on L_quant = MSE(q(z_S), sg(q(z_T))), q = the SAME (frozen/'
                             'shared) quantizer_high applied to every token\'s z (not just '
                             'FIM-routed-high tokens -- this loss asks "would encoder_S and '
                             'encoder_T land in the same Voronoi cell", independent of routing). '
                             'The point of this term: small feature MSE does not imply the same '
                             'nearest-codeword assignment near a Voronoi boundary, and the decoder '
                             'only ever sees the quantized codeword, never the raw projected '
                             'feature. No effect unless --distill_teacher_ckpt is set.')
    parser.add_argument('--distill_weight_idx', default=0.0, type=float,
                        help='2026-09-18 pivot: weight on L_idx = mean over RVQ stages of '
                             'CrossEntropy(student_stage_logits, teacher_stage_index) -- '
                             'directly trains the student to land on the SAME codeword '
                             'index as the teacher (per stage), rather than relying on '
                             'feature/quantized-vector MSE as an indirect proxy for it. '
                             'This is the objective that actually matters: the decoder '
                             'only ever sees the discrete codeword, and a small feature '
                             'MSE does not guarantee the same nearest-codeword assignment '
                             'near a Voronoi boundary. No effect unless --distill_teacher_ckpt is set.')
    parser.add_argument('--distill_weight_fim', default=0.0, type=float,
                        help='Weight on L_FIM-KD = MSE(fim_raw_logits_S, sg(fim_raw_logits_T)). '
                             'Keeps the student\'s routing decisions aligned with the teacher\'s as '
                             'the encoder changes underneath FIM. No effect unless '
                             '--distill_teacher_ckpt is set.')
    parser.add_argument('--distill_task_weight', default=1.0, type=float,
                        help='Multiplier on the ordinary AquaSC task loss (BBW reconstruction + '
                             'FIM BCE + VQ) when a distillation teacher is active. Set low (e.g. '
                             '0.1) to prioritize feature/codeword/routing matching over task '
                             'performance early in a KD run; 1.0 (default) leaves the task loss '
                             'at its normal magnitude. No effect unless --distill_teacher_ckpt is set.')
    parser.add_argument('--distill_teacher_encoder_embed_dim', default=512, type=int)
    parser.add_argument('--distill_teacher_encoder_depth', default=8, type=int)
    parser.add_argument('--distill_teacher_encoder_num_heads', default=8, type=int)
    parser.add_argument('--distill_init_from_teacher', action='store_true',
                        help='Also warm-start every student submodule whose shape matches the '
                             'teacher\'s (decoder, quantizers, channel_to_decoder_proj, '
                             'norm_before_quantizer -- everything independent of '
                             'encoder_embed_dim) from --distill_teacher_ckpt, instead of '
                             'training the whole student from scratch. img_encoder/fim_module/'
                             'encoder_to_channel_proj (shape-mismatched) stay randomly '
                             'initialized regardless.')
    parser.add_argument('--per_image_ap_out', default=None, type=str,
                        help='[plan 16.9 Track F1] In --eval, save per-image AP@0.5 (stats.per_image_ap50), '
                             'in dataloader (sorted-filename) order, to this .npy path. For paired '
                             'bootstrap/Wilcoxon against a competitor evaluated on the same image order.')

    # ---------------- YOLO EVALUATION ----------------
    # ... (keep existing: --yolo_weights, --yolo_conf_thres, --yolo_iou_thres) ...
    parser.add_argument('--yolo_weights', default='best.pt', type=str)
    parser.add_argument('--yolo_conf_thres', default=0.4, type=float)
    parser.add_argument('--yolo_iou_thres', default=0.5, type=float)
    parser.add_argument('--map_conf_thres', default=0.001, type=float,
                        help='Confidence threshold for YOLO detections fed to torchmetrics mAP. Keep low (0.001) so the PR curve is not truncated; yolo_conf_thres=0.4 is used for visualization only.')

    # ---------------- MISC & RUN CONTROL ----------------
    # ... (keep existing: --mask_ratio, --save_freq, --device, --seed, --num_workers, --pin_mem, --resume, --eval, etc.)
    parser.add_argument('--mask_ratio', default=0.0, type=float)
    parser.add_argument('--save_freq', default=10, type=int)
    parser.add_argument('--device', default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_true'); parser.set_defaults(pin_mem=True)
    parser.add_argument('--resume', default='', help='Path to checkpoint to resume training from')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--model_key', default='model', type=str) # For loading checkpoints
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true'); parser.set_defaults(dist_on_itp=False)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--csv_log_file', default=None, type=str,
                        help='Path to CSV file to log evaluation metrics.')
    parser.add_argument('--append_csv', action='store_true',
                        help='If set, append results to the CSV file instead of overwriting.')
    parser.add_argument('--save_recons_dir', default='', type=str,
                        help='If set (and --eval), save per-image {orig, recon, boxes, labels} .pt files to this dir for post-hoc metrics (metrics.py).')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    print("Parsed arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
