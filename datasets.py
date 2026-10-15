# datasets.py
import os
import math
import torch
import numpy as np
import torch.utils.data as data
from PIL import Image
import glob
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from typing import Tuple, Any, List, Dict
import cv2
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import functional as F

class RandomMaskingGenerator:
    """
    Generates a random mask for MAE-style processing.
    Output: A boolean numpy array where True means "masked".
    """
    def __init__(self, input_size_patches: Tuple[int, int], mask_ratio: float):
        self.height_patches, self.width_patches = input_size_patches
        self.num_patches = self.height_patches * self.width_patches
        self.num_mask = int(mask_ratio * self.num_patches)
        if self.num_mask < 0: self.num_mask = 0
        if self.num_mask > self.num_patches: self.num_mask = self.num_patches

    def __repr__(self):
        return f"RandomMasker(total_patches={self.num_patches}, num_mask={self.num_mask})"

    def __call__(self) -> np.ndarray:
        if self.num_mask == 0:
            return np.zeros(self.num_patches, dtype=bool)
        if self.num_mask == self.num_patches:
            return np.ones(self.num_patches, dtype=bool)
        
        mask = np.hstack([
            np.ones(self.num_mask, dtype=bool),
            np.zeros(self.num_patches - self.num_mask, dtype=bool)
        ])
        np.random.shuffle(mask)
        return mask

class SemComInputProcessor:
    """
    Applies image transformations (resize, ToTensor) and generates the
    SemCom patch mask for the ViT encoder.
    """
    def __init__(self,
                 image_pixel_size: int,
                 semcom_patch_grid_size: Tuple[int, int],
                 mask_ratio: float,
                 is_train: bool):
        self.image_pixel_size = image_pixel_size
        self.image_transform = transforms.Compose([
            transforms.Resize((image_pixel_size, image_pixel_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        self.mask_generator = RandomMaskingGenerator(semcom_patch_grid_size, mask_ratio)

    def __call__(self, image_pil: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        transformed_image_tensor = self.image_transform(image_pil)
        semcom_patch_mask_np = self.mask_generator()
        semcom_patch_mask_tensor = torch.from_numpy(semcom_patch_mask_np)
        return transformed_image_tensor, semcom_patch_mask_tensor

class YOLODataset(data.Dataset):
    def __init__(self,
                 img_root_dir_for_split: str,
                 img_pixel_size: int,
                 semcom_vit_patch_size: int,
                 semcom_encoder_mask_ratio: float,
                  is_train_split: bool,
                  num_object_classes: int,
                  augment: bool = True
                  ):
        self.img_dir = os.path.join(img_root_dir_for_split, 'images')
        self.label_dir = os.path.join(img_root_dir_for_split, 'labels')
        self.img_pixel_size = img_pixel_size
        self.is_train_split = is_train_split
        self.vit_patch_size = semcom_vit_patch_size
        self.num_patches_h = img_pixel_size // semcom_vit_patch_size
        self.num_patches_w = img_pixel_size // semcom_vit_patch_size
        self.num_total_patches = self.num_patches_h * self.num_patches_w
        self.num_object_classes = num_object_classes

        self.img_files = sorted(
            glob.glob(os.path.join(self.img_dir, '*.jpg')) +
            glob.glob(os.path.join(self.img_dir, '*.png')) +
            glob.glob(os.path.join(self.img_dir, '*.jpeg'))
        )
        self.label_files = [
            os.path.join(self.label_dir, os.path.splitext(os.path.basename(f))[0] + '.txt')
            for f in self.img_files
        ]
        initial_img_count = len(self.img_files)
        if self.is_train_split or (self.label_files and os.path.exists(self.label_files[0])):
            valid_indices = [i for i, lf in enumerate(self.label_files) if os.path.exists(lf)]
            self.img_files = [self.img_files[i] for i in valid_indices]
            self.label_files = [self.label_files[i] for i in valid_indices]
            if len(self.img_files) < initial_img_count:
                print(f"Warning: {initial_img_count - len(self.img_files)} images removed from {self.img_dir} due to missing labels.")
        if not self.img_files:
            raise FileNotFoundError(f"No image/label pairs found for {img_root_dir_for_split}.")

        # P0-4 (Wave-4 AUG): deterministic-seeded (42), stochastic per-sample RNG
        # used ONLY on the training path. Valid/test stays deterministic Resize+ToTensor.
        # --no_augment (EDGE STEP3) turns this OFF on the training path too, because the
        # Wave-6 SWEEP found augmentation collapses detection ~14x at 10 dB.
        self.aug_rng = np.random.RandomState(42) if (self.is_train_split and augment) else None

        self.semcom_processor = SemComInputProcessor(
            image_pixel_size=img_pixel_size,
            semcom_patch_grid_size=(self.num_patches_h, self.num_patches_w),
            mask_ratio=semcom_encoder_mask_ratio,
            is_train=is_train_split
        )

    def __len__(self):
        return len(self.img_files)

    def _sample_crop_params(self, rng: np.random.RandomState, width: int, height: int):
        # P0-4: random resized crop window (top, left, height, width) in ORIGINAL
        # image pixels. Same algorithm family as torchvision RandomResizedCrop
        # (scale [0.8,1.0], ratio [3/4,4/3]), but drawn from the deterministic
        # per-instance rng so the whole pipeline is reproducible.
        scale = rng.uniform(0.8, 1.0)
        ratio = rng.uniform(3.0 / 4.0, 4.0 / 3.0)
        area = width * height
        for _ in range(10):
            target_area = area * scale
            aspect = math.sqrt(ratio)
            crop_h = int(round(math.sqrt(target_area * aspect)))
            crop_w = int(round(math.sqrt(target_area / aspect)))
            if crop_w <= width and crop_h <= height:
                left = rng.randint(0, width - crop_w + 1) if width > crop_w else 0
                top = rng.randint(0, height - crop_h + 1) if height > crop_h else 0
                return top, left, crop_h, crop_w
        # fallback: largest centered crop
        scale = min(1.0, width / height, height / width)
        crop_w = int(width * scale)
        crop_h = int(height * scale)
        return (height - crop_h) // 2, (width - crop_w) // 2, crop_h, crop_w

    def _augment_image_and_boxes(self, img_pil: Image.Image, boxes_np: np.ndarray,
                                 force_crop: Tuple[int, int, int, int] = None,
                                 force_flips: Tuple[bool, bool] = None):
        """P0-4 (Wave-4 AUG): training-path augmentation that transforms boxes
        with the image. boxes_np: Nx4 normalized xyxy in the ORIGINAL image.
        Returns (augmented_pil, transformed_boxes_np).
        Val/test path never calls this (aug_rng is None there).
        """
        rng = self.aug_rng
        W, H = img_pil.size
        new_boxes = boxes_np.astype(np.float64).copy() if boxes_np.shape[0] > 0 else boxes_np.copy()

        # 1) random resized crop -> crop image AND boxes, drop boxes with kept area < 20%
        if force_crop is not None:
            top, left, crop_h, crop_w = force_crop
        else:
            top, left, crop_h, crop_w = self._sample_crop_params(rng, W, H)
        img_pil = F.resized_crop(img_pil, top, left, crop_h, crop_w,
                                 (self.img_pixel_size, self.img_pixel_size),
                                 interpolation=InterpolationMode.BICUBIC)
        if new_boxes.shape[0] > 0:
            bx1 = new_boxes[:, 0] * W
            by1 = new_boxes[:, 1] * H
            bx2 = new_boxes[:, 2] * W
            by2 = new_boxes[:, 3] * H
            ix1 = np.maximum(bx1, left)
            iy1 = np.maximum(by1, top)
            ix2 = np.minimum(bx2, left + crop_w)
            iy2 = np.minimum(by2, top + crop_h)
            inter_area = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
            orig_area = (bx2 - bx1) * (by2 - by1)
            keep = inter_area >= 0.2 * orig_area
            kept = []
            for k in np.where(keep)[0]:
                nx1 = (ix1[k] - left) / crop_w
                ny1 = (iy1[k] - top) / crop_h
                nx2 = (ix2[k] - left) / crop_w
                ny2 = (iy2[k] - top) / crop_h
                if nx2 > nx1 and ny2 > ny1:
                    kept.append([nx1, ny1, nx2, ny2])
            new_boxes = np.asarray(kept, dtype=np.float64).reshape(-1, 4)

        # 2) flips (50% each) -> cx -> 1-cx on the cropped/normalized coordinates
        do_hflip = rng.uniform() < 0.5 if force_flips is None else force_flips[0]
        do_vflip = rng.uniform() < 0.5 if force_flips is None else force_flips[1]
        if do_hflip:
            img_pil = F.hflip(img_pil)
            if new_boxes.shape[0] > 0:
                new_boxes[:, [0, 2]] = 1.0 - new_boxes[:, [2, 0]]
        if do_vflip:
            img_pil = F.vflip(img_pil)
            if new_boxes.shape[0] > 0:
                new_boxes[:, [1, 3]] = 1.0 - new_boxes[:, [3, 1]]

        # 3) mild colour jitter (brightness/contrast/saturation +/-0.2, no hue shift)
        img_pil = F.adjust_brightness(img_pil, 1.0 + rng.uniform(-0.2, 0.2))
        img_pil = F.adjust_contrast(img_pil, 1.0 + rng.uniform(-0.2, 0.2))
        img_pil = F.adjust_saturation(img_pil, 1.0 + rng.uniform(-0.2, 0.2))

        # 4) light Gaussian blur, applied with low probability (kernel 3, sigma <= 0.5)
        if rng.uniform() < 0.15:
            img_pil = F.gaussian_blur(img_pil, kernel_size=3, sigma=rng.uniform(0.1, 0.5))

        return img_pil, new_boxes

    def _create_patch_importance_map(self, gt_boxes_abs_xyxy: torch.Tensor) -> torch.Tensor:
        target_map_flat = torch.zeros(self.num_total_patches, 1, dtype=torch.float32)
        if gt_boxes_abs_xyxy.numel() == 0:
            return target_map_flat

        patch_coords_x = torch.arange(0, self.img_pixel_size, self.vit_patch_size)
        patch_coords_y = torch.arange(0, self.img_pixel_size, self.vit_patch_size)
        patch_idx = 0
        for r_idx in range(self.num_patches_h):
            for c_idx in range(self.num_patches_w):
                p_x1 = patch_coords_x[c_idx]
                p_y1 = patch_coords_y[r_idx]
                p_x2 = p_x1 + self.vit_patch_size
                p_y2 = p_y1 + self.vit_patch_size
                is_important = False
                for box_abs in gt_boxes_abs_xyxy:
                    b_x1, b_y1, b_x2, b_y2 = box_abs
                    inter_x1 = torch.max(p_x1, b_x1)
                    inter_y1 = torch.max(p_y1, b_y1)
                    inter_x2 = torch.min(p_x2, b_x2)
                    inter_y2 = torch.min(p_y2, b_y2)
                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        is_important = True
                        break
                if is_important:
                    target_map_flat[patch_idx, 0] = 1.0
                patch_idx += 1
        return target_map_flat

    def __getitem__(self, index) -> Tuple[Tuple[torch.Tensor, torch.Tensor],
                                         Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]]:
        img_path = self.img_files[index]
        label_path = self.label_files[index]

        try:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                raise IOError(f"cv2.imread returned None for path: {img_path}")
            img_rgb_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_pil = to_pil_image(img_rgb_np)
            
        except Exception as e:
            print(f"CRITICAL Error loading image {img_path}: {e}. Skipping and returning next.")
            next_index = (index + 1) % len(self)
            return self.__getitem__(next_index)

        boxes_normalized_xyxy = []
        class_labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        try:
                            class_id = int(parts[0])
                            cx, cy, w, h = map(float, parts[1:])
                            x_min = np.clip(cx - w / 2, 0.0, 1.0)
                            y_min = np.clip(cy - h / 2, 0.0, 1.0)
                            x_max = np.clip(cx + w / 2, 0.0, 1.0)
                            y_max = np.clip(cy + h / 2, 0.0, 1.0)
                            if x_max > x_min and y_max > y_min:
                                boxes_normalized_xyxy.append([x_min, y_min, x_max, y_max])
                                class_labels.append(class_id)
                        except ValueError:
                            print(f"Warning: Malformed line in {label_path}: '{line.strip()}'")

        # P0-4 (Wave-4 AUG): training path augments image AND boxes together;
        # val/test uses the plain Resize+ToTensor pipeline (aug_rng is None).
        if self.aug_rng is not None:
            boxes_np = np.asarray(boxes_normalized_xyxy, dtype=np.float64).reshape(-1, 4)
            img_pil, boxes_np = self._augment_image_and_boxes(img_pil, boxes_np)
            boxes_normalized_xyxy = boxes_np.tolist()

        img_tensor_for_semcom, semcom_encoder_patch_mask = self.semcom_processor(img_pil)

        boxes_normalized_tensor = torch.as_tensor(boxes_normalized_xyxy, dtype=torch.float32)
        labels_tensor = torch.as_tensor(class_labels, dtype=torch.int64)
        abs_pixel_boxes_tensor = boxes_normalized_tensor.clone()
        if abs_pixel_boxes_tensor.numel() > 0:
            abs_pixel_boxes_tensor[:, [0, 2]] *= self.img_pixel_size
            abs_pixel_boxes_tensor[:, [1, 3]] *= self.img_pixel_size
        
        yolo_gt_for_metrics_dict = {"boxes": abs_pixel_boxes_tensor, "labels": labels_tensor}
        fim_target_importance_map = self._create_patch_importance_map(abs_pixel_boxes_tensor)
        semcom_input_tuple = (img_tensor_for_semcom, semcom_encoder_patch_mask)
        targets_tuple = (img_tensor_for_semcom.clone(), yolo_gt_for_metrics_dict, fim_target_importance_map)

        return semcom_input_tuple, targets_tuple


def build_dataset(is_train: bool, args: Any) -> data.Dataset:
    if args.data_set == 'fish':
        if is_train:
            data_split_subdir = 'train'
        elif not args.eval:
            data_split_subdir = 'valid'
        else:
            data_split_subdir = 'test'
        current_split_root_dir = os.path.join(args.data_path, data_split_subdir)
        if not os.path.isdir(current_split_root_dir):
            raise FileNotFoundError(f"Dataset dir for '{data_split_subdir}' not found: {current_split_root_dir}")

        dataset = YOLODataset(
            img_root_dir_for_split=current_split_root_dir,
            img_pixel_size=args.input_size,
            semcom_vit_patch_size=args.patch_size,
            semcom_encoder_mask_ratio=args.mask_ratio,
            is_train_split=is_train,
            num_object_classes=args.num_object_classes,
            augment=not getattr(args, 'no_augment', False)
        )
    else:
        raise NotImplementedError(f"Dataset '{args.data_set}' not implemented.")
    return dataset

def yolo_collate_fn(batch: List[Tuple[Tuple[torch.Tensor, torch.Tensor],
                                       Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]]]
                   ) -> Tuple[Tuple[torch.Tensor, torch.Tensor],
                              Tuple[torch.Tensor, List[Dict[str, torch.Tensor]], torch.Tensor]]:
    semcom_input_images = []
    semcom_encoder_masks = []
    semcom_reconstruction_targets = []
    yolo_gt_target_list_of_dicts = []
    fim_target_importance_maps = []

    for item in batch:
        semcom_input_tuple, targets_tuple = item
        
        semcom_input_images.append(semcom_input_tuple[0])
        semcom_encoder_masks.append(semcom_input_tuple[1])
        
        semcom_reconstruction_targets.append(targets_tuple[0])
        yolo_gt_target_list_of_dicts.append(targets_tuple[1])
        fim_target_importance_maps.append(targets_tuple[2])

    collated_semcom_input_images = torch.stack(semcom_input_images, 0)
    collated_semcom_encoder_masks = torch.stack(semcom_encoder_masks, 0)
    collated_semcom_reconstruction_targets = torch.stack(semcom_reconstruction_targets, 0)
    collated_fim_target_importance_maps = torch.stack(fim_target_importance_maps, 0)

    collated_semcom_input_tuple = (collated_semcom_input_images, collated_semcom_encoder_masks)
    collated_targets_tuple = (
        collated_semcom_reconstruction_targets,
        yolo_gt_target_list_of_dicts,
        collated_fim_target_importance_maps
    )
    return collated_semcom_input_tuple, collated_targets_tuple
