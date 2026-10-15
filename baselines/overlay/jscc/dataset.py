# Trong lo1/dataset.py hoặc lo1/train.py (phần CustomDataset)

from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import numpy as np
from torchvision import transforms # Thêm import này

class CustomDatasetForJSCC(Dataset): # Đổi tên để phân biệt
    def __init__(self, image_dir, label_dir=None, transform=None, target_image_size=(128, 128)):
        self.image_dir = image_dir
        self.label_dir = label_dir # Tùy chọn, có thể không dùng cho training JSCC
        self.transform = transform
        self.target_image_size = target_image_size # Kích thước ảnh mục tiêu

        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        # print(f"Found {len(self.image_paths)} images in {image_dir}")


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Could not load image {img_path}. Skipping. Error: {e}")
            # Trả về một ảnh dummy hoặc ảnh tiếp theo
            next_idx = (idx + 1) % len(self)
            if next_idx == idx : # Nếu dataset chỉ có 1 ảnh và ảnh đó lỗi
                 return torch.zeros(3, self.target_image_size[0], self.target_image_size[1]), {} # Dummy image và target
            return self.__getitem__(next_idx)


        if self.transform:
            image_tensor = self.transform(image)
        else:
            # Default transform nếu không có
            default_transform = transforms.Compose([
                transforms.Resize(self.target_image_size),
                transforms.ToTensor(),
                # image_normalization('normalization') # Hàm này của bạn, có thể cần import
            ])
            image_tensor = default_transform(image)

        # Đối với JSCC thuần túy, label có thể không cần thiết cho training loss
        # Nhưng chúng ta cần GT bounding box cho việc đánh giá mAP sau này
        # Tạo một dictionary cho target, tương tự như YOLODataset của AquaSC
        # để hàm đánh giá mAP có thể dùng chung.
        target_for_map = {"boxes": torch.empty((0,4)), "labels": torch.empty(0, dtype=torch.long)}
        if self.label_dir:
            label_filename = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
            label_path_full = os.path.join(self.label_dir, label_filename)
            if os.path.exists(label_path_full):
                boxes_normalized_xyxy = []
                class_labels = []
                with open(label_path_full, 'r') as f:
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
                                print(f"Warning: Malformed line in {label_path_full}: '{line.strip()}'")
                
                boxes_tensor_norm = torch.as_tensor(boxes_normalized_xyxy, dtype=torch.float32)
                # Chuyển sang pixel coordinates (giả sử kích thước ảnh sau transform là target_image_size)
                # Điều này quan trọng cho torchmetrics
                abs_pixel_boxes_tensor = boxes_tensor_norm.clone()
                if abs_pixel_boxes_tensor.numel() > 0:
                     abs_pixel_boxes_tensor[:, [0, 2]] *= self.target_image_size[1] # width
                     abs_pixel_boxes_tensor[:, [1, 3]] *= self.target_image_size[0] # height

                target_for_map["boxes"] = abs_pixel_boxes_tensor
                target_for_map["labels"] = torch.as_tensor(class_labels, dtype=torch.int64)

        # Hàm __getitem__ của dataset cho training JSCC chỉ cần trả về ảnh
        # Nhưng để hàm collate và đánh giá mAP dễ dàng hơn, ta trả về cả target_for_map
        return image_tensor, target_for_map