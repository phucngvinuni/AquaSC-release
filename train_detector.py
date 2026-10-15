# train_detector.py - Wave-6 TRANSFER: train YOLOv8s detectors (Site B, both-sites)
# Same schedule as DETECTOR2 (wave-4): yolov8s.pt pretrained, imgsz 224, batch 16,
# epochs 200, patience 30, seed 42, deterministic. Output weights copied to an
# explicit path with SHA-256 recorded.
import argparse
import shutil
import os
import hashlib

from ultralytics import YOLO


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--project', required=True)
    ap.add_argument('--out', required=True, help='final best.pt copy path')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--imgsz', type=int, default=224)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='0')
    args = ap.parse_args()

    model = YOLO('yolov8s.pt')
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
        device=args.device,
        workers=8,
        val=True,
        split='val',
        amp=True,
    )
    # locate best.pt via the trainer's actual save_dir (robust to ultralytics'
    # runs/<task>/ prefixing of relative project paths)
    save_dir = model.trainer.save_dir
    best = os.path.join(save_dir, 'weights', 'best.pt')
    if not os.path.exists(best):
        raise FileNotFoundError(f'best.pt not found at {best}')
    shutil.copy(best, args.out)
    print(f'TRAINED: {args.name} -> {args.out}')
    print(f'SHA256 {args.out}: {sha256(args.out)}')
    print(f'SHA256 {best}: {sha256(best)}')


if __name__ == '__main__':
    main()
