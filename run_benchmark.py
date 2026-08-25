#!/usr/bin/env python3
"""Benchmark stf8 checkpoints on multiple test sets."""
import json
import math
import os
import sys
import tempfile
import time

import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms

import compressai
from compressai.utils.eval_model.__main__ import (
    collect_images,
    inference,
    inference_entropy_estimation,
    load_checkpoint,
    psnr,
)

CHECKPOINT_DIR = "/data/Dataset/checkpoints/Learned-image-compression-with-transformers/checkpoints"
CHECKPOINTS = ["save1.ckpt", "save2.ckpt", "save3.ckpt", "save4.ckpt"]
DATASETS = {
    "kodim": "/data/Dataset/testdata/kodim/original",
    "tecnick": "/data/Dataset/testdata/tecnick/original",
    "CLIC": "/data/Dataset/testdata/CLIC/original",
}
ARCH = "stf8"
OUTPUT_JSON = "/data/Dataset/Learned-image-compression-with-transformers/benchmark_results.json"


def eval_dataset(model, dataset_path, entropy_estimation=True, verbose=False):
    filepaths = collect_images(dataset_path)
    if not filepaths:
        raise RuntimeError(f"No images in {dataset_path}")

    device = next(model.parameters()).device
    metrics = {"bpp": 0.0, "psnr": 0.0, "ms-ssim": 0.0, "n_images": len(filepaths)}

    recon_dir = tempfile.mkdtemp(prefix="stf8_bench_")
    try:
        for f in filepaths:
            filename = os.path.basename(f)
            x = transforms.ToTensor()(Image.open(f).convert("RGB")).to(device)
            if entropy_estimation:
                rv = _eval_one_entropy(model, x, filename, recon_dir)
            else:
                rv = inference(model, x, filename, recon_path=recon_dir)
            if verbose:
                print(f"    {filename} bpp={rv['bpp']:.4f} psnr={rv['psnr']:.3f} ms-ssim={rv['ms-ssim']:.4f}")
            for k in ("bpp", "psnr", "ms-ssim"):
                metrics[k] += rv[k]
    finally:
        import shutil
        shutil.rmtree(recon_dir, ignore_errors=True)

    for k in ("bpp", "psnr", "ms-ssim"):
        metrics[k] /= len(filepaths)
    return metrics


@torch.no_grad()
def _eval_one_entropy(model, x, filename, recon_dir):
    import torch.nn.functional as F
    from compressai.utils.eval_model.__main__ import psnr, reconstruct

    x = x.unsqueeze(0)
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    h, w = x.size(2), x.size(3)
    p = 64
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    out_net = model.forward(x_padded)
    grid_img = out_net["x_hat"]
    grid_img = F.pad(
        grid_img, (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out_net["likelihoods"].values()
    )
    reconstruct(grid_img, filename, recon_dir)
    return {
        "bpp": bpp.item(),
        "psnr": psnr(x, grid_img),
        "ms-ssim": ms_ssim(x, grid_img, data_range=1.0).item(),
    }


def main():
    compressai.set_entropy_coder("ans")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    results = {}
    for ckpt_name in CHECKPOINTS:
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
        print(f"\n=== Loading {ckpt_name} ===", flush=True)
        t0 = time.time()
        model = load_checkpoint(ARCH, ckpt_path)
        model = model.to(device)
        model.update(force=True)
        print(f"Loaded in {time.time() - t0:.1f}s", flush=True)

        results[ckpt_name] = {}
        for ds_name, ds_path in DATASETS.items():
            print(f"  Evaluating {ds_name} ({ds_path})...", flush=True)
            t1 = time.time()
            metrics = eval_dataset(model, ds_path, entropy_estimation=True)
            print(
                f"    bpp={metrics['bpp']:.4f} psnr={metrics['psnr']:.3f} "
                f"ms-ssim={metrics['ms-ssim']:.4f} ({time.time()-t1:.1f}s)",
                flush=True,
            )
            results[ckpt_name][ds_name] = metrics

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT_JSON}", flush=True)
    return results


if __name__ == "__main__":
    main()
