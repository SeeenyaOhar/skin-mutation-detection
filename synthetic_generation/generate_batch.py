
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import torch



def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5,
                     x2: float = 4096, y2: float = 1.15) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(num_steps: int, image_seq_len: int, base_shift: float = 0.5,
                 max_shift: float = 1.15, shift: bool = True) -> list[float]:
    timesteps = torch.linspace(1, 0, num_steps + 1)
    if shift:
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)
    return timesteps.tolist()


def denoise(model, img, img_ids, txt, txt_ids, vec, timesteps,
            guidance: float, t5_attn_mask: Optional[torch.Tensor] = None):
    guidance_vec = torch.full((img.shape[0],), guidance,
                              device=img.device, dtype=img.dtype)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = model(img=img, img_ids=img_ids, txt=txt, txt_ids=txt_ids, y=vec,
                     timesteps=t_vec, guidance=guidance_vec,
                     txt_attention_mask=t5_attn_mask)
        img = img + (t_prev - t_curr) * pred
    return img


def is_fp8(dt) -> bool:
    return dt in (torch.float8_e4m3fn, torch.float8_e4m3fnuz,
                  torch.float8_e5m2, torch.float8_e5m2fnuz)


DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp8": torch.float8_e4m3fn,
}


class DermaFluxGenerator:

    def __init__(self, root: Path, device: torch.device, compute_dtype: torch.dtype,
                 flux_dtype: torch.dtype, t5_dtype: torch.dtype,
                 lora_path: Path, lora_multiplier: float = 1.0):
        sys.path.insert(0, str(root))
        from library import flux_utils, strategy_flux, lora_flux
        from safetensors.torch import load_file

        self.flux_utils, self.strategy_flux = flux_utils, strategy_flux
        self.device = device
        self.compute_dtype = compute_dtype
        self.flux_dtype = flux_dtype
        self.t5_dtype = t5_dtype
        vram_gb = (torch.cuda.get_device_properties(device).total_memory / 1e9
                   if device.type == "cuda" else 0)
        self.text_device = (torch.device("cpu") if device.type == "cuda" and vram_gb < 32
                            else device)

        ck = root / "checkpoints"
        paths = {
            "flux": ck / "flux" / "flux1-dev.safetensors",
            "ae": ck / "flux" / "ae.safetensors",
            "clip_l": ck / "flux_text_encoders" / "clip_l.safetensors",
            "t5xxl": ck / "flux_text_encoders" / "t5xxl_fp16.safetensors",
        }
        missing = [str(p) for p in list(paths.values()) + [lora_path] if not p.exists()]
        if missing:
            sys.exit("Missing checkpoints:\n  " + "\n  ".join(missing) +
                     "\n\nSee dermaflux-synth/README.md for the download steps.")

        print("Loading CLIP-L …")
        self.clip_l = flux_utils.load_clip_l(
            str(paths["clip_l"]), compute_dtype, self.text_device).eval().to(self.text_device)
        print("Loading T5-XXL …")
        self.t5xxl = flux_utils.load_t5xxl(
            str(paths["t5xxl"]), t5_dtype, self.text_device).eval().to(self.text_device)
        print("Loading FLUX DiT …")
        self.is_schnell, self.model = flux_utils.load_flow_model(str(paths["flux"]), None, device)
        self.model.eval().to(device=device, dtype=flux_dtype)
        print("Loading VAE …")
        self.vae = flux_utils.load_ae(str(paths["ae"]), compute_dtype, device).eval().to(device)

        self.t5_max_len = 256 if self.is_schnell else 512
        self.tokenize_strategy = strategy_flux.FluxTokenizeStrategy(self.t5_max_len)
        self.encoding_strategy = strategy_flux.FluxTextEncodingStrategy()

        print(f"Applying LoRA {lora_path.name} (multiplier {lora_multiplier}) …")
        sd = load_file(str(lora_path))
        self.lora, _ = lora_flux.create_network_from_weights(
            lora_multiplier, None, self.vae, [self.clip_l, self.t5xxl],
            self.model, sd, True)
        self.lora.apply_to([self.clip_l, self.t5xxl], self.model)
        self.lora.load_state_dict(sd, strict=True)
        self.lora.eval().to(device)
        self.clip_l.to(self.text_device)
        self.t5xxl.to(self.text_device)
        self.lora_multiplier = lora_multiplier
        print("Ready.")

    def encode(self, prompt: str):
        toks = self.tokenize_strategy.tokenize(prompt)
        text_autocast_type = self.text_device.type
        with torch.no_grad():
            with torch.autocast(device_type=text_autocast_type, dtype=self.compute_dtype):
                l_pooled, _, _, _ = self.encoding_strategy.encode_tokens(
                    self.tokenize_strategy, [self.clip_l, None], toks)
            with torch.autocast(device_type=text_autocast_type, dtype=self.compute_dtype):
                _, t5_out, txt_ids, t5_mask = self.encoding_strategy.encode_tokens(
                    self.tokenize_strategy, [None, self.t5xxl], toks, False)
        if torch.isnan(l_pooled).any():
            raise ValueError("NaN in l_pooled — check dtype, fp8 is the usual cause")
        if torch.isnan(t5_out).any():
            raise ValueError("NaN in t5_out — check dtype, fp8 is the usual cause")
        l_pooled = l_pooled.to(self.device)
        t5_out = t5_out.to(self.device)
        txt_ids = txt_ids.to(self.device)
        return l_pooled, t5_out, txt_ids, t5_mask

    def encode_init_image(self, image, size: int):
        import numpy as np

        img = image.convert("RGB").resize((size, size))
        arr = (np.asarray(img).astype("float32") / 127.5) - 1.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        z = self.vae.encode(x.to(self.compute_dtype))
        return self.flux_utils.pack_latents(z.float())

    @torch.no_grad()
    def generate(self, prompt: str, seed: int, size: int, steps: int, guidance: float,
                init_latent: Optional[torch.Tensor] = None, strength: float = 1.0):
        import einops
        import numpy as np

        ph, pw = math.ceil(size / 16), math.ceil(size / 16)
        noise_dtype = torch.float32 if is_fp8(self.flux_dtype) else self.compute_dtype
        gen_device = self.device if self.device.type == "cuda" else torch.device("cpu")
        noise = torch.randn(1, ph * pw, 16 * 2 * 2, device=gen_device, dtype=noise_dtype,
                            generator=torch.Generator(device=gen_device).manual_seed(seed))
        noise = noise.to(self.device)

        img_ids = self.flux_utils.prepare_img_ids(1, ph, pw).to(self.device)
        l_pooled, t5_out, txt_ids, _ = self.encode(prompt)

        full_timesteps = get_schedule(steps, noise.shape[1], shift=not self.is_schnell)
        if strength >= 1.0 or init_latent is None:
            x0, timesteps = noise, full_timesteps
        else:
            cut = max(1, int(round(strength * steps)))
            timesteps = full_timesteps[-(cut + 1):]
            t0 = timesteps[0]
            x0 = t0 * noise + (1.0 - t0) * init_latent.to(self.device, noise.dtype)

        with torch.autocast(device_type=self.device.type, dtype=self.compute_dtype):
            x = denoise(self.model, x0, img_ids, t5_out, txt_ids, l_pooled,
                        timesteps, guidance, None)

        x = x.float()
        x = einops.rearrange(x, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                             h=ph, w=pw, ph=2, pw=2)
        x = self.vae.decode(x.to(self.compute_dtype))
        x = x.clamp(-1, 1).permute(0, 2, 3, 1)
        return (127.5 * (x + 1.0)).float().cpu().numpy().astype(np.uint8)[0]


MANIFEST_COLUMNS = [
    "filename", "prompt_id", "variant", "class", "slug", "skin_tone", "seed",
    "steps", "guidance", "size", "lora", "lora_multiplier", "flux_dtype",
    "n_t5_tokens", "body_source", "body_image_id", "dermaflux_sha",
    "parent_path", "strength", "generated_at", "seconds",
]


def git_sha(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", type=Path, required=True, help="a prompts_*.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dermaflux-root", type=Path, default=Path("../dermaflux"))
    ap.add_argument("--lora", type=Path, default=None,
                    help="default: <root>/checkpoints/flux_lora_64_64.safetensors")
    ap.add_argument("--lora-multiplier", type=float, default=1.0)
    ap.add_argument("--size", type=int, default=512, choices=[128, 256, 512],
                    help="gradio_app.py:524 exposes only these; square only")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES),
                    help="bf16 needs ~24GB for the DiT (A100). Use fp8 on L4.")
    ap.add_argument("--device", default=None, help="default: cuda if available")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--classes", nargs="*", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="skip prompts whose output file already exists")
    ap.add_argument("--parent-root", type=Path, default=None,
                    help="root dir real parent images are resolved relative to, "
                         "when a prompt row has 'parent_path' and 'strength' < 1.0")
    args = ap.parse_args()

    root = args.dermaflux_root.resolve()
    lora = args.lora or (root / "checkpoints" / "flux_lora_64_64.safetensors")
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if dev.type == "cpu":
        print("[warn] running on CPU — this will take hours per image")

    compute_dtype = DTYPES[args.dtype]
    flux_dtype = compute_dtype
    t5_dtype = compute_dtype
    if is_fp8(compute_dtype):
        compute_dtype = torch.bfloat16
        t5_dtype = torch.bfloat16

    prompts = [json.loads(l) for l in args.prompts.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.classes:
        prompts = [p for p in prompts if p["class"] in args.classes]
    if args.limit:
        prompts = prompts[:args.limit]
    if not prompts:
        sys.exit("no prompts selected")

    img_dir = args.out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.csv"

    done = set()
    if args.resume and manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as fh:
            done = {r["prompt_id"] for r in csv.DictReader(fh)}
        print(f"resuming: {len(done)} already generated")
    todo = [p for p in prompts if p["prompt_id"] not in done]
    print(f"{len(todo)} images to generate at {args.size}px, {args.steps} steps")
    if not todo:
        return

    gen = DermaFluxGenerator(root, dev, compute_dtype, flux_dtype, t5_dtype,
                             lora, args.lora_multiplier)
    sha = git_sha(root)

    from PIL import Image
    new_file = not manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        if new_file:
            writer.writeheader()

        for i, p in enumerate(todo, 1):
            t0 = time.time()
            strength = float(p.get("strength", 1.0))
            init_latent = None
            if strength < 1.0:
                parent_path = p.get("parent_path")
                if not parent_path:
                    sys.exit(f"{p['prompt_id']}: strength={strength} < 1.0 but no "
                             f"'parent_path' field")
                full_path = (args.parent_root / parent_path) if args.parent_root else Path(parent_path)
                parent_img = Image.open(full_path)
                init_latent = gen.encode_init_image(parent_img, args.size)
            arr = gen.generate(p["text"], int(p["seed"]), args.size,
                               args.steps, args.guidance,
                               init_latent=init_latent, strength=strength)
            fname = f"{p['prompt_id']}.png"
            Image.fromarray(arr).save(img_dir / fname)
            dt = time.time() - t0

            writer.writerow({
                "filename": fname, "prompt_id": p["prompt_id"],
                "variant": p.get("variant"), "class": p["class"],
                "slug": p.get("slug") or "", "skin_tone": p.get("skin_tone") or "",
                "seed": p["seed"], "steps": args.steps, "guidance": args.guidance,
                "size": args.size, "lora": lora.name,
                "lora_multiplier": args.lora_multiplier, "flux_dtype": args.dtype,
                "n_t5_tokens": p.get("n_t5_tokens", ""),
                "body_source": p.get("body_source", ""),
                "body_image_id": p.get("body_image_id", ""),
                "dermaflux_sha": sha,
                "parent_path": p.get("parent_path", ""),
                "strength": strength,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "seconds": f"{dt:.1f}",
            })
            fh.flush()
            os.fsync(fh.fileno())

            eta = (len(todo) - i) * dt / 60
            print(f"[{i}/{len(todo)}] {p['prompt_id']} {dt:.1f}s  eta {eta:.0f}min")

    (args.out / "prompts_used.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in prompts),
        encoding="utf-8")
    print(f"\ndone -> {args.out}")


if __name__ == "__main__":
    main()
