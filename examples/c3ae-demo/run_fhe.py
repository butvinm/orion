#!/usr/bin/env python3
"""Standalone FHE benchmark for C3AE age verification.

Runs cleartext baseline on full test set, then FHE inference on a subset.
Measures memory and timing for each phase.

Usage:
    python run_fhe.py --weights weights.pth --data-dir ./data/UTKFace
    python run_fhe.py --weights weights.pth --data-dir ./data/UTKFace --samples 5
"""

import argparse
import resource
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from lattigo.ckks import Encoder, Parameters
from lattigo.rlwe import (
    BootstrapParams,
    Decryptor,
    Encryptor,
    KeyGenerator,
    MemEvaluationKeySet,
)
from lattigo.rlwe import (
    Ciphertext as RLWECiphertext,
)
from model import C3AE
from orion_evaluator import Evaluator, Model
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

AGE_MAX = 100


class UTKFaceDataset(Dataset):
    def __init__(self, data_dir, img_size=64, age_threshold=18):
        self.img_size = img_size
        self.samples = []
        for img_path in Path(data_dir).glob("*.jpg*"):
            try:
                age = min(max(int(img_path.name.split("_")[0]), 0), AGE_MAX)
                is_adult = 1.0 if age >= age_threshold else 0.0
                self.samples.append((img_path, age, is_adult))
            except (ValueError, IndexError):
                continue
        if not self.samples:
            raise ValueError(f"No samples found in {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, age, is_adult = self.samples[idx]
        img = Image.open(img_path).convert("RGB").resize((self.img_size, self.img_size))
        img = np.array(img, dtype=np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img, torch.tensor([is_adult], dtype=torch.float32), age


def get_rss_mb():
    """Current RSS in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="weights.pth")
    parser.add_argument(
        "--model",
        type=str,
        default="model.orion",
        help="Pre-compiled .orion model (skip compilation)",
    )
    parser.add_argument("--stride", type=int, default=2, choices=[1, 2])
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--data-dir", type=str, default="./data/UTKFace")
    args = parser.parse_args()

    measurements = {}

    # Load model
    net = C3AE(img_size=64, first_stride=args.stride)
    state_dict = torch.load(args.weights, map_location="cpu", weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"C3AE (stride={args.stride}): {n_params:,} parameters")

    # Load dataset
    dataset = UTKFaceDataset(args.data_dir, img_size=64)
    train_size = int(0.70 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    _, _, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Test set: {len(test_set)} samples")

    # Cleartext baseline
    print("\n--- Cleartext Baseline ---")
    loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)
    all_probs, all_targets = [], []
    net.eval()
    with torch.no_grad():
        for images, targets, _ in loader:
            probs = torch.sigmoid(net(images))
            all_probs.extend(probs.cpu().squeeze().tolist())
            all_targets.extend(targets.cpu().squeeze().tolist())
    probs_arr = np.array(all_probs)
    targets_arr = np.array(all_targets)
    pred_adult = probs_arr >= 0.5
    true_adult = targets_arr >= 0.5
    fpr = pred_adult[~true_adult].mean() if (~true_adult).sum() > 0 else 0
    fnr = (~pred_adult[true_adult]).mean() if true_adult.sum() > 0 else 0
    accuracy = (pred_adult == true_adult).mean()
    print(f"FPR: {fpr * 100:.1f}%, FNR: {fnr * 100:.1f}%, Accuracy: {accuracy * 100:.1f}%")
    measurements["cleartext"] = {"fpr": fpr, "fnr": fnr, "accuracy": accuracy}

    # Select diverse samples for FHE
    minors, adults = [], []
    for idx in range(len(test_set)):
        _, target, age = test_set[idx]
        entry = {"index": idx, "age": age, "is_adult": target.item() >= 0.5}
        (adults if entry["is_adult"] else minors).append(entry)
    n_min = min(len(minors), args.samples // 2)
    n_adu = min(len(adults), args.samples - n_min)
    n_min = min(len(minors), args.samples - n_adu)
    minors.sort(key=lambda x: x["age"])
    adults.sort(key=lambda x: x["age"])

    def pick(lst, n):
        if n >= len(lst):
            return lst
        step = len(lst) / n
        return [lst[int(i * step)] for i in range(n)]

    selected = pick(minors, n_min) + pick(adults, n_adu)
    samples = []
    for entry in selected:
        img, target, age = test_set[entry["index"]]
        samples.append({"image": img.unsqueeze(0), "age": age, "is_adult": target.item()})
    print(
        f"\nSelected {len(samples)} FHE samples: "
        f"{sum(1 for s in samples if s['is_adult'] < 0.5)} minors + "
        f"{sum(1 for s in samples if s['is_adult'] >= 0.5)} adults"
    )

    # Load pre-compiled model (compilation uses 125+ GB, must be done separately)
    print(f"\n--- Loading Pre-compiled Model ({args.model}) ---")
    t0 = time.time()
    with open(args.model, "rb") as f:
        model_bytes = f.read()
    model = Model.load(model_bytes)
    del model_bytes  # Free ~5.5 GB
    params_dict, manifest, input_level = model.client_params()
    load_time = time.time() - t0
    print(f"Model load: {load_time:.2f}s")

    with ExitStack() as stack:
        stack.enter_context(model)

        # Keygen
        print("\n--- Key Generation ---")
        rss_before = get_rss_mb()
        t0 = time.time()

        params = stack.enter_context(Parameters.from_dict(params_dict))
        kg = KeyGenerator(params)
        sk = stack.enter_context(kg.gen_secret_key())
        pk = stack.enter_context(kg.gen_public_key(sk))

        rlk = kg.gen_relin_key(sk) if manifest["needs_rlk"] else None
        gks = [kg.gen_galois_key(sk, int(ge)) for ge in manifest["galois_elements"]]
        evk = MemEvaluationKeySet(rlk=rlk, galois_keys=gks)
        keys_bytes = evk.marshal_binary()

        # Free keygen Go objects before creating evaluator so Go can
        # reuse heap pages (two .so files = two Go runtimes, handles
        # can't pass between them, so we must serialize).
        evk.close()
        for gk in gks:
            gk.close()
        del gks
        if rlk:
            rlk.close()
        kg.close()

        # Bootstrap keys
        bootstrap_slots = manifest.get("bootstrap_slots", [])
        btp_keys_bytes = None
        if bootstrap_slots:
            print(f"Generating bootstrap keys (slots: {bootstrap_slots})...")
            boot_logp = manifest.get("boot_logp", [61] * 8)
            btp_logn = manifest.get("btp_logn", params_dict.get("logn", 14))
            min_slots = min(bootstrap_slots)
            log_slots = int(np.log2(min_slots))
            with BootstrapParams(
                params,
                logn=btp_logn,
                logp=boot_logp,
                h=192,
                log_slots=log_slots,
            ) as btp:
                _evk, btp_keys = btp.gen_eval_keys(sk)
                btp_keys_bytes = btp_keys.marshal_binary()
                _evk.close()
                btp_keys.close()
            print(f"Bootstrap keys: {len(btp_keys_bytes):,} bytes")

        keygen_time = time.time() - t0
        rss_after = get_rss_mb()
        print(f"Keygen: {keygen_time:.2f}s")
        print(f"Eval keys: {len(keys_bytes):,} bytes ({len(keys_bytes) / (1024**3):.2f} GB)")
        if btp_keys_bytes:
            btp_gb = len(btp_keys_bytes) / (1024**3)
            print(f"Bootstrap keys: {len(btp_keys_bytes):,} bytes ({btp_gb:.2f} GB)")
        print(f"RSS delta: {rss_after - rss_before:.1f} MB")
        measurements["keygen"] = {
            "time": keygen_time,
            "eval_keys_bytes": len(keys_bytes),
            "btp_keys_bytes": len(btp_keys_bytes) if btp_keys_bytes else 0,
            "rss_delta_mb": rss_after - rss_before,
        }

        # Create evaluator — Go runtime reuses heap freed by keygen above
        t0 = time.time()
        evaluator = stack.enter_context(
            Evaluator(params_dict, keys_bytes, btp_keys_bytes=btp_keys_bytes)
        )
        del keys_bytes, btp_keys_bytes
        eval_init_time = time.time() - t0
        print(f"Evaluator init: {eval_init_time:.2f}s")

        encoder = stack.enter_context(Encoder(params))
        encryptor = stack.enter_context(Encryptor(params, pk))
        decryptor = stack.enter_context(Decryptor(params, sk))
        max_slots = params.max_slots()
        scale = params.default_scale()

        # FHE inference
        print(f"\n--- FHE Inference ({len(samples)} samples) ---")
        fhe_times, enc_times, dec_times = [], [], []
        fhe_outputs = []
        cleartext_outputs = []
        last_ct_bytes_size = 0

        for i, sample in enumerate(samples):
            # Cleartext reference
            with torch.no_grad():
                clear_prob = torch.sigmoid(net(sample["image"])).item()
            cleartext_outputs.append(clear_prob)

            # Encrypt — split into chunks of max_slots
            t0 = time.time()
            flat = sample["image"].flatten().double().tolist()
            ct_bytes_list = []
            for chunk_start in range(0, len(flat), max_slots):
                chunk = flat[chunk_start : chunk_start + max_slots]
                padded = chunk + [0.0] * (max_slots - len(chunk))
                with (
                    encoder.encode(padded, input_level, scale) as pt,
                    encryptor.encrypt_new(pt) as ct,
                ):
                    ct_bytes = ct.marshal_binary()
                    ct_bytes_list.append(ct_bytes)
                    last_ct_bytes_size = len(ct_bytes)
            enc_time = time.time() - t0
            enc_times.append(enc_time)

            # FHE forward
            t0 = time.time()
            result_bytes_list = evaluator.forward(model, ct_bytes_list)
            result_bytes = result_bytes_list[0]
            inf_time = time.time() - t0
            fhe_times.append(inf_time)

            # Decrypt
            t0 = time.time()
            with (
                RLWECiphertext.unmarshal_binary(result_bytes) as result_ct,
                decryptor.decrypt_new(result_ct) as result_pt,
            ):
                decoded = encoder.decode(result_pt, max_slots)
            dec_time = time.time() - t0
            dec_times.append(dec_time)

            fhe_logit = decoded[0]
            fhe_prob = 1 / (1 + np.exp(-fhe_logit))
            fhe_outputs.append(fhe_prob)
            mae = abs(clear_prob - fhe_prob)

            print(
                f"  Sample {i + 1}: age={sample['age']:2d}, "
                f"clear={clear_prob:.4f}, fhe={fhe_prob:.4f}, mae={mae:.6f}, "
                f"enc={enc_time:.3f}s, infer={inf_time:.3f}s, dec={dec_time:.3f}s"
            )

        # Summary
        print(f"\n{'=' * 70}")
        print("  MEASUREMENTS SUMMARY")
        print(f"{'=' * 70}")
        print(f"Model: C3AE (stride={args.stride}), {n_params:,} parameters")

        print(f"\n--- Cleartext (full test set, {len(test_set)} samples) ---")
        m = measurements["cleartext"]
        fpr_pct = m["fpr"] * 100
        fnr_pct = m["fnr"] * 100
        acc_pct = m["accuracy"] * 100
        print(f"  FPR: {fpr_pct:.1f}%, FNR: {fnr_pct:.1f}%, Accuracy: {acc_pct:.1f}%")

        if "compilation" in measurements:
            print("\n--- Compilation ---")
            m = measurements["compilation"]
            print(f"  Fit time:     {m['fit_time']:.2f}s")
            print(f"  Compile time: {m['compile_time']:.2f}s")
            print(f"  Model size:   {m['model_bytes']:,} bytes")
            print(f"  Peak Python memory: {m['peak_mem_mb']:.1f} MB")

        print("\n--- Key Generation ---")
        m = measurements["keygen"]
        print(f"  Time:         {m['time']:.2f}s")
        evk_gb = m["eval_keys_bytes"] / (1024**3)
        print(f"  Eval keys:    {m['eval_keys_bytes']:,} bytes ({evk_gb:.2f} GB)")
        if m["btp_keys_bytes"]:
            btp_gb = m["btp_keys_bytes"] / (1024**3)
            print(f"  BTP keys:     {m['btp_keys_bytes']:,} bytes ({btp_gb:.2f} GB)")
        print(f"  RSS delta:    {m['rss_delta_mb']:.1f} MB")

        print(f"\n--- FHE Inference ({len(samples)} samples) ---")
        avg_enc = np.mean(enc_times)
        avg_inf = np.mean(fhe_times)
        avg_dec = np.mean(dec_times)
        std_inf = np.std(fhe_times)
        avg_mae = np.mean(
            [abs(cleartext_outputs[i] - fhe_outputs[i]) for i in range(len(samples))]
        )
        print(f"  Avg encrypt:  {avg_enc:.3f}s")
        print(f"  Avg inference:{avg_inf:.2f}s +/- {std_inf:.2f}s")
        print(f"  Avg decrypt:  {avg_dec:.3f}s")
        print(f"  Avg MAE:      {avg_mae:.6f}")
        print(f"  Ciphertext:   ~{last_ct_bytes_size:,} bytes")
        print(f"  Peak RSS:     {get_rss_mb():.0f} MB")

        measurements["fhe"] = {
            "avg_encrypt": avg_enc,
            "avg_inference": avg_inf,
            "std_inference": std_inf,
            "avg_decrypt": avg_dec,
            "avg_mae": avg_mae,
            "peak_rss_mb": get_rss_mb(),
        }


if __name__ == "__main__":
    main()
