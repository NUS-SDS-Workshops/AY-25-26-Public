# make_request.py
import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=32)
    parser.add_argument("--output_path", type=str, default="outputs/request.json")
    args = parser.parse_args()

    data_root = Path(args.data_dir)

    # ImageFolder-style layout: data_root/<class_name>/*.png
    img_paths = list(data_root.rglob("*.png"))
    chosen = random.sample(img_paths, k=min(args.n_samples, len(img_paths)))

    # Where to save images for human verification
    samples_dir = Path("outputs/samples")
    samples_dir.mkdir(parents=True, exist_ok=True)

    batch = []
    manifest = []  # keeps index -> filename mapping

    for i, src_path in enumerate(chosen):
        # Read + resize + flatten for request payload
        img = Image.open(src_path).convert("L").resize((28, 28))
        arr = (np.asarray(img, dtype=np.float32) / 255.0).reshape(-1)  # 784 floats
        batch.append(arr.tolist())

        # Save the image we actually used (resized) so humans see the same thing the model saw
        out_name = f"{i:03d}.png"
        out_path = samples_dir / out_name
        img.save(out_path)

        manifest.append({"idx": i, "sample_file": str(out_path)})

    # Write request.json (endpoint payload)
    out_req = Path(args.output_path)
    out_req.parent.mkdir(parents=True, exist_ok=True)
    out_req.write_text(json.dumps({"inputs": batch}))

    # Write manifest.json (so notebook knows which images correspond to which row)
    Path("outputs/manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Wrote request: {out_req}")
    print("Wrote manifest: outputs/manifest.json")
    print(f"Wrote samples: {samples_dir} ({len(manifest)} images)")


if __name__ == "__main__":
    main()
