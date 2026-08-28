import argparse
import sys
from pathlib import Path

import numpy as np

from asv_infer import SpeakerVerifier


def main():
    ap = argparse.ArgumentParser(description="Trích embedding người nói từ file audio")
    ap.add_argument("audio", nargs="+", help="file audio (wav/mp3/flac/m4a...)")
    ap.add_argument("--out", "-o", help="file kết quả (.npy cho 1 file, .npz cho nhiều file)")
    ap.add_argument("--device", help="cpu hoặc cuda (mặc định: tự chọn)")
    args = ap.parse_args()

    # File thứ 2 là đường dẫn output nếu nó có đuôi .npy/.npz (cho tiện dùng nhanh)
    if args.out is None and len(args.audio) == 2 and args.audio[1].endswith((".npy", ".npz")):
        args.out = args.audio.pop()

    asv = SpeakerVerifier(device=args.device)
    embs = {}
    for path in args.audio:
        try:
            e = asv.embed(path).numpy()
        except Exception as ex:
            print(f"LỖI {path}: {type(ex).__name__}: {ex}", file=sys.stderr)
            continue
        embs[Path(path).name] = e
        print(f"{path}: {e.shape[0]} chiều, norm={np.linalg.norm(e):.4f}")
        if not args.out:
            print("  " + " ".join(f"{v:+.5f}" for v in e[:8]) + " ...")

    if not embs:
        return 1
    if args.out:
        if args.out.endswith(".npz") or len(embs) > 1:
            out = args.out if args.out.endswith(".npz") else args.out + ".npz"
            np.savez(out, **embs)
        else:
            out = args.out
            np.save(out, next(iter(embs.values())))
        print(f"-> đã lưu {len(embs)} vector vào {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
