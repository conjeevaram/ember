"""
GATE CHECK -- run this FIRST on the 5090 box, before anything else.

The RTX 5090 is Blackwell (compute capability sm_120). JAX only sees it with a
recent jaxlib built for sm_120; otherwise MJX silently falls back to CPU and
"training" crawls. This script fails loudly if the GPU isn't actually usable by
JAX. Do not start a training run until this prints your 5090 and PASS.
"""
import sys


def main():
    try:
        import jax
        import jax.numpy as jnp
    except Exception as e:
        print("FAIL: could not import jax:", e)
        print("  -> pip install -U 'jax[cuda12]'")
        sys.exit(1)

    print("jax version:", jax.__version__)
    devices = jax.devices()
    print("jax.devices():", devices)

    gpus = [d for d in devices if d.platform == "gpu"]
    if not gpus:
        print("\nFAIL: JAX sees NO GPU. On Blackwell this usually means jaxlib "
              "is too old for sm_120.")
        print("  -> pip install -U 'jax[cuda12]'   (need a build with sm_120)")
        print("  -> verify driver: nvidia-smi (CUDA 12.8+/13.x)")
        sys.exit(1)

    # Actually run a kernel on the GPU -- import-seeing-a-device isn't enough.
    try:
        x = jnp.ones((4096, 4096))
        y = (x @ x).sum().block_until_ready()
        dev = y.devices() if hasattr(y, "devices") else {gpus[0]}
        print(f"\nGPU matmul ok (result={float(y):.1f}) on {dev}")
    except Exception as e:
        print("\nFAIL: GPU compute errored:", e)
        sys.exit(1)

    print("\nPASS -- JAX can train on the GPU. Proceed to train_g1.py.")


if __name__ == "__main__":
    main()
