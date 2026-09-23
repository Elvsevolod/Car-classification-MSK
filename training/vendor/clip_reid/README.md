# Vendored CLIP-ReID primitives

Source: https://github.com/Syliz517/CLIP-ReID
Commit: `eb1898b72c882875f478bebfc6d41644eece0a5d`.

- `model.py`: upstream `model/clip/model.py`, code unchanged; a final newline was added.
  Original SHA-256: `31990f19a71d646b9d807d95f0a0398ddeebef985b1207e8c4cf8c6b735e59ad`.
  Local SHA-256: `6cd4621f8fb7d20cf80a3362f939ceb6c88d7cbf11b10d841da4a7d539dc65a2`.
- `veri_config.yml`: upstream `configs/veri/vit_prom.yml`; reference only, not an executable local config.
- `LICENSE`: Syliz517 MIT notice; `LICENSE_OPENAI`: underlying OpenAI CLIP MIT notice
  from https://github.com/openai/CLIP/blob/main/LICENSE.

Only `VisionTransformer`, `Transformer`, `LayerNorm` are imported locally.
The adapter/trainer live outside this directory; no upstream CUDA training scripts
or automatic base-model downloads are executed.
