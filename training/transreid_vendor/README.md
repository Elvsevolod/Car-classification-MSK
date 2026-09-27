# Pinned TransReID backbone

Source: [damo-cv/TransReID](https://github.com/damo-cv/TransReID/tree/dec55046fcdfadee14e2c28e2df89305d8f7557a),
revision `dec55046fcdfadee14e2c28e2df89305d8f7557a`.
`vit_pytorch.py` is the upstream `model/backbones/vit_pytorch.py`; `LICENSE` is its MIT license.
The sole compatibility edit replaces removed `torch._six.container_abcs` with
`collections.abc`. No architecture edit is made in this vendored file.

The project wrapper is `training/transreid_model.py`, not the complete upstream trainer.
It implements the published global/JPM branch construction, shuffle group 2 / shift 8,
four local parts, pre-BN retrieval features, and half global / half mean-local losses.
The wrapper uses 384-dimensional DeiT-Small heads (including JPM), disables camera/view SIE,
and uses our own optimizer, augmentations and loss grid. This is an adapted ReID experiment,
not a claim to reproduce the authors' paper metrics or full VeRi recipe.

## Public initialization

The [official TransReID README](https://github.com/damo-cv/TransReID/blob/dec55046fcdfadee14e2c28e2df89305d8f7557a/README.md)
links [DeiT-Small distilled ImageNet weights](https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth).

- Filename: `deit_small_distilled_patch16_224-649709d9.pth`
- Size: 89,795,170 bytes
- SHA256: `649709d94f9fd790ea86c16f99d788e709b86a1f64315a19d887895f9948fb09`
- Download needs no login; no Hugging Face mirror, DINOv3 access or new ReID dataset.
- DeiT repository revision inspected: `7e160fe43f0252d17191b71cbb5826254114ea5b`.
- [DeiT LICENSE](https://github.com/facebookresearch/deit/blob/7e160fe43f0252d17191b71cbb5826254114ea5b/LICENSE)
  is included as `DEIT_LICENSE` (Apache 2.0). Keep both upstream license files with redistributed code.

Strict initialization verifies the entire file, drops exactly `dist_token`,
`head.{weight,bias}`, `head_dist.{weight,bias}`, removes the distillation position token,
and interpolates the 14×14 positional grid to 16×16 using the upstream bilinear helper.
All remaining 150 backbone tensors must match with `strict=True`.
The upstream permissive `load_param` method is deliberately not used.

Public provenance and licenses are recorded; this does not certify every item of a future
contest delivery, plate-signal invariance, or Linux/GPU deployment compatibility.
