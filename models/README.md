# Vehicle OSNet weights

## Active ensemble: MVP_fusion_v25

The default profile combines this legacy MVP encoder with three frozen R1
encoders in `models/frozen/`: 2048-D output, 50/50 MVP/R1 ranking fusion,
separate R1 raw-top1 candidate and threshold `0.534365177154541`.
All four ONNX files are committed as ordinary Git blobs. Profile definitions,
member paths and checksums are in `profiles.json` and the referenced bundles.
See [MODEL_REPORT.md](../docs/MODEL_REPORT.md) and the [current run guide](../README.md).
The sections below document the legacy member and public initialization.

## Legacy MVP member checkpoint

File: `osnet_ain_x1_0_vehicle_reid_hpo_best_map.onnx`.

OSNet-AIN x1.0 encoder fine-tuned on the 925 development-training identities.
The configuration was selected by staged Optuna HPO with BNNeck and supervised
contrastive loss. The active best-mAP checkpoint was selected at epoch 5 using
the identity-disjoint local validation protocol. It is a development model,
not the final train+validation model.

Size: 8,754,346 bytes.

SHA-256: `01466f503232467224774b6e3bafe6c4393b1de30b47b1bd714908a62f2006a2`

SHA-384 (checked at every runtime load):
`4832ca8134b31f84ec52b9a6a72aa90f9e55e0b8ab7d52d02820040536677d712df39b9c9604d6da21ee7bc823db6818`

PyTorch→ONNX verification on a real crop: output `(1, 512)`, maximum
absolute difference `1.22e-5`, cosine similarity `1.00000012`.

## Stock initialization checkpoint

Model: `vehicle-reid-0001`, OSNet-AIN x1.0, 512-dimensional descriptor.
Public pretrained checkpoint; no training or fine-tuning on this project's data.
The original ONNX bytes are preserved. CPU inference uses ONNX Runtime.

- Model documentation: https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/vehicle-reid-0001/README.md
- Checksum source: https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/vehicle-reid-0001/model.yml
- Download: https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/vehicle-reid-0001/osnet_ain_x1_0_vehicle_reid.onnx
- Training source: https://github.com/sovrasov/deep-person-reid/tree/vehicle_reid
- Preprocessing source: https://github.com/sovrasov/deep-person-reid/blob/vehicle_reid/torchreid/data/transforms.py
- Original model license: MIT, see `LICENSE.osnet`.

Size: 8,836,743 bytes.

SHA-256: `4aaad3e5db648618b0df3d2ff21c61323985ff9e50194c3d2edd4fb87c92d91f`

SHA-384:
`0515ce72f653c39780d5b87dfed7255d396dd2b1e8b6e91fbaacdfad1da189166343157273c02f3b0fede3050ef7abb7`

Preprocessing: EXIF orientation → RGB → strict `(x,y,w,h)` crop → PIL bilinear
resize 208×208 → float32 / 255 → ImageNet mean `(0.485,0.456,0.406)` and
std `(0.229,0.224,0.225)` → NCHW. This is the original ONNX, not the
OpenVINO-converted BGR model. Output is L2-normalized in Python.

No OCR, explicit plate features or detector is used by the deployed ensemble.
The stock checkpoint remains frozen and is used only to initialize new training runs.
Changing the active checkpoint requires updating its metadata, recalculating gallery
embeddings, and calibrating the refusal threshold.
