# Frozen vehicle OSNet baseline

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

SHA-384 (checked at every model load):
`0515ce72f653c39780d5b87dfed7255d396dd2b1e8b6e91fbaacdfad1da189166343157273c02f3b0fede3050ef7abb7`

Preprocessing: EXIF orientation → RGB → strict `(x,y,w,h)` crop → PIL bilinear
resize 208×208 → float32 / 255 → ImageNet mean `(0.485,0.456,0.406)` and
std `(0.229,0.224,0.225)` → NCHW. This is the original ONNX, not the
OpenVINO-converted BGR model. Output is L2-normalized in Python.

No OCR, plate features, detector, or additional embedding model is used.
The checkpoint remains frozen; replacing it requires updating model metadata,
preprocessing if necessary, recalculating embeddings and calibrating the threshold.
