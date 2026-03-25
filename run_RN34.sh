python ai/demo.py \
    --config-file cubercnn://omni3d/cubercnn_Res34_FPN.yaml \
    --threshold 0.25 \
    --display \
    --focal-length 0.0 \
    MODEL.WEIGHTS cubercnn://omni3d/cubercnn_Res34_FPN.pth \
    MODEL.DEVICE cpu