GPUS=1
export CUDA_VISIBLE_DEVICES=$GPUS

BASE_ROOT=drive/Shareddrives/Image-Text-Retrieval/deepcca
IMAGE_DIR=/content
ANNO_DIR=$BASE_ROOT/data/processed_data
CKPT_DIR=$BASE_ROOT/data/model_data
LOG_DIR=$BASE_ROOT/data/logs
# PRETRAINED_PATH=$BASE_ROOT/pretrained_models/mobilenet.tar
#PRETRAINED_PATH=$BASE_ROOT/resnet50.pth
IMAGE_MODEL=mobilenet_v1
lr=0.0002
num_epoches=300
batch_size=16
lr_decay_ratio=0.9
epoches_decay=80_150_200

# python3 $BASE_ROOT/my_code/train.py \
python3 /content/train.py \
    --CMPC \
    --CMPM \
    --bidirectional \
    --image_model $IMAGE_MODEL \
    --log_dir $LOG_DIR/lr-$lr-decay-$lr_decay_ratio-batch-$batch_size \
    --checkpoint_dir $CKPT_DIR/lr-$lr-decay-$lr_decay_ratio-batch-$batch_size \
    --image_dir $IMAGE_DIR \
    --anno_dir $ANNO_DIR \
    --batch_size $batch_size \
    --gpus $GPUS \
    --num_epoches $num_epoches \
    --lr $lr \
    --lr_decay_ratio $lr_decay_ratio \
    --epoches_decay ${epoches_decay} \
    --num_classes 12305

