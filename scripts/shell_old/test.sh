GPUS=0
export CUDA_VISIBLE_DEVICES=$GPUS

BASE_ROOT=/mnt/c/Users/RAGHAV/Raghav_Goyal/Sem_6/Machine_Learning/project/Image_Text_Retrieval
IMAGE_DIR=/mnt/c/Users/RAGHAV/Raghav_Goyal/Sem_6/Machine_Learning/project/Image_Text_Retrieval/dataset
ANNO_DIR=$BASE_ROOT/data/processed_data
CKPT_DIR=$BASE_ROOT/data/model_data
LOG_DIR=$BASE_ROOT/data/logs
IMAGE_MODEL = mobilenet_v1

test_batch_size=64

# Train parameters for locating checkpoint directory
lr=0.0002
lr_decay_ratio=0.9
train_batch_size=16
epoches_decay=80_150_200


python3 $BASE_ROOT/test.py \
    --bidirectional \
    --batch_size $test_batch_size \
    --model_path $CKPT_DIR/lr-$lr-decay-$lr_decay_ratio-batch-$train_batch_size \
    --image_model $IMAGE_MODEL \
    --log_dir $LOG_DIR/lr-$lr-decay-$lr_decay_ratio-batch-$batch_size \
    --image_dir $IMAGE_DIR \
    --anno_dir $ANNO_DIR \
    --gpus $GPUS 
