# Used for training the model

import os

GPUS = '1'

os.system('export CUDA_VISIBLE_DEVICES='+GPUS)


BASE_ROOT='/content/Image_Text_Retrieval/deep_cmpl_model'
BASE_ROOT_2='drive/Shareddrives/Image-Text-Retrieval/tempckpt'
IMAGE_DIR='/content'
ANNO_DIR=BASE_ROOT+'/data/processed_data'
CKPT_DIR=BASE_ROOT_2+'/data/model_data'
LOG_DIR=BASE_ROOT_2+'/data/logs'
IMAGE_MODEL='efficient_net'
lr='0.0002'
num_epoches='300'
batch_size='16'
lr_decay_ratio='0.9'
epoches_decay='80_150_200'

models_path='{CKPT_DIR}/lr-{lr}-decay-{lr_decay_ratio}-batch-{batch_size}'.format(CKPT_DIR=CKPT_DIR,lr=lr,lr_decay_ratio=lr_decay_ratio,batch_size=batch_size)

def get_model_path():
    # print(models_path)
    MODEL_PATH=None
    if os.path.exists(models_path):
        l=os.listdir(models_path)
        try:
            l.remove('model_best')
        except: 
            pass
        # print(l)
        if len(l)==0:
            return None
        # print(len(l))
        
        l.sort(key=lambda x:int(x.split('.')[0]))
        if len(l)>=2:
            x=l[-2]
        else:
            x=l[-1]
        MODEL_PATH=os.path.join(models_path,x)

    return MODEL_PATH

string = 'python3 {BASE_ROOT}/code/train.py --CMPM --bidirectional --image_model {IMAGE_MODEL} --log_dir {LOG_DIR}/lr-{lr}-decay-{lr_decay_ratio}-batch-{batch_size} --checkpoint_dir {CKPT_DIR}/lr-{lr}-decay-{lr_decay_ratio}-batch-{batch_size} --image_dir {IMAGE_DIR} --anno_dir {ANNO_DIR} --batch_size {batch_size} --gpus {GPUS} --num_epoches {num_epoches} --lr {lr} --lr_decay_ratio {lr_decay_ratio} --epoches_decay {epoches_decay} --num_images 12305'.format(BASE_ROOT=BASE_ROOT, IMAGE_DIR=IMAGE_DIR, IMAGE_MODEL=IMAGE_MODEL, ANNO_DIR=ANNO_DIR, CKPT_DIR=CKPT_DIR, LOG_DIR=LOG_DIR, lr=lr, num_epoches=num_epoches, batch_size=batch_size, lr_decay_ratio=lr_decay_ratio, epoches_decay=epoches_decay, GPUS=GPUS)

MODEL_PATH = get_model_path()
if(MODEL_PATH!=None):
  string += ' --resume --model_path {MODEL_PATH}'.format(MODEL_PATH=MODEL_PATH)

os.system(string)

