import os
import sys
from PIL import Image
import shutil
import pickle
import string
import time
from imageio import imread
import numpy as np
import torch
import torchvision.transforms as transforms
from test_config import config
from config import network_config
from werkzeug.utils import secure_filename
from flask import Flask, render_template, flash, request, redirect, url_for
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
app.debug = True
run_with_ngrok(app)
network = None
model_path = 'saved_model/299.pth.tar'
test_sort_path = 'saved_model/test_sort.pkl'
word_to_index_path = 'saved_model/word_to_index.pkl'
test_sort = None
word_to_index = None
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def load_model(args):
    network, _ = network_config(args,'test', None, True, model_path, False)
    network.eval()
    with open(test_sort_path, 'rb') as f_pkl:
        test_sort = pickle.load(f_pkl)
    with open(word_to_index_path, 'rb') as f_2_pkl:
        word_to_index = pickle.load(f_2_pkl)
        word_to_index = {k.lower(): v for k, v in word_to_index.items()}
    return network, test_sort, word_to_index

def index_to_word(indexed_caption):
    index_to_word_dict = {value:key for key, value in word_to_index.items()}

    caption = []
    for token in indexed_caption:
        if token in index_to_word_dict:
            caption.append(index_to_word_dict[token])
        else:
            caption.append(' ')

    return ' '.join(caption[1:-1])

def retrieve_captions():
    img = imread('static/temp/'+os.listdir('static/temp/')[-1])
    img = np.array(Image.fromarray(img).resize(size=(224,224)))
    images = test_transform(img)
    images = torch.reshape(images, (1, images.shape[0], images.shape[1], images.shape[2]))

    captions = test_sort['caption_id']
    caption_lengths = torch.tensor([len(c) for c in captions])
    captions = torch.tensor([c + [0]*(100-len(c)) for c in captions])

    with torch.no_grad():
        image_embeddings, text_embeddings = network(images, captions, caption_lengths)

    sim = torch.matmul(text_embeddings, image_embeddings.t())
    ind = sim.topk(20, 0)[1].reshape(-1)

    retrieved_captions = ['temp/'+os.listdir('static/temp/')[-1]]
    for i in ind:
        retrieved_captions.append(index_to_word(test_sort['caption_id'][i]))

    return retrieved_captions

def retrieve_images(caption):
    exclude = set(string.punctuation)
    cap = ''.join(c for c in caption if c not in exclude)
    tokens = cap.split()
    tokens = ['<start>'] + tokens + ['<end>']
    indexed_caption = []

    for token in tokens:
        if token.lower() in word_to_index:
            indexed_caption.append(word_to_index[token.lower()])
        else:
            indexed_caption.append(0)
    
    caption_lengths = torch.tensor([len(indexed_caption)])
    indexed_caption += [0] * (100 - len(indexed_caption))
    captions = torch.tensor([indexed_caption])

    paths = test_sort['images_path']
    images = []
    for img_path in paths:
        img = imread('static/'+img_path)
        img = np.array(Image.fromarray(img).resize(size=(224,224)))
        images.append(test_transform(img))
    images = torch.stack(images)

    with torch.no_grad():
        image_embeddings, text_embeddings = network(images, captions, caption_lengths)

    sim = torch.matmul(image_embeddings, text_embeddings.t())
    ind = sim.topk(20, 0)[1].reshape(-1)

    retrieved_images = [caption]
    for i in ind:
        retrieved_images.append(paths[i])

    return retrieved_images

def get_query(request):
    try:
        text = request.form['textquery']
    except:
        text = None
    
    try:
        image = request.files['imagequery']
        for fname in os.listdir('static/temp'):
            os.remove('static/temp/'+fname)
        image.save('static/temp/'+str(int(time.time()))+'.jpg')
    except:
        image = None

    if text is None:
        return (image, 'image')
    else:
        return (text, 'text')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict_from_location():
    query, query_type = get_query(request)
    if query_type == 'text':
        retrieved_images = retrieve_images(query)
        return render_template('image_results.html', data=retrieved_images)
    else:
        retrieved_captions = retrieve_captions()
        return render_template('text_results.html', data=retrieved_captions)

if __name__ == '__main__':
    print('Parsing arguments...')
    args = config()
    print('Loading model weights...')
    network, test_sort, word_to_index = load_model(args)
    app.run()