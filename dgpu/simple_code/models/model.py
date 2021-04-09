import torch.nn as nn
from .bi_lstm import BiLSTM
from .mobilenet import MobileNetV1
from .resnet import resnet50


class Image_Encoder(nn.Module):

    def __init__(self, image_dim):
        
        super(Image_Encoder, self).__init__()
        
        self.img_dims = [input_vec_dim, 300, 200, 50]
        self.fc1_img = nn.Linear(self.img_dims[0], self.img_dims[1])
        self.fc2_img = nn.Linear(self.img_dims[1], self.img_dims[2])
        self.fc3_img = nn.Linear(self.img_dims[2], self.img_dims[3])


    def forward(self, img):

        x = F.relu(self.fc1_img(img))
        x = F.relu(self.fc2_img(x))
        x = F.relu(self.fc3_img(x))

        return F.relu(x)
        

class Text_Encoder(nn.Module):

    def __init__(self, text_dim):
        
        super(Text_Encoder, self).__init__()
        
        self.text_dims = [text_dim, 300, 200, 50]
        self.fc1_txt = nn.Linear(self.text_dims[0], self.text_dims[1])
        self.fc2_txt = nn.Linear(self.text_dims[1], self.text_dims[2])
        self.fc3_txt = nn.Linear(self.text_dims[2], self.text_dims[3])


    def forward(self, txt):

        y = F.relu(self.fc1_txt(txt))
        y = F.relu(self.fc2_txt(y))
        y = F.relu(self.fc3_txt(y))

        return F.relu(y)


class Decoder(nn.Module):

    def __init__(self,output_vec_dim):
        
        super(Decoder,self).__init__()
        
        self.img_dims=[output_vec_dim,200,300,512]
        self.fc1_img=nn.Linear(self.img_dims[0],self.img_dims[1])
        self.fc2_img=nn.Linear(self.img_dims[1],self.img_dims[2])
        self.fc3_img=nn.Linear(self.img_dims[2],self.img_dims[3])
        
        self.txt_dims=[output_vec_dim,200,300,512]
        self.fc1_txt=nn.Linear(self.txt_dims[0],self.txt_dims[1])
        self.fc2_txt=nn.Linear(self.txt_dims[1],self.txt_dims[2])
        self.fc3_txt=nn.Linear(self.txt_dims[2],self.txt_dims[3])

    def forward(self,rep):

        x=F.relu(self.fc1_img(rep))
        x=F.relu(self.fc2_img(x))
        x=F.relu(self.fc3_img(x))

        y=F.relu(self.fc1_txt(rep))
        y=F.relu(self.fc2_txt(y))
        y=F.relu(self.fc3_txt(y))

        combined=F.relu(torch.cat((x,y),1))
        return combined


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        if args.image_model == 'mobilenet_v1':
            self.image_model = MobileNetV1()
            self.image_model.apply(self.image_model.weight_init)
        elif args.image_model == 'resnet50':
            self.image_model = resnet50()
        elif args.image_model == 'resent101':
            self.image_model = resnet101()

        # self.bilstm = BiLSTM(args.num_lstm_units, num_stacked_layers = 1, vocab_size = args.vocab_size, embedding_dim = 512)
        self.bilstm = BiLSTM(args)
        self.bilstm.apply(self.bilstm.weight_init)

        inp_size = 1024
        if args.image_model == 'resnet50' or args.image_model == 'resnet101':
            inp_size = 2048
        # shorten the tensor using 1*1 conv
        self.conv_images = nn.Conv2d(inp_size, args.feature_size, 1)
        self.conv_text = nn.Conv2d(1024, args.feature_size, 1)

        


    def forward(self, images, text, text_length):

        image_features = self.image_model(images) 
        text_features = self.bilstm(text, text_length) 
        
        print(image_features.shape)
        print(text_features.shape)
        
        # Here we create pass the text and image through the respective encoders
        # image_embeddings = self.image_encode(image_features)
        # text_embeddings = self.text_encode(text_features)

        image_embeddings, text_embeddings= self.build_joint_embeddings(image_features, text_features)

        return image_embeddings, text_embeddings


    def build_joint_embeddings(self, images_features, text_features):
        
        #images_features = images_features.permute(0,2,3,1)
        #text_features = text_features.permute(0,3,1,2)
        image_embeddings = self.conv_images(images_features).squeeze()
        text_embeddings = self.conv_text(text_features).squeeze()

        return image_embeddings, text_embeddings


    # def image_encode(self, image_features):
        # encoder_obj = 