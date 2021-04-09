import torch.nn as nn
from .bi_lstm import BiLSTM
from .mobilenet import MobileNetV1
from .resnet import resnet50
import torch


# class Image_Encoder(nn.Module):

#     def __init__(self, common_dim = 100, input_dim=1024):
        
#         super(Image_Encoder, self).__init__()
        
#         self.dims = [input_dim, common_dim]
#         self.layers = nn.ModuleList()
#         self.num_layers = len(self.dims) - 1
#         for i in range(self.num_layers):
#             self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))

#     def forward(self, inp):

#         x = inp
#         for i in range(self.num_layers):
#             x =F.relu(self.layers[i](x))

#         return F.relu(x)
        


# class Text_Encoder(nn.Module):
    
#     def __init__(self, common_dim = 100, input_dim = 1024):
        
#         super(Text_Encoder, self).__init__()
        
#         self.dims = [input_dim, common_dim]
#         self.layers = nn.ModuleList()
#         self.num_layers = len(self.dims) - 1
#         for i in range(self.num_layers):
#             self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))

#     def forward(self, inp):

#         x = inp
#         for i in range(self.num_layers):
#             x =F.relu(self.layers[i](x))

#         return F.relu(x)



# class Image_Decoder(nn.Module):

#     def __init__(self, common_dim = 100, output_dim = 1024):
        
#         super(Image_Decoder, self).__init__()
        
#         self.dims = [common_dim, output_dim]
#         self.layers = nn.ModuleList()
#         self.num_layers = len(self.dims) - 1
#         for i in range(self.num_layers):
#             self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))

#     def forward(self, inp):

#         x = inp
#         for i in range(self.num_layers):
#             x =F.relu(self.layers[i](x))

#         return F.relu(x)

       
# class Text_Decoder(nn.Module):

#     def __init__(self, common_dim = 100, output_dim = 1024):
        
#         super(Text_Decoder, self).__init__()
        
#         self.dims = [common_dim, output_dim]
#         self.layers = nn.ModuleList()
#         self.num_layers = len(self.dims) - 1
#         for i in range(self.num_layers):
#             self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))

#     def forward(self, inp):

#         x = inp
#         for i in range(self.num_layers):
#             x =F.relu(self.layers[i](x))

#         return F.relu(x)


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

        # self.image_encode = Image_Encoder(common_dim = 100, input_dim = 1024)
        # self.text_encode = Text_Encoder(common_dim = 100, input_dim = 1024)
        
        # self.image_decode = Image_Decoder(common_dim = 100, output_dim = 1024)
        # self.text_decode = Text_Decoder(common_dim = 100, output_dim = 1024)



    def forward(self, images, text, text_length, is_image_zero=False, is_text_zero=False):

        # images.shape -> 16*3*224*224
        # text.shape -> 16*100

        # print("inp_images: ",images.shape)
        # print("inp_txt: ",text.shape)
        
        image_features = self.image_model(images)
        if is_image_zero:
            image_features = torch.zeros_like(image_features)
             
        text_features = self.bilstm(text, text_length) 
        if is_text_zero:
            text_features = torch.zeros_like(text_features)
        
        # print("img_out: ",image_features.shape)
        # print("txt_out: ",text_features.shape)

        image_features = image_features.squeeze()
        text_features = text_features.squeeze()


        
        # Here we create pass the text and image through the respective encoders
        image_embeddings = self.image_encode(image_features) #16 * 100
        text_embeddings = self.text_encode(text_features)

        # image_embeddings, text_embeddings= self.build_joint_embeddings(image_features, text_features)

        common_rep = torch.add(image, text_embeddings) #16 * 100
        
        image_decoded = self.image_decode(common_rep) #16 * 1024
        text_decoded = self.text_decode(common_rep)
        
        # print("img_ret: ",image_embeddings.shape)
        # print("txt_ret: ",text_embeddings.shape)

        z = torch.cat(image_features, text_features)
        z_dash = torch.cat(image_decoded, text_decoded)

        return z, common_rep, z_dash
        # return image_features, text_features, common_rep, image_decoded, text_decoded

        return image_embeddings,text_embeddings


    def build_joint_embeddings(self, images_features, text_features):
        
        #images_features = images_features.permute(0,2,3,1)
        #text_features = text_features.permute(0,3,1,2)
        image_embeddings = self.conv_images(images_features).squeeze()
        text_embeddings = self.conv_text(text_features).squeeze()

        return image_embeddings, text_embeddings
        




