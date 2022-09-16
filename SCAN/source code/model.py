# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on 
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""SCAN model"""

import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def EncoderImage(data_name, img_dim, embed_size, precomp_enc_type='basic', 
                 no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if precomp_enc_type == 'basic':
        img_enc = EncoderImagePrecomp(
            img_dim, embed_size, no_imgnorm)
    elif precomp_enc_type == 'weight_norm':
        img_enc = EncoderImageWeightNormPrecomp(
            img_dim, embed_size, no_imgnorm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))
    #print("The shape of img_enc:", img_enc.shape)
    return img_enc


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self): #initialize weight and bias for the FC layer
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features) #seems that lots of weights are initialized in this way
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


class EncoderImageWeightNormPrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImageWeightNormPrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = weight_norm(nn.Linear(img_dim, embed_size), dim=None)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImageWeightNormPrecomp, self).load_state_dict(new_state)


# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_bi_gru=False, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        if self.use_bi_gru:
            #print("cap_emb.size(2):",cap_emb.size(2))
            #print("cap_emb.size(2) type:",type(cap_emb.size(2)//2))
            #print("cap_emb,",cap_emb)
            cap_emb_size2 = cap_emb.size(2)//2
            cap_emb = (cap_emb[:,:,:cap_emb_size2] + cap_emb[:,:,cap_emb_size2:])/2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)
        #print("the shape of cap_emb,",cap_emb.shape) #([128, 58, 1024])
        
        return cap_emb, cap_len

### Change the attention function to class

'''class EncoderSimilarityi2t(nn.Module):
    """Attention nn module that is responsible for computing the alignment scores."""

    def __init__(self, d, opt):
        super(EncoderSimilarityi2t, self).__init__()
        self.d =d #torch.tensor(d).to(torch.device('cuda'))
        self.opt = opt
        self.smooth = opt.lambda_softmax
        # Define layers
        if self.opt.attention_method == 'general':
            self.attention = nn.Linear(self.d, self.d)
        elif self.opt.attention_method == 'concat':
            self.attention = nn.Linear(self.d * 2, self.d)
            self.other = nn.Parameter(torch.FloatTensor(1, self.d))
            

    def forward(self, images, captions, cap_lens):
        """
        query: (n_context, queryL, d)
        context: (n_context, sourceL, d)
        word(query): (n_image, n_word, d)
        image(context): (n_image, n_region, d)
        weiContext: (n_image, n_region, d)
        attn: (n_image, n_word, n_region)
        """
        smooth = self.opt.lambda_softmax
        eps=1e-8
        similarities = []
        n_image = images.size(0)
        n_caption = captions.size(0)
        n_region = images.size(1)
        for i in range(n_caption):
            # Get the i-th text description
            n_word = cap_lens[i]
            cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
            # (n_image, n_word, d)
            context = cap_i.repeat(n_image, 1, 1)
    
            query = images

            # func_attention
            batch_size_q, queryL = query.size(0), query.size(1)  # i2t: query:(n_image, n_word, d)
            batch_size, sourceL = context.size(0), context.size(1) #i2t: context(n_image, n_regions, d)
            queryT = torch.transpose(query, 1, 2)
             
            if self.opt.attention_method=='dot':
                attn = torch.bmm(context, queryT)  
            # (batch, sourceL, d)(batch, d, queryL) == i2t (n_image, n_regions, d)(n_image, d, n_word)
            # --> (batch, sourceL, queryL)          == i2t (n_image, n_regions, n_word) = (128, n_regions, 36)
        
            elif opt.attention_method == 'general':

                energy = self.attention(queryT)
                attn = torch.bmm(context,energy)
            elif self.opt.attention_method == 'concat':
                energy = self.attention(torch.cat((context, queryT), 1))
                attn = torch.bmm(self.other,energy)
            
            # normalize s_ij =>s_ij'
            if self.opt.raw_feature_norm == "softmax":
                # --> (batch*sourceL, queryL)
                attn = attn.view(batch_size*sourceL, queryL)
                attn = nn.Softmax()(attn)
                # --> (batch, sourceL, queryL)
                attn = attn.view(batch_size, sourceL, queryL)
            elif self.opt.raw_feature_norm == "l2norm":
                attn = l2norm(attn, 2)
            elif self.opt.raw_feature_norm == "clipped_l2norm":
                attn = nn.LeakyReLU(0.1)(attn)
                attn = l2norm(attn, 2)
            elif self.opt.raw_feature_norm == "l1norm":
                attn = l1norm_d(attn, 2)
            elif self.opt.raw_feature_norm == "clipped_l1norm":
                attn = nn.LeakyReLU(0.1)(attn)
                attn = l1norm_d(attn, 2)
            elif self.opt.raw_feature_norm == "clipped":
                attn = nn.LeakyReLU(0.1)(attn)
            elif self.opt.raw_feature_norm == "no_norm":
                pass
            else:
                raise ValueError("unknown first norm type:", self.opt.raw_feature_norm)

                # --> (batch, queryL, sourceL)
            attn = torch.transpose(attn, 1, 2).contiguous() # i2t (n_image, n_word, n_regions)
                # --> (batch*queryL, sourceL)
            attn = attn.view(batch_size*queryL, sourceL)

            # Calculate alpha_ij
            attn = nn.Softmax()(attn*smooth)
                # --> (batch, queryL, sourceL)
            attn = attn.view(batch_size, queryL, sourceL)
                # --> (batch, sourceL, queryL) alpha_ij
            attnT = torch.transpose(attn, 1, 2).contiguous()

            #Calculate a_j^t: weighted combination of word representation
                # --> (batch, d, sourceL) 
            contextT = torch.transpose(context, 1, 2)           #t21:v_j
                # (batch x d x sourceL)(batch x sourceL x queryL)
                # --> (batch, d, queryL)
            weightedContext = torch.bmm(contextT, attnT)
                # --> (batch, queryL, d)
            weightedContext = torch.transpose(weightedContext, 1, 2)
            # (n_image, n_region)
            print('weightedContext',weightedContext.shape)
            print('images',images.shape)

            row_sim = cosine_similarity(images, weightedContext, dim=2)
            print('row_sim',row_sim.shape)
            if opt.agg_func == 'LogSumExp':
                row_sim.mul_(opt.lambda_lse).exp_()
                row_sim = row_sim.sum(dim=1, keepdim=True)
                row_sim = torch.log(row_sim)/opt.lambda_lse
            elif opt.agg_func == 'Max':
                row_sim = row_sim.max(dim=1, keepdim=True)[0]
            elif opt.agg_func == 'Sum':
                row_sim = row_sim.sum(dim=1, keepdim=True)
            elif opt.agg_func == 'Mean':
                row_sim = row_sim.mean(dim=1, keepdim=True)
            else:
                raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
            similarities.append(row_sim)


        # (n_image, n_caption)
        similarities = torch.cat(similarities, 1)
        #print("the shape of similarities:",similarities.shape)
        return similarities'''
        

class EncoderSimilarityt2i(nn.Module):
    """Attention nn module that is responsible for computing the alignment scores."""

    def __init__(self, d, opt):
        super(EncoderSimilarityt2i, self).__init__()
        self.d =d #torch.tensor(d).to(torch.device('cuda'))
        self.d2 =int(d/4)
        self.opt = opt
        self.smooth = opt.lambda_softmax
        self.batch_size = opt.batch_size
        self.embed_size = opt.embed_size

        # Define layers
        self.attention_general = nn.Linear(self.d, self.d,bias=False)
            #self.W_matrix = nn.Parameter(torch.FloatTensor(self.d, self.d))
            #b = torch.randn(self.batch_size,self.d,self.d).to('cuda')
            #self.W_matrix = torch.tensor(b,requires_grad=True)

        self.attention_biased_general = nn.Linear(self.d, self.d)

        
        self.attention_activated_general = nn.Linear(self.d, self.d)
        #self.attention = nn.Parameter(torch.FloatTensor(self.d, self.d))
        self.b_activated_general = nn.Parameter(torch.FloatTensor(36, 1))
        

            

    def forward(self, images, captions, cap_lens):
        """
        query: (n_context, queryL, d)
        context: (n_context, sourceL, d)
        captions: (n_image, n_word, d)
        images: (n_image, n_region, d)
        weiContext: (n_image, n_region, d)
        attn: (n_image, n_word, n_region)
        """
        similarities = []
        smooth = self.opt.lambda_softmax
        n_image = images.size(0) #number of images
        n_caption = captions.size(0) #numer of captions
        eps=1e-8
        for i in range(n_caption):
            # Get the i-th text description => cap_i
            n_word = cap_lens[i] #the length of i-th sentence
            cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous() ## the i-th sentence representation(1,n_word,d)
                                                                        # captions[i, :n_word, :]                   =>(n_word,d)
                                                                    #captions[i, :n_word, :].unsqueeze(0)       =>(1,n_word,d)
            query = cap_i.repeat(n_image, 1, 1) # --> (n_image, n_word, d) 
            cap_i_expand = cap_i.repeat(n_image, 1, 1) 
            context = images
            context = context.to(torch.float32)
            query =query.to(torch.float32)

            # func_attention
            batch_size_q, queryL = query.size(0), query.size(1)  # i2t: query:(n_image, n_word, d)
            batch_size, sourceL = context.size(0), context.size(1) #i2t: context(n_image, n_regions, d)
            queryT = torch.transpose(query, 1, 2)
            contextT = torch.transpose(context, 1, 2)

            if self.opt.attention_method=='dot':
                attn = torch.bmm(context, queryT)  
            # (batch, sourceL, d)(batch, d, queryL) == i2t (n_image, n_regions, d)(n_image, d, n_word)
            # --> (batch, sourceL, queryL)          == i2t (n_image, n_regions, n_word) = (128, n_regions, 36)

            elif self.opt.attention_method=='scaled_dot':
                attn = torch.bmm(context, queryT)/32

            elif self.opt.attention_method=='cosine':
                context_norm = torch.linalg.norm(context,dim=2)
                context_norm = context_norm.unsqueeze(2).repeat(1,1,queryL)
                #print(context_norm.shape)
                query_norm = torch.linalg.norm(queryT,dim=1)
                query_norm = query_norm.unsqueeze(1).repeat(1,sourceL,1)
                attn = torch.bmm(context, queryT)
                attn = attn/(context_norm*query_norm)
                
            elif self.opt.attention_method == 'general_qwk':
              #qwk
                energy = self.attention_general(context.to(torch.float32)) #(128,l,1024)
                energyT = torch.transpose(energy, 1, 2)
                attn = torch.bmm(query,energyT)

                # using nn.Parameter=>out of memory
                #energy = torch.bmm(self.W_matrix,context)
                #attn = torch.bmm(query,energy)

                # using nn.Tensor
                #energy = torch.bmm(self.W_matrix,context)
                #attn = torch.bmm(query,energy)

            elif self.opt.attention_method == 'general_kwq':
                # KWQ
                energy = self.attention_general(query.to(torch.float32)) #(128,l,1024)
                energyT = torch.transpose(energy, 1, 2)
                attn = torch.bmm(context,energyT)
                
            elif self.opt.attention_method == 'biased_general_qwk':
                energy = self.attention_biased_general(context.to(torch.float32)) #(128,l,1024)
                energyT = torch.transpose(energy, 1, 2)
                attn = torch.bmm(query.to(torch.float32),energyT.to(torch.float32))
                #energy = self.attention(query) #(128,l,1024)
                #a#ttn = torch.bmm(energy,contextT)
            
            elif self.opt.attention_method == 'biased_general_kwq':
                energy = self.attention_biased_general(query) #(128,l,1024)
                attn = torch.bmm(energy,contextT)
                
            elif self.opt.attention_method == 'activated_general_tanh':
                energy = self.attention_activated_general(context.to(torch.float32)) #(128,l,1024)
                energyT = torch.transpose(energy, 1, 2)
                attn = torch.bmm(query.to(torch.float32),energyT.to(torch.float32))
                attn = nn.functional.tanh(attn)

            elif self.opt.attention_method == 'activated_general_relu':
                energy = self.attention_activated_general(context.to(torch.float32)) #(128,l,1024)
                energyT = torch.transpose(energy, 1, 2)
                attn = torch.bmm(query.to(torch.float32),energyT.to(torch.float32))
                attn = nn.functional.relu(attn)
            elif self.opt.attention_method == 'activated_general_leakyrelu':
                energy = self.attention_activated_general(context.to(torch.float32)) #(128,l,1024)
                energyT = torch.transpose(energy, 1, 2)
                attn = torch.bmm(query.to(torch.float32),energyT.to(torch.float32))
                attn = nn.functional.leaky_relu(attn)
            '''
            elif self.opt.attention_method == 'concat':
               attn = torch.zeros(batch_size, sourceL,queryL)#.to(torch.device('cuda'))
               for li in range(queryL):
                   scores = self.attention(torch.cat((context, query[:,li,:].unsqueeze(1).repeat(1,sourceL,1)), 2))
  
                   #nn.linear
                   attn[:,:,li] = self.W(scores).squeeze() 

                   #using nn.parameter
                   #scores=torch.transpose(scores, 1, 2)
                   #attn[:,:,li] = torch.bmm(self.W_matrix,scores).squeeze(1)  
                   

           elif self.opt.attention_method == 'additive':
               attn = torch.zeros(batch_size, sourceL,queryL)#.to(torch.device('cuda'))
               
               for li in range(queryL):
                 wk = self.w1(context.to(torch.float32))
                 #wq = self.w2(query[:,li,:].unsqueeze(1).repeat(1,sourceL,1).to(torch.float32))
                 wq = self.w2(query[:,li,:].unsqueeze(1)).repeat(1,sourceL,1)
                 energy = torch.nn.functional.tanh(torch.add(wk,wq))

                 # nn.linear
                 #attn[:,:,li] = self.W(energy).squeeze() 
       
                 #nn.parameter
                 energyT = torch.transpose(energy, 1, 2)
                 attn[:,:,li] = torch.matmul(self.W_matrix,energyT).squeeze()  
            '''                       
            #print(attn.shape)    
            #else:
            #  print("please assign the right attention method")
            
            # normalize s_ij =>s_ij'
            if self.opt.raw_feature_norm == "softmax":
                # --> (batch*sourceL, queryL)
                attn = attn.view(batch_size*sourceL, queryL)
                attn = nn.Softmax()(attn)
                # --> (batch, sourceL, queryL)
                attn = attn.view(batch_size, sourceL, queryL)
            elif self.opt.raw_feature_norm == "l2norm":
                attn = l2norm(attn, 2)
            elif self.opt.raw_feature_norm == "clipped_l2norm":
                attn = nn.LeakyReLU(0.1)(attn.to(torch.device('cuda')))
                attn = l2norm(attn, 2)
            #elif self.opt.raw_feature_norm == "l1norm":
            #    attn = l1norm_d(attn, 2)
            #elif self.opt.raw_feature_norm == "clipped_l1norm":
            #    attn = nn.LeakyReLU(0.1)(attn)
            #    attn = l1norm_d(attn, 2)
            elif self.opt.raw_feature_norm == "clipped":
                attn = nn.LeakyReLU(0.1)(attn)
            elif self.opt.raw_feature_norm == "no_norm":
                pass
            else:
                raise ValueError("unknown first norm type:", self.opt.raw_feature_norm)

                # --> (batch, queryL, sourceL)
            attn = torch.transpose(attn, 1, 2).contiguous() # i2t (n_image, n_word, n_regions)
                # --> (batch*queryL, sourceL)
            attn = attn.view(batch_size*queryL, sourceL)

            # Calculate alpha_ij
            attn = nn.Softmax()(attn*smooth)
                # --> (batch, queryL, sourceL)
            attn = attn.view(batch_size, queryL, sourceL)
                # --> (batch, sourceL, queryL) alpha_ij
            attnT = torch.transpose(attn, 1, 2).contiguous()

            #Calculate a_j^t: weighted combination of word representation
                # --> (batch, d, sourceL) 
            contextT = torch.transpose(context, 1, 2)           #t21:v_j
                # (batch x d x sourceL)(batch x sourceL x queryL)
                # --> (batch, d, queryL)
            weightedContext = torch.bmm(contextT.to(torch.float32), attnT.to(torch.float32))
                # --> (batch, queryL, d)
            weightedContext = torch.transpose(weightedContext, 1, 2)
            # (n_image, n_region)
            
            cap_i_expand = cap_i_expand.contiguous()
            weightedContext = weightedContext.contiguous()
            row_sim = cosine_similarity(cap_i_expand, weightedContext, dim=2)

            if self.opt.agg_func == 'LogSumExp':
                row_sim.mul_(self.opt.lambda_lse).exp_()
                row_sim = row_sim.sum(dim=1, keepdim=True)
                row_sim = torch.log(row_sim)/self.opt.lambda_lse
            elif self.opt.agg_func == 'Max':
                row_sim = row_sim.max(dim=1, keepdim=True)[0]
            elif self.opt.agg_func == 'Sum':
                row_sim = row_sim.sum(dim=1, keepdim=True)
            elif self.opt.agg_func == 'Mean':
                row_sim = row_sim.mean(dim=1, keepdim=True)
            else:
                raise ValueError("unknown aggfunc: {}".format(self.opt.agg_func))
            similarities.append(row_sim)

        # (n_image, n_caption)
        similarities = torch.cat(similarities, 1)
        #print("the shape of similarities:",similarities.shape)
        return similarities
        




'''def func_attention(query, context, opt, smooth, eps=1e-8,attn_model='general'):
    """
    in paper stage 1,2: 
        input sentence+image representation
        output S final similarity score
    query: (n_context, queryL, d)               #queryL: lenth of the sentence
    context: (n_context, sourceL, d)            #sourceL: number of regions for one image
    remember:
        i2t: weiContext, attn = func_attention(cap_i_expand, images, opt, smooth=opt.lambda_softmax)
    """
    batch_size_q, queryL = query.size(0), query.size(1)  # i2t: query:(n_image, n_word, d)
    batch_size, sourceL = context.size(0), context.size(1) #i2t: context(n_image, n_regions, d)
    #batch_size_q == batch_size


    # Get attention
            # --> (batch, d, queryL)                    # == (n_image, d, n_word)

    queryT = torch.transpose(query, 1, 2)
    

    # calcualte s_ij 

    # Original (cosine similarity for all pairs)=> attn
    #attn = torch.bmm(context, queryT)  
        # (batch, sourceL, d)(batch, d, queryL) == i2t (n_image, n_regions, d)(n_image, d, n_word)
        # --> (batch, sourceL, queryL)          == i2t (n_image, n_regions, n_word) = (128, n_regions, 36)
    
    # my try
    attention = Attention(attn_model, queryT.size(1))
    attn = attention(context, queryT)
    print('atten.shape',attn.shape)

    
    # normalize s_ij =>s_ij'
    if opt.raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size*sourceL, queryL)
        attn = nn.Softmax()(attn)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif opt.raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "l1norm":
        attn = l1norm_d(attn, 2)
    elif opt.raw_feature_norm == "clipped_l1norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l1norm_d(attn, 2)
    elif opt.raw_feature_norm == "clipped":
        attn = nn.LeakyReLU(0.1)(attn)
    elif opt.raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", opt.raw_feature_norm)

        # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous() # i2t (n_image, n_word, n_regions)
        # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)

    # Calculate alpha_ij
    attn = nn.Softmax()(attn*smooth)
        # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
        # --> (batch, sourceL, queryL) alpha_ij
    attnT = torch.transpose(attn, 1, 2).contiguous()

    #Calculate a_j^t: weighted combination of word representation
        # --> (batch, d, sourceL) 
    contextT = torch.transpose(context, 1, 2)           #t21:v_j
        # (batch x d x sourceL)(batch x sourceL x queryL)
        # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
        # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT # i2t: a_j^t, alpha_ij
                                  # i2t: a_i^t, alpha_ij
'''

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


# not change yet
def xattn_score_t2i(images, captions, cap_lens, opt):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    #rememd: scores = xattn_score_t2i(im, s, s_l, self.opt)
    similarities = []
    n_image = images.size(0) #number of images
    n_caption = captions.size(0) #numer of captions
    for i in range(n_caption):
        # Get the i-th text description => cap_i
        n_word = cap_lens[i] #the length of i-th sentence
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous() ## the i-th sentence representation(1,n_word,d)
                                                                    # captions[i, :n_word, :]                   =>(n_word,d)
                                                                    #captions[i, :n_word, :].unsqueeze(0)       =>(1,n_word,d)
        
        #cap_i_expand is query => e in paper
        cap_i_expand = cap_i.repeat(n_image, 1, 1) # --> (n_image, n_word, d) 
                                                    #相当于多重复几次句子，让这个句子和每个image匹配比较
        """
            word(query): (n_image, n_word, d) => cap_i_expand is query; n_image: number of context （only 1 sentence expand for all images）
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """
        
        #weiContext, attn = EncoderSimilarity(cap_i_expand, images, opt, smooth=opt.lambda_softmax)
        weiContext, attn = func_attention(cap_i_expand, images, opt, smooth=opt.lambda_softmax)
                # in paper stage 1
                #  weiContext: a_j^t
                #  attn: alpha_ij                                                                                
        
        #### stage 2 in paper #####
        cap_i_expand = cap_i_expand.contiguous()
        weiContext = weiContext.contiguous()
        # R(e,a) -->(n_image, n_word)
        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2) #stage 2 attend to the important iamge region given weighted context
        # pooling     
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)/opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1) # torch.Size([128, 128])
    #print("similarities in xattn_score_t2i:",similarities.shape)
    return similarities


def xattn_score_i2t(images, captions, cap_lens, opt):
    """
    Images: (batch_size, n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    n_region = images.size(1)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_region, d)
            weiContext: (n_image, n_region, d)
            attn: (n_image, n_word, n_region)
        """
        #weiContext, attn = EncoderSimilarity(images, cap_i_expand, opt, smooth=opt.lambda_softmax)
        weiContext, attn = func_attention(images, cap_i_expand, opt, smooth=opt.lambda_softmax)
        # (n_image, n_region)
        row_sim = cosine_similarity(images, weiContext, dim=2)
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)/opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    #print("the shape of similarities:",similarities.shape)
    return similarities


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, scores,img_emb):

        """
        # compute image-sentence score matrix
        if self.opt.cross_attn == 't2i':
            scores = xattn_score_t2i(im, s, s_l, self.opt) #similarity scores, the output of stage 2
            # shape([128, 128])
        elif self.opt.cross_attn == 'i2t':
            scores = xattn_score_i2t(im, s, s_l, self.opt)
        else:
            raise ValueError("unknown first norm type:", opt.raw_feature_norm)

        """
        diagonal = scores.diag().view(img_emb.size(0), 1)
        #print("diagonal:",diagonal)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
            #Pairwise Ranking Loss forces representations to have 0 distance for positive pairs, and a distance greater than a margin for negative pairs.
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()


class SCAN(object):
    """
    Stacked Cross Attention Network (SCAN) model
    """
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    precomp_enc_type=opt.precomp_enc_type,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers, 
                                   use_bi_gru=opt.bi_gru,  
                                   no_txtnorm=opt.no_txtnorm)
        self.sim_enc = EncoderSimilarityt2i(opt.embed_size,opt)
        #def __init__(self, d, query, context, opt , smooth,method='dot'):#images, captions, cap_lens, opt
    
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.sim_enc.cuda()
            cudnn.benchmark = True ##加速卷积

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())
        params += list(self.sim_enc.parameters())

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(),self.sim_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        self.sim_enc.load_state_dict(state_dict[2])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()
        self.sim_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()
        self.sim_enc.eval()

    def forward_sim(self, img_embs, cap_embs, cap_lens):
        # Forward similarity encoding
        sims = self.sim_enc(img_embs, cap_embs, cap_lens)
        return sims

    def forward_emb(self, images, captions, lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images, volatile=volatile)
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward
        img_emb = self.img_enc(images)

        # cap_emb (tensor), cap_lens (list)
        cap_emb, cap_lens = self.txt_enc(captions, lengths)
        return img_emb, cap_emb, cap_lens

    def forward_loss(self, sims,img_emb, **kwargs):
        """Compute the ContrastiveLoss loss given pairs of image and caption embeddings
        """
        #print("img_emb shape:", img_emb.shape)
        #print("cap_emb shape:",cap_emb.shape)
        #print("cap_len",len(cap_len))  #128
        loss = self.criterion(sims,img_emb)
        #print("loss.data[0]:",loss.item())
        #print("img_emb shape:", img_emb.shape)
        self.logger.update('Le', loss.item(), img_emb.size(0))

        return loss

    def train_emb(self, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb, cap_lens = self.forward_emb(images, captions, lengths)
        sims = self.forward_sim(img_emb, cap_emb, cap_lens)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(sims,img_emb)

        # compute gradient and do SGD step
        loss.backward()

        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()
