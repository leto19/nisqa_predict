import torch
import torch.nn.functional as F
from torch import Tensor, nn
try:
    from wavLM_wrapper import WavLMfeatureExtractor,WavLMFull,WavLMFull_all
    from transformer_config import CenterCrop,Config,Input
    from transformer_wrapper import TransformerWrapper
except:
    from models.wavLM_wrapper import WavLMfeatureExtractor,WavLMFull,WavLMFull_all
    from models.transformer_config import CenterCrop,Config,Input
    from models.transformer_wrapper import TransformerWrapper

from speechbrain.processing.features import spectral_magnitude,STFT

class PoolAttFF(torch.nn.Module):
    '''
    PoolAttFF: Attention-Pooling module with additonal feed-forward network.
    '''         
    def __init__(self, dim_head_in):
        super().__init__()
        
        self.linear1 = nn.Linear(dim_head_in, 2*dim_head_in)
        self.linear2 = nn.Linear(2*dim_head_in, 1)
        
        self.linear3 = nn.Linear(dim_head_in, 1)
        
        self.activation = F.relu
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: Tensor):

        att = self.linear2(self.dropout(self.activation(self.linear1(x))))
        att = att.transpose(2,1)
        att = F.softmax(att, dim=2)
        x = torch.bmm(att, x) 
        x = x.squeeze(1)
        x = self.linear3(x)
        return x  


class wavLMMetricPredictorEncoderTransformerSmall(nn.Module):
    def __init__(self, feat_seq=384):
        super().__init__()

        self.norm_input = nn.BatchNorm1d(512)

        self.cc = CenterCrop(feat_seq)

        self.feat_extract = WavLMfeatureExtractor()
        self.feat_extract.requires_grad_(False)

        self.config = Config(
            "HUBERT_ENCODER_CONFIG",
            Input.XLSR,
            feat_seq_len=feat_seq,
            dim_transformer=256,
            xlsr_name="hubert_encoder",
            nhead_transformer=4,
            nlayers_transformer=4,
        )
        
        self.transformer = TransformerWrapper(self.config)
        
        
        self.attenPool = PoolAttFF(self.config.dim_transformer)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #input is (batch_size,time)        
        out_feats = self.feat_extract(x) #wavLM returns (batch_size,time,512)
        out_feats = self.cc(out_feats) #pad / crop the time dimension to 256 (batch_size,256,512)
        out = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize the input (batch_size,256,512)
        out = self.transformer(out_feats) #transformer returns (batch_size,256,512)
        out = self.attenPool(out) #attenPool returns (batch_size,1)
        out = self.sigmoid(out) #sigmoid returns (batch_size,1)
        return out

class wavLMMetricPredictorEncoderTransformerSmallT(nn.Module):
    def __init__(self, feat_seq=384):
        super().__init__()

        self.norm_input = nn.BatchNorm1d(feat_seq)

        self.cc = CenterCrop(feat_seq)

        self.feat_extract = WavLMfeatureExtractor()
        self.feat_extract.requires_grad_(False)

        self.config = Config(
            "HUBERT_ENCODER_CONFIG",
            Input.XLSR,
            feat_seq_len=512,
            dim_transformer=256,
            xlsr_name="hubert_encoder_t",
            nhead_transformer=4,
            nlayers_transformer=4,
        )
        self.transformer = TransformerWrapper(self.config)
        self.attenPool = PoolAttFF(self.config.dim_transformer)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #input is (batch_size,time)        
        out_feats = self.feat_extract(x) #wavLM returns (batch_size,time,512)
        out_feats = self.cc(out_feats) #pad / crop the time dimension to 256 (batch_size,256,512)


        out_feats = out_feats.permute(0,2,1) #swap the time and feature dimension (batch_size,512,256)

        out = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize the input (batch_size,256,512)

        print(out.shape)
        out = self.transformer(out_feats) #transformer returns (batch_size,256,512)
        out = self.attenPool(out) #attenPool returns (batch_size,1)
        out = self.sigmoid(out) #sigmoid returns (batch_size,1)
        return out


class wavLMMetricPredictorFullTransformerSmall(nn.Module):
    def __init__(self, feat_seq=384):
        super().__init__()

        self.feat_extract = WavLMFull()
        self.feat_extract.requires_grad_(False)
        self.cc = CenterCrop(feat_seq)
        self.config = Config(
            "HUBERT_FULL_CONFIG",
            Input.XLSR,
            feat_seq_len=feat_seq,
            dim_transformer=256,
            xlsr_name="hubert_full",
            nhead_transformer=4,
            nlayers_transformer=4,
        )
        self.norm_input = nn.BatchNorm1d(768)
        self.transformer = TransformerWrapper(self.config)
        self.attenPool = PoolAttFF(self.config.dim_transformer)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #input is (batch_size,time)        
        out_feats = self.feat_extract(x) #wavLM returns (batch_size,time,768)
        out_feats = self.cc(out_feats) #pad / crop the time dimension to 256 (batch_size,256,768)
        out = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize the input (batch_size,256,768)
        out = self.transformer(out_feats) #transformer returns (batch_size,256,768)
        out = self.attenPool(out) #attenPool returns (batch_size,1)
        out = self.sigmoid(out) #sigmoid returns (batch_size,1)
        return out

class wavLMMetricPredictorFullTransformerSmallT(nn.Module):
    def __init__(self, feat_seq=384):
        super().__init__()

        self.feat_extract = WavLMFull()
        self.feat_extract.requires_grad_(False)
        self.cc = CenterCrop(feat_seq)
        self.config = Config(
            "HUBERT_FULL_CONFIG",
            Input.XLSR,
            feat_seq_len=768,
            dim_transformer=256,
            xlsr_name="hubert_full_t",
            nhead_transformer=4,
            nlayers_transformer=4,
        )
        self.norm_input = nn.BatchNorm1d(feat_seq)
        self.transformer = TransformerWrapper(self.config)
        self.attenPool = PoolAttFF(self.config.dim_transformer)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #input is (batch_size,time)        
        out_feats = self.feat_extract(x) #wavLM returns (batch_size,time,768)
        out_feats = self.cc(out_feats) #pad / crop the time dimension to 256 (batch_size,256,768)

        out_feats = out_feats.permute(0,2,1) #swap the time and feature dimension (batch_size,768,256)

        out = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize the input (batch_size,768,256)
        out = self.transformer(out_feats) #transformer returns (batch_size,768,256)
        out = self.attenPool(out) #attenPool returns (batch_size,1)
        out = self.sigmoid(out) #sigmoid returns (batch_size,1)
        return out

    
class wavLMMetricPredictorFullLayersTransformerSmall(nn.Module):
    def __init__(self, feat_seq=384):
        super().__init__()

        self.feat_extract = WavLMFull_all()
        self.feat_extract.requires_grad_(False)
        self.cc = CenterCrop(feat_seq)
        self.config = Config(
            "HUBERT_FULL_CONFIG",
            Input.XLSR,
            feat_seq_len=feat_seq,
            dim_transformer=256,
            xlsr_name="hubert_full",
            nhead_transformer=4,
            nlayers_transformer=4,
        )
        self.norm_input = nn.BatchNorm1d(768)
        self.transformer = TransformerWrapper(self.config)
        self.attenPool = PoolAttFF(self.config.dim_transformer)
        self.sigmoid = nn.Sigmoid()

        self.layer_weights = nn.Parameter(torch.ones(13))
        self.softmax = nn.Softmax(dim=0)


        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        #input is (batch_size,time)    
        out_feats_list = self.feat_extract(x) # wavLM returns a list of 13 tensors (batch_size,time,768)
        
        out_feats = torch.stack(out_feats_list,dim=-1) #stack the list of tensors into a single tensor (batch_size,time,768,13)

        
        out_feats = out_feats @ self.softmax(self.layer_weights) #multiply the tensor by the softmax of the layer weights (batch_size,time,768,13) * (13) -> (batch_size,time,768,1)
        print(self.layer_weights)
        out_feats = self.cc(out_feats) #pad / crop the time dimension to feat_seq (batch_size,feat_seq,768)
        out = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize the input (batch_size,768,feat_seq)

        out = self.transformer(out_feats) #transformer returns (batch_size,feat_seq,768)

        out = self.attenPool(out) #attenPool returns (batch_size,1)
        out = self.sigmoid(out) #sigmoid returns (batch_size,1)
        return out


class wavLMMetricPredictorFullLayersTransformerSmallT(nn.Module):
    def __init__(self, feat_seq=384):
        super().__init__()

        self.feat_extract = WavLMFull_all()
        self.feat_extract.requires_grad_(False)
        self.cc = CenterCrop(feat_seq)
        self.config = Config(
            "HUBERT_FULL_CONFIG",
            Input.XLSR,
            feat_seq_len=768,
            dim_transformer=256,
            xlsr_name="hubert_full_t",
            nhead_transformer=4,
            nlayers_transformer=4,
        )
        self.norm_input = nn.BatchNorm1d(feat_seq)
        self.transformer = TransformerWrapper(self.config)
        self.attenPool = PoolAttFF(self.config.dim_transformer)
        self.sigmoid = nn.Sigmoid()

        self.layer_weights = nn.Parameter(torch.ones(13))
        self.softmax = nn.Softmax(dim=0)


        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        #input is (batch_size,time)    
        out_feats_list = self.feat_extract(x) # wavLM returns a list of 13 tensors (batch_size,time,768)
        
        out_feats = torch.stack(out_feats_list,dim=-1) #stack the list of tensors into a single tensor (batch_size,time,768,13)

        
        out_feats = out_feats @ self.softmax(self.layer_weights) #multiply the tensor by the softmax of the layer weights (batch_size,time,768,13) * (13) -> (batch_size,time,768,1)
        print(self.layer_weights)
        out_feats = self.cc(out_feats) #pad / crop the time dimension to feat_seq (batch_size,feat_seq,768)
        out_feats = out_feats.permute(0,2,1) #swap the time and feature dimension (batch_size,768,time)
        out = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize the input (batch_size,time,768)

        out = self.transformer(out_feats) #transformer returns (batch_size,time,768)

        out = self.attenPool(out) #attenPool returns (batch_size,1)
        out = self.sigmoid(out) #sigmoid returns (batch_size,1)
        return out




if __name__ == "__main__":
    import torchinfo
    import torchaudio





    aud_path = "/mnt/parscratch/users/acp20glc/VoiceBank/clean_testset_wav_16k/p232_001.wav"
    input,_ = torchaudio.load(aud_path)
    input = input.cuda()
    
    model = wavLMMetricPredictorEncoderTransformerSmall().cuda()
    torchinfo.summary(model, input_size=(16,16000))
    print(input.shape)
    output= model(input)
    print(output.shape)

    del model
    model = wavLMMetricPredictorEncoderTransformerSmallT().cuda()
    torchinfo.summary(model, input_size=(16,16000))
    print(input.shape)
    output= model(input)
    print(output.shape)

    del model
    model = wavLMMetricPredictorFullTransformerSmall().cuda()
    torchinfo.summary(model, input_size=(16,16000))
    print(input.shape)
    output= model(input)
    print(output.shape)

    del model
    model = wavLMMetricPredictorFullTransformerSmallT().cuda()
    torchinfo.summary(model, input_size=(16,16000))
    print(input.shape)
    output= model(input)
    print(output.shape)

    del model
    model = wavLMMetricPredictorFullLayersTransformerSmall().cuda()
    torchinfo.summary(model, input_size=(16,16000))
    print(input.shape)
    output= model(input)
    print(output.shape)

    del model
    model = wavLMMetricPredictorFullLayersTransformerSmallT().cuda()
    torchinfo.summary(model, input_size=(16,16000))
    print(input.shape)
    output= model(input)
    print(output.shape)