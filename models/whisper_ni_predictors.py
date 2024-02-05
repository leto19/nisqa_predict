import torch
import torch.nn.functional as F
from torch import Tensor, nn
try:
    from whisper_wrapper import WhisperWrapper_full,WhisperWrapper_encoder,pad_or_trim, log_mel_spectrogram
    from transformer_wrapper import TransformerWrapper
    from transformer_config import CenterCrop,Config,Input
except:
    from models.whisper_wrapper import WhisperWrapper_full,WhisperWrapper_encoder, pad_or_trim, log_mel_spectrogram
    from models.transformer_wrapper import TransformerWrapper
    from models.transformer_config import CenterCrop,Config,Input    
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



    
class whisperMetricPredictorEncoderTransformerSmall(nn.Module):
    """Transformer based varient on metric estimator

    based on https://github.com/lcn-kul/xls-r-analysis-sqa/
    """
    def __init__(
        self, feat_seq=1500):
        super().__init__()
        self.norm_input = nn.BatchNorm1d(768)

        self.feat_extract = WhisperWrapper_encoder(use_feat_extractor=True)
        self.feat_extract.requires_grad_(False)

        self.config  = Config(
        "WHISPER_ENCODER_CONFIG",
        Input.XLSR,
        feat_seq_len=feat_seq,
        dim_transformer=256,
        xlsr_name="whisper_encoder",
        nhead_transformer=4,
        nlayers_transformer=4,
        )
        self.transformer = TransformerWrapper(self.config)

        
        
        self.attenPool = PoolAttFF(self.config.dim_transformer)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
    
        out_feats = self.feat_extract(x) #whisper encoder returns (B, 1500, 512)
        out_feats = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 1500, 512)
        out = self.transformer(out_feats) # transformer returns (B, 1500, 256)
        out = self.attenPool(out) #attenPool returns (B, 1)
        out = self.sigmoid(out) #sigmoid returns (B, 1)
        return out


class whisperMetricPredictorEncoderTransformerSmallT(nn.Module):
    """Transformer based varient on metric estimator

    based on
    """
    def __init__(
        self, feat_seq=1500):
        super().__init__()
        self.norm_input = nn.BatchNorm1d(feat_seq)

        self.feat_extract = WhisperWrapper_encoder(use_feat_extractor=True)
        self.feat_extract.requires_grad_(False)

        self.config  = Config(
        "WHISPER_ENCODER_CONFIG",
        Input.XLSR,
        feat_seq_len=768,
        dim_transformer=256,
        xlsr_name="whisper_encoder_t",
        nhead_transformer=4,
        nlayers_transformer=4,
        )
        self.transformer = TransformerWrapper(self.config)

        
        
        self.attenPool = PoolAttFF(self.config.dim_transformer)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
    
        out_feats = self.feat_extract(x) #whisper encoder returns (B, 1500, 512)
        out_feats = out_feats.permute(0,2,1) #normalize and permute back to (B, 1500, 512)
        out_feats = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 1500, 512)

        out = self.transformer(out_feats) # transformer returns (B, 1500, 256)
        out = self.attenPool(out) #attenPool returns (B, 1)
        out = self.sigmoid(out) #sigmoid returns (B, 1)
        return out
    
class whisperMetricPredictorEncoderLayersTransformerSmall(nn.Module):
    """Transformer based varient on metric estimator

    based on https://github.com/lcn-kul/xls-r-analysis-sqa/
    """
    def __init__(
        self, feat_seq=1500):
        super().__init__()
        self.norm_input = nn.BatchNorm1d(768)

        self.feat_extract = WhisperWrapper_encoder(use_feat_extractor=True, layer=-1)
        self.feat_extract.requires_grad_(False)
        self.layer_weights = nn.Parameter(torch.ones(13))
        self.softmax = nn.Softmax(dim=0)

        self.config  = Config(
        "WHISPER_ENCODER_CONFIG",
        Input.XLSR,
        feat_seq_len=feat_seq,
        dim_transformer=256,
        xlsr_name="whisper_encoder",
        nhead_transformer=4,
        nlayers_transformer=4,
        )
        self.transformer = TransformerWrapper(self.config)

        
        
        self.attenPool = PoolAttFF(self.config.dim_transformer)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
    
        out_feats = self.feat_extract(x) #whisper encoder a list of 13 tensors of shape (B, 1500, 512)
        out_feats = out_feats @ self.softmax(self.layer_weights) #weighted sum of the 13 tensors
        print(self.layer_weights)
        out_feats = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 1500, 512)
        out = self.transformer(out_feats) # transformer returns (B, 1500, 256)
        out = self.attenPool(out) #attenPool returns (B, 1)
        out = self.sigmoid(out) #sigmoid returns (B, 1)
        return out


class whisperMetricPredictorEncoderLayersTransformerSmallT(nn.Module):
    """Transformer based varient on metric estimator

    based on https://github.com/lcn-kul/xls-r-analysis-sqa/
    """
    def __init__(
        self, feat_seq=1500):
        super().__init__()
        self.norm_input = nn.BatchNorm1d(feat_seq)

        self.feat_extract = WhisperWrapper_encoder(use_feat_extractor=True, layer=-1)
        self.feat_extract.requires_grad_(False)
        self.layer_weights = nn.Parameter(torch.ones(13))
        self.softmax = nn.Softmax(dim=0)

        self.config  = Config(
        "WHISPER_ENCODER_CONFIG",
        Input.XLSR,
        feat_seq_len=768,
        dim_transformer=256,
        xlsr_name="whisper_encoder_t",
        nhead_transformer=4,
        nlayers_transformer=4,
        )
        self.transformer = TransformerWrapper(self.config)

        
        
        self.attenPool = PoolAttFF(self.config.dim_transformer)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
    
        out_feats = self.feat_extract(x) #whisper encoder a list of 13 tensors of shape (B, 1500, 512)
        out_feats = out_feats @ self.softmax(self.layer_weights) #weighted sum of the 13 tensors
        print(self.layer_weights)

        out_feats = out_feats.permute(0,2,1) #swap axes to (B, 512, 1500)

        out_feats = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 1500, 512)
        out = self.transformer(out_feats) # transformer returns (B, 1500, 256)
        out = self.attenPool(out) #attenPool returns (B, 1)
        out = self.sigmoid(out) #sigmoid returns (B, 1)
        return out


class whisperMetricPredictorMelTransformerSmall(nn.Module):
    """Transformer based varient on metric estimator

    based on https://github.com/lcn-kul/xls-r-analysis-sqa/
    """
    def __init__(self, feat_seq=3000):
        super().__init__()


        self.config = Config(
        "MFCC_TRANSFORMER_32DEEP_CONFIG",
        Input.MFCC,
        feat_seq_len=feat_seq,
        dim_transformer=256,
        xlsr_name=None,
        nhead_transformer=4,
        nlayers_transformer=4,
    )
        self.norm_input = nn.BatchNorm1d(80)

        self.transformer = TransformerWrapper(self.config)

        self.attenPool = PoolAttFF(self.config.dim_transformer)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N_SAMPLES = 16000*30
        data_padded = pad_or_trim(x, length=N_SAMPLES) #pad or trim to 30 seconds, returns (B, 480000)
        data_feats = log_mel_spectrogram(data_padded).swapaxes(1,2) #returns (B, 3000, 80)
    
        data_feats = self.norm_input(data_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 3000, 80)
        out_trans = self.transformer(data_feats) # transformer returns (B, 3000, 256)
        out = self.attenPool(out_trans) #attenPool returns (B, 1)
        out = self.sigmoid(out)

        return out


class whisperMetricPredictorMelTransformerSmallT (nn.Module):
    """Transformer based varient on metric estimator

    based on https://github.com/lcn-kul/xls-r-analysis-sqa/
    """
    def __init__(self, feat_seq=3000):
        super().__init__()


        self.config = Config(
        "MFCC_TRANSFORMER_32DEEP_CONFIG",
        Input.MFCC,
        feat_seq_len=80,
        dim_transformer=256,
        xlsr_name="mel_T",
        nhead_transformer=4,
        nlayers_transformer=4,
    )
        self.norm_input = nn.BatchNorm1d(feat_seq)

        self.transformer = TransformerWrapper(self.config)

        self.attenPool = PoolAttFF(self.config.dim_transformer)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N_SAMPLES = 16000*30
        data_padded = pad_or_trim(x, length=N_SAMPLES) #pad or trim to 30 seconds, returns (B, 480000)
        data_feats = log_mel_spectrogram(data_padded) #returns (B, 80, 3000)
    
        data_feats = self.norm_input(data_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 80, 3000)
        out_trans = self.transformer(data_feats) # transformer returns (B, 3000, 256)
        out = self.attenPool(out_trans) #attenPool returns (B, 1)
        out = self.sigmoid(out)

        return out



class whisperMetricPredictorFullTransformerSmall(nn.Module):
    def __init__(self, feat_seq=768//2):
        super().__init__()



        self.feat_extract = WhisperWrapper_full(layer=-1,use_feat_extractor=True)
        self.feat_extract.requires_grad_(False)
        self.config = Config(
            "WHISPER_FULL_CONFIG",
            Input.XLSR,
            feat_seq_len=feat_seq,
            dim_transformer=256,
            xlsr_name="whisper_full",
            nhead_transformer=4,
            nlayers_transformer=4,
        )
        self.cc = CenterCrop(feat_seq)
        self.norm_input = nn.BatchNorm1d(768)
        self.transformer = TransformerWrapper(self.config)
        self.norm_input = nn.BatchNorm1d(768)
        self.attenPool = PoolAttFF(self.config.dim_transformer)


        self.sigmoid = nn.Sigmoid()
    def forward(self, x):        
        out_feats = self.feat_extract(x)[:,:,:,-1] #whisper encoder returns (B, 1500, 768)
        out_feats = self.cc (out_feats) #center crop to 384
        out_feats = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 384, 768)
        out = self.transformer(out_feats) # transformer returns (B, 384, 256)
        out = self.attenPool(out) #attenPool returns (B, 1)
        out = self.sigmoid(out) #sigmoid returns (B, 1)
        return out

class whisperMetricPredictorFullTransformerSmallT(nn.Module):
    def __init__(self, feat_seq=384):
        super().__init__()



        self.feat_extract = WhisperWrapper_full(layer=-1,use_feat_extractor=True)
        self.feat_extract.requires_grad_(False)
        self.config = Config(
            "WHISPER_FULL_CONFIG",
            Input.XLSR,
            feat_seq_len=768,
            dim_transformer=256,
            xlsr_name="whisper_full_t",
            nhead_transformer=4,
            nlayers_transformer=4,
        )
        self.cc = CenterCrop(feat_seq)
        self.norm_input = nn.BatchNorm1d(feat_seq)

        self.transformer = TransformerWrapper(self.config)
        self.attenPool = PoolAttFF(self.config.dim_transformer)


        self.sigmoid = nn.Sigmoid()
    def forward(self, x):        
        out_feats = self.feat_extract(x)[:,:,:,-1] #whisper encoder returns (B, W, 768)
        out_feats = self.cc (out_feats) #center crop to 384
        
        out_feats= out_feats.permute(0,2,1) #swap axes to (B, 768, 384)

        out_feats = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 768, 384)
        
        out = self.transformer(out_feats) # transformer returns (B, 768, 256)
        out = self.attenPool(out) #attenPool returns (B, 1)
        out = self.sigmoid(out) #sigmoid returns (B, 1)
        return out



class whisperMetricPredictorFullLayersTransformerSmall(nn.Module):
    def __init__(self, feat_seq=768//2):
        super().__init__()



        self.feat_extract = WhisperWrapper_full(layer=-1,use_feat_extractor=True)
        self.feat_extract.requires_grad_(False)
        self.config = Config(
            "WHISPER_FULL_CONFIG",
            Input.XLSR,
            feat_seq_len=feat_seq,
            dim_transformer=256,
            xlsr_name="whisper_full",
            nhead_transformer=4,
            nlayers_transformer=4,
        )
        self.cc = CenterCrop(feat_seq)
        self.norm_input = nn.BatchNorm1d(768)
        self.transformer = TransformerWrapper(self.config)
        self.norm_input = nn.BatchNorm1d(768)
        self.attenPool = PoolAttFF(self.config.dim_transformer)
        self.layer_weights = nn.Parameter(torch.ones(12))
        self.softmax = nn.Softmax(dim=0)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):        
        out_feats = self.feat_extract(x) #whisper encoder returns list (B, 1500, 768,12)
        out_feats = out_feats @ self.softmax(self.layer_weights) #weighted sum of the 12 tensors (B, 1500, 768) 
        print(self.layer_weights)
        out_feats = self.cc (out_feats) #center crop to 384
        out_feats = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 384, 768)
        out = self.transformer(out_feats) # transformer returns (B, 384, 256)
        out = self.attenPool(out) #attenPool returns (B, 1)
        out = self.sigmoid(out) #sigmoid returns (B, 1)
        return out


class whisperMetricPredictorFullLayersTransformerSmallT(nn.Module):
    def __init__(self, feat_seq=384):
        super().__init__()



        self.feat_extract = WhisperWrapper_full(layer=-1,use_feat_extractor=True)
        self.feat_extract.requires_grad_(False)
        self.config = Config(
            "WHISPER_FULL_CONFIG",
            Input.XLSR,
            feat_seq_len=768,
            dim_transformer=256,
            xlsr_name="whisper_full_t",
            nhead_transformer=4,
            nlayers_transformer=4,
        )
        self.cc = CenterCrop(feat_seq)
        self.norm_input = nn.BatchNorm1d(feat_seq)

        self.transformer = TransformerWrapper(self.config)
        self.attenPool = PoolAttFF(self.config.dim_transformer)
        self.layer_weights = nn.Parameter(torch.ones(12))
        self.softmax = nn.Softmax(dim=0)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):        
        out_feats = self.feat_extract(x) #whisper encoder returns list (B, 1500, 768,12)
        out_feats = out_feats @ self.softmax(self.layer_weights) #weighted sum of the 12 tensors (B, 1500, 768) 
        print(self.layer_weights)
        out_feats = self.cc (out_feats) #center crop to 384
        
        out_feats= out_feats.permute(0,2,1) #swap axes to (B, 768, 384)

        out_feats = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 768, 384)
        
        out = self.transformer(out_feats) # transformer returns (B, 768, 256)
        out = self.attenPool(out) #attenPool returns (B, 1)
        out = self.sigmoid(out) #sigmoid returns (B, 1)
        return out





# class whisperMetricPredictorMelTransformerSmall(nn.Module):
#     """Transformer based varient on metric estimator

#     based on https://github.com/lcn-kul/xls-r-analysis-sqa/
#     """
#     def __init__(
#         self, dim_extractor=768, hidden_size=768//2, activation=nn.LeakyReLU,):
#         super().__init__()

#         self.activation = activation(negative_slope=0.3)

#         self.transformer = TransformerWrapper(MFCC_TRANSFORMER_32DEEP_CONFIG)

#         self.attenPool = PoolAttFF(256)
        
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         N_SAMPLES = 16000*30
#         data_padded = pad_or_trim(x, length=N_SAMPLES)
#         #print("data padded shape: ",data_padded.shape)
#         data_feats = log_mel_spectrogram(data_padded).swapaxes(1,2)
#         #print("data feats shape: ",data_feats.shape)
#         #print(self.transformer)
#         out_trans = self.transformer(data_feats)
#         #print("transformer out",out_trans.shape)
#         out = self.attenPool(out_trans)
#         out = self.sigmoid(out)

#         return out#,data_feats,out_trans

# class whisperMetricPredictorEncoderLayersTransformerSmall(nn.Module):
#     def __init__(
#         self, dim_extractor=768, hidden_size=768//2, activation=nn.LeakyReLU,
#     ):
#         super().__init__()

#         self.activation = activation(negative_slope=0.3)

#         self.feat_extract = WhisperWrapper_encoder(use_feat_extractor=True, layer=-1)
#         self.feat_extract.requires_grad_(False)

#         self.transformer = TransformerWrapper(WHISPER_ENCODER_CONFIG_SMALL)
#         self.layer_weights = nn.Parameter(torch.ones(13))
#         self.softmax = nn.Softmax(dim=0)

#         self.attenPool = PoolAttFF(256)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         """Forward pass of the model.

#         Args:
#             x (torch.Tensor): Input tensor.

#         Returns:
#             torch.Tensor: Output tensor.
#         """
#         out_feats = self.feat_extract(x)
#         out_feats = out_feats @ self.softmax(self.layer_weights)
#         print(self.layer_weights)

#         out = self.transformer(out_feats)
#         out = self.attenPool(out)
#         out = self.sigmoid(out)

#         return out


# class whisperMetricPredictorEncoderTransformerSmall(nn.Module):
#     """Transformer based varient on metric estimator

#     based on https://github.com/lcn-kul/xls-r-analysis-sqa/
#     """
#     def __init__(
#         self, dim_extractor=768, hidden_size=768//2, activation=nn.LeakyReLU,
#     ):
#         super().__init__()

#         self.activation = activation(negative_slope=0.3)



#         self.feat_extract = WhisperWrapper_encoder(use_feat_extractor=True)
#         self.feat_extract.requires_grad_(False)

        
#         self.transformer = TransformerWrapper(WHISPER_ENCODER_CONFIG_SMALL)

        
        
#         self.attenPool = PoolAttFF(256)
        
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
    
#         out_feats = self.feat_extract(x)
#         out = self.transformer(out_feats)
#         out = self.attenPool(out)
#         out = self.sigmoid(out)

#         return out

# class whisperMetricPredictorEncoderTransformerSmall2(nn.Module):
#     """Transformer based varient on metric estimator

#     based on https://github.com/lcn-kul/xls-r-analysis-sqa/
#     """
#     def __init__(
#         self, dim_extractor=768, hidden_size=768//2, activation=nn.LeakyReLU,
#     ):
#         super().__init__()

#         self.activation = activation(negative_slope=0.3)



#         self.feat_extract = WhisperWrapper_encoder(use_feat_extractor=True)
#         self.feat_extract.requires_grad_(False)

        
#         self.transformer = TransformerWrapper(WHISPER_ENCODER_CONFIG_SMALLER)

        
        
#         self.attenPool = PoolAttFF(128)
        
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
    
#         out_feats = self.feat_extract(x)
#         out = self.transformer(out_feats)
#         out = self.attenPool(out)
#         out = self.sigmoid(out)

#         return out

# class whisperMetricPredictorEncoderTransformerSmall2_T(nn.Module):
#     """Transformer based varient on metric estimator

#     based on https://github.com/lcn-kul/xls-r-analysis-sqa/
#     """
#     def __init__(
#         self, dim_extractor=768, hidden_size=768//2, activation=nn.LeakyReLU,
#     ):
#         super().__init__()

#         self.activation = activation(negative_slope=0.3)



#         self.feat_extract = WhisperWrapper_encoder(use_feat_extractor=True)
#         self.feat_extract.requires_grad_(False)

        
#         self.transformer = TransformerWrapper(WHISPER_ENCODER_CONFIG_SMALLER_T)

        
        
#         self.attenPool = PoolAttFF(128)
        
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
    
#         out_feats = self.feat_extract(x)
#         out_feats = out_feats.permute(0,2,1)
#         print(out_feats.shape)
#         out = self.transformer(out_feats)
#         out = self.attenPool(out)
#         out = self.sigmoid(out)

#         return out


# class whisperMetricPredictorEncoderTransformerSmall2noSig(nn.Module):
#     """Transformer based varient on metric estimator

#     based on https://github.com/lcn-kul/xls-r-analysis-sqa/
#     """
#     def __init__(
#         self, dim_extractor=768, hidden_size=768//2, activation=nn.LeakyReLU,
#     ):
#         super().__init__()

#         self.activation = activation(negative_slope=0.3)



#         self.feat_extract = WhisperWrapper_encoder(use_feat_extractor=True)
#         self.feat_extract.requires_grad_(False)

        
#         self.transformer = TransformerWrapper(WHISPER_ENCODER_CONFIG_SMALLER)

        
        
#         self.attenPool = PoolAttFF(128)
        
#         #self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
    
#         out_feats = self.feat_extract(x)
#         out = self.transformer(out_feats)
#         out = self.attenPool(out)
#         #out = self.sigmoid(out)

#         return out

# class whisperMetricPredictorEncoderLayers(nn.Module):
#     """Metric estimator for enhancement training.

#     Consists of:
#      * four 2d conv layers
#      * channel averaging
#      * three linear layers

#     Arguments
#     ---------
#     kernel_size : tuple
#         The dimensions of the 2-d kernel used for convolution.
#     base_channels : int
#         Number of channels used in each conv layer.
#     """

#     def __init__(
#         self, dim_extractor=768, hidden_size=768//2, activation=nn.LeakyReLU,
#     ):
#         super().__init__()

#         self.activation = activation(negative_slope=0.3)

#         #self.BN = nn.BatchNorm1d(num_features=1, momentum=0.01)


#         self.feat_extract = WhisperWrapper_encoder(use_feat_extractor=True,layer=-1)
#         self.feat_extract.requires_grad_(False)
#         self.layer_weights = nn.Parameter(torch.ones(13))
#         self.softmax = nn.Softmax(dim=0)
        
#         self.blstm = nn.LSTM(
#             input_size=dim_extractor,
#             hidden_size=hidden_size,
#             num_layers=2,
#             dropout=0.1,
#             bidirectional=True,
#             batch_first=True,
#         )
        
        
#         self.attenPool = PoolAttFF(dim_extractor)
        
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         #print("----- IN THE MODEL -----")
#         #print(x.shape)
        
#         #x = self.BN(x)
        
#         out_feats = self.feat_extract(x)#.permute(0,2,1)
#         #print(out_feats.shape)
#         out_feats = out_feats @ self.softmax(self.layer_weights)
#         print(self.layer_weights)
#         #print(out_feats.shape)
#         out,_ = self.blstm(out_feats)
#         #out = out_feats
#         out = self.attenPool(out)
#         out = self.sigmoid(out)
#         #print("----- LEAVING THE MODEL -----")

#         return out
    
# class whisperMetricPredictorFull(nn.Module):
#     def __init__(
#         self, dim_extractor=768, hidden_size=768//2, activation=nn.LeakyReLU,
#     ):
#         super().__init__()

#         self.activation = activation(negative_slope=0.3)


#         self.feat_extract = WhisperWrapper_full(layer=12,use_feat_extractor=True)
#         self.feat_extract.requires_grad_(False)

        
#         self.blstm = nn.LSTM(
#             input_size=dim_extractor,
#             hidden_size=hidden_size,
#             num_layers=2,
#             dropout=0.1,
#             bidirectional=True,
#             batch_first=True,
#         )

#         self.attenPool1 = PoolAttFF(dim_extractor)


#         self.sigmoid = nn.Sigmoid()
#     def forward(self, x):
#         #print("----- IN THE MODEL -----")
#         #print(x.shape)
#         #out = self.BN(x)
        
#         out_feats = self.feat_extract(x)


#         out,_ = self.blstm(out_feats)
#         #out = out_feats
#         out1 = self.attenPool1(out)
#         out1 = self.sigmoid(out1)

        

#         return out1

# class whisperMetricPredictorFullTransformerSmall(nn.Module):
#     def __init__(
#         self, dim_extractor=256, hidden_size=768//2, activation=nn.LeakyReLU,
#     ):
#         super().__init__()

#         self.activation = activation(negative_slope=0.3)


#         self.feat_extract = WhisperWrapper_full(layer=-1,use_feat_extractor=True)
#         self.feat_extract.requires_grad_(False)

#         self.transformer = TransformerWrapper(WHISPER_FULL_CONFIG_SMALL)

#         self.attenPool1 = PoolAttFF(dim_extractor)


#         self.sigmoid = nn.Sigmoid()
#     def forward(self, x):
#         #print("----- IN THE MODEL -----")
#         #print(x.shape)
#         #out = self.BN(x)
        
#         out_feats = self.feat_extract(x)[:,:,:,-1]
#         print("out_feats",out_feats.shape)

#         out = self.transformer(out_feats)
#         print(out.shape)
#         #out = out_feats
#         out1 = self.attenPool1(out)
#         out1 = self.sigmoid(out1)

        

#         return out1

# class whisperMetricPredictorFullLayersTransformerSmall(nn.Module):
#     def __init__(
#         self, dim_extractor=256, hidden_size=768//2, activation=nn.LeakyReLU,
#     ):
#         super().__init__()

#         self.activation = activation(negative_slope=0.3)


#         self.feat_extract = WhisperWrapper_full(layer=-1,use_feat_extractor=True)
#         self.feat_extract.requires_grad_(False)


#         self.layer_weights = nn.Parameter(torch.ones(12))
#         self.softmax = nn.Softmax(dim=0)

#         self.transformer = TransformerWrapper(WHISPER_FULL_CONFIG_SMALL)

#         self.attenPool1 = PoolAttFF(dim_extractor)


#         self.sigmoid = nn.Sigmoid()
#     def forward(self, x):
#         #print("----- IN THE MODEL -----")
#         #print(x.shape)
#         #out = self.BN(x)
        
#         out_feats = self.feat_extract(x)
#         print("out_feats",out_feats.shape)
#         out_feats = out_feats @ self.softmax(self.layer_weights)
#         print(self.layer_weights)
#         out = self.transformer(out_feats)
#         print(out.shape)
#         #out = out_feats
#         out1 = self.attenPool1(out)
#         out1 = self.sigmoid(out1)

        

#         return out1




if __name__ == "__main__":
    import torchinfo
    import torchaudio





    aud_path = "/mnt/parscratch/users/acp20glc/VoiceBank/clean_testset_wav_16k/p232_001.wav"
    input,_ = torchaudio.load(aud_path)
    input = input#.cuda()
    
    model = whisperMetricPredictorEncoderTransformerSmall()
    torchinfo.summary(model, input_size=(16,16000))
    print(input.shape)
    output= model(input)
    print(output.shape)

    del model
    model = whisperMetricPredictorEncoderLayersTransformerSmall()
    torchinfo.summary(model, input_size=(16,16000))
    print(input.shape)
    output= model(input)
    print(output.shape)

    del model
    model = whisperMetricPredictorEncoderTransformerSmallT()
    torchinfo.summary(model, input_size=(16,16000))
    print(input.shape)
    output= model(input)
    print(output.shape)

    del model
    model = whisperMetricPredictorEncoderLayersTransformerSmallT()
    torchinfo.summary(model, input_size=(16,16000))
    print(input.shape)
    output= model(input)
    print(output.shape)

    del model
    model = whisperMetricPredictorMelTransformerSmall()
    torchinfo.summary(model, input_size=(16,16000))
    print(input.shape)
    output= model(input)
    print(output.shape)

    del model
    model = whisperMetricPredictorMelTransformerSmallT()
    torchinfo.summary(model, input_size=(16,16000))
    print(input.shape)
    output= model(input)
    print(output.shape)

    del model
    model = whisperMetricPredictorFullTransformerSmall()
    torchinfo.summary(model, input_size=(16,16000))
    print(input.shape)
    output= model(input)
    print(output.shape)

    del model
    model = whisperMetricPredictorFullLayersTransformerSmall()
    torchinfo.summary(model, input_size=(16,16000))
    print(input.shape)
    output= model(input)
    print(output.shape)

    del model
    model = whisperMetricPredictorFullTransformerSmallT()
    torchinfo.summary(model, input_size=(16,16000))
    print(input.shape)
    output= model(input)
    print(output.shape)

    del model
    model = whisperMetricPredictorFullLayersTransformerSmallT()
    torchinfo.summary(model, input_size=(16,16000))
    print(input.shape)
    output= model(input)
    print(output.shape)

    print("done :)")
    