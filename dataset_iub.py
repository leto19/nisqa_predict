"""
handles mapping between csv files and torch datasets for IUB dataset

"""
import pandas as pd
import os
from speechbrain.dataio.dataset import DynamicItemDataset
import speechbrain as sb
from speechbrain.processing.speech_augmentation import Resample
def get_dataframe(data_root,name):
    
    FILE_FILE_PATH = os.path.join(data_root,"%s.csv"%name)

    #create an empty dataframe
    data = pd.DataFrame()

    file_data_in = pd.read_csv(FILE_FILE_PATH)

    data['id'] = file_data_in['AudioName'].apply(lambda x: x.split("/")[-1].split(".")[0])
    data['deg_path'] = file_data_in['AudioName'].apply(lambda x: os.path.join(data_root,"16k_speech",x.split("/")[-1]))
    data['mos'] = (file_data_in['ScaledMOS'] / 12.5) + 0.2
    #fix all mos values below 0.2 to 0.2

    #print(data.head()) 
    #set the db column to be 'iub'
    data['db'] = "iub" 
    return data

def convert_dataframe_to_dict(in_df):
    """
    convert a dataframe to a dictionary of lists
    """
    out_dict = {}
    for row in in_df.itertuples():
        out_dict[row.id] = {"deg_path":row.deg_path,"db":row.db,"mos":row.mos}
    return out_dict

resampler = Resample(orig_freq=32000, new_freq=16000)


def load_audio(path):
    """load the audio and downsample to 16kHz"""
    
    audio_16k = sb.dataio.dataio.read_audio(path)
    #audio_16k = resampler(audio_48k.unsqueeze(0)).squeeze(0)   
    
    return audio_16k

def normalise_score(score):
    return score/5.0


def get_dataset(data_root,name):

    if type(name) == list:
        data_dict = {}
        for n in name:
            dataframe = get_dataframe(data_root,n)
            data_dict.update(convert_dataframe_to_dict(dataframe))
    else:   
        dataframe = get_dataframe(data_root,name)
        data_dict = convert_dataframe_to_dict(dataframe)
    dynamic_items = [
        {"func": lambda l: load_audio(l),
         "takes": "deg_path",
        "provides": "deg_audio"},
        #{"func": lambda l: load_audio(l),
        # "takes": "ref_path",
        #"provides": "ref_audio"},
    ]
    dataset = DynamicItemDataset(data_dict, dynamic_items)
    dataset.set_output_keys(["id","db","deg_audio","mos"])

    return dataset,data_dict


if __name__ =="__main__":
   dataset1 = get_dataset("iub_data/",["audio_scaled_mos_cosine","audio_scaled_mos_voices"])
   print(len(dataset1))
   for batch in dataset1:
       print(batch["deg_audio"])
       print(batch["mos"])
       break