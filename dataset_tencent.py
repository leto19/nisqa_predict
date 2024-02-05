"""
handles mapping between csv files and torch datasets for NISQA dataset

"""
import pandas as pd
import os
from speechbrain.dataio.dataset import DynamicItemDataset
import speechbrain as sb
from speechbrain.processing.speech_augmentation import Resample
def get_dataframe(data_root,name):
    
    FILE_FILE_PATH = os.path.join(data_root,"%s.csv"%name.replace("_16k","MOS_16k"))

    #create an empty dataframe
    data = pd.DataFrame()

    file_data_in = pd.read_csv(FILE_FILE_PATH)

    data['id'] = file_data_in['deg_wav'].apply(lambda x: x.split("/")[-1].split(".")[0])
    data['deg_path'] = file_data_in['deg_wav'].apply(lambda x: os.path.join(data_root,x.strip("./")))
    data['mos'] = file_data_in['mos'] /5
    #set the db column to be 'tencent'
    data['db'] = "tencent"
    #print(data.head())  
    return data

def convert_dataframe_to_dict(in_df):
    """
    convert a dataframe to a dictionary of lists
    """
    out_dict = {}
    for row in in_df.itertuples():
        out_dict[row.id] = {"deg_path":row.deg_path,"db":row.db,"mos":row.mos}
    return out_dict

resampler = Resample(orig_freq=48000, new_freq=16000)


def load_audio(path):
    """load the audio and downsample to 16kHz"""
    
    audio_48k = sb.dataio.dataio.read_audio(path)
    audio_16k = resampler(audio_48k.unsqueeze(0)).squeeze(0)   
    
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
   dataset1 = get_dataset("tencent_data/","withoutReverberationTrainDev_16k")
   dataset2 = get_dataset("tencent_data/","withReverberationTrainDev_16k")
   print(len(dataset1))
   print(len(dataset2))
   