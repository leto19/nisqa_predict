"""
handles mapping between csv files and torch datasets for NISQA dataset

"""
import pandas as pd
import os
from speechbrain.dataio.dataset import DynamicItemDataset
import speechbrain as sb
from speechbrain.processing.speech_augmentation import Resample
def get_dataframe(data_root,name):
    
    FILE_FILE_PATH = os.path.join(data_root,name,"%s_file.csv"%name)

    #create an empty dataframe
    data = pd.DataFrame()

    file_data_in = pd.read_csv(FILE_FILE_PATH)

    # combine the db and file columns to create a new column called 'id'
    data['id'] = file_data_in['db'] + "_" + file_data_in['file'].astype(str)
    data['deg_path'] = data_root + "/" + name + "/" + "deg/" + file_data_in['filename_deg']
    #data['ref_path'] = data_root + "/" + name + "/" + "ref/" + file_data_in['filename_ref'] 
    data['mos'] = file_data_in['mos'] /5
    data['noi'] = file_data_in['noi'] /5
    data['col'] = file_data_in['col'] /5
    data['dis'] = file_data_in['dis'] /5
    data['loud'] = file_data_in['loud'] /5
    
    return data

def convert_dataframe_to_dict(in_df):
    """
    convert a dataframe to a dictionary of lists
    """
    out_dict = {}
    for row in in_df.itertuples():
        out_dict[row.id] = {"deg_path":row.deg_path,"mos":row.mos,"noi":row.noi,"col":row.col,"dis":row.dis,"loud":row.loud}
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
    dataset.set_output_keys(["id","deg_audio","mos","noi","col","dis","loud"])

    return dataset


if __name__ =="__main__":
    dataset1= get_dataset("data","NISQA_TRAIN_SIM")
    dataset2= get_dataset("data","NISQA_TRAIN_LIVE")
    print(len(dataset1))
    print(len(dataset2))
    joint_dataset = get_dataset("data",["NISQA_TRAIN_SIM","NISQA_TRAIN_LIVE"])
    print(len(joint_dataset))