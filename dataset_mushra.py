"""
handles mapping between csv files and torch datasets for NISQA dataset

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
    #delete the last column
    #remove the rows where file is 'training', 'anchor70' or 'anchor35'
    file_data_in = file_data_in[file_data_in.file != 'training']
    file_data_in = file_data_in[file_data_in.condition != 'anchor70']
    file_data_in = file_data_in[file_data_in.condition != 'anchor35']
    print(file_data_in.head())

    #replace all instances of 'reference; in row 'condition' with 'clean'
    file_data_in['condition'] = file_data_in['condition'].replace(['reference'],'clean')
    #replace all instances of 'degraded' in row 'condition' with 'noisy'
    file_data_in['condition'] = file_data_in['condition'].replace(['degraded'],'noisy')
    #replace all instances of 'CMGAN' in row 'condition' with 'CMGAN_enhanced'
    file_data_in['condition'] = file_data_in['condition'].replace(['CMGAN'],'CMGAN_enhanced')
    #replace all instances of 'DFT-FSNET' in row 'condition' with 'DFT-FSNET_enhanced'
    file_data_in['condition'] = file_data_in['condition'].replace(['DFT-FSNET'],'dptfsnet_enhanced')
    #replace all instances of 'py_nr' in row 'condition' with 'pyNR_enhanced'
    file_data_in['condition'] = file_data_in['condition'].replace(['py_nr'],'pyNR_enhanced')
    #replace all instances of 'sgmse-bbed' in row 'condition' with 'sgmse_bbed_enhanced'
    file_data_in['condition'] = file_data_in['condition'].replace(['sgmse-bbed'],'sgmse_bbed_enhanced')

    data['id'] = file_data_in['file'] + "_" + file_data_in['condition'].astype(str)
    data['mos'] = file_data_in['score'] /100
    data['deg_path'] = data_root + "/" + file_data_in['condition'] + "/" + file_data_in['file'] + ".wav"
    




    print(data)
    return data

def convert_dataframe_to_dict(in_df):
    """
    convert a dataframe to a dictionary of lists
    """
    out_dict = {}
    for row in in_df.itertuples():
        out_dict[row.id] = {"deg_path":row.deg_path,"mos":row.mos}
    return out_dict

#resampler = Resample(orig_freq=48000, new_freq=16000)


def load_audio(path):
    """load the audio and downsample to 16kHz"""
    
    audio_16k = sb.dataio.dataio.read_audio(path).squeeze(0)
    
    #print(audio_16k.shape)

    #get only the first channel
    audio_16k = audio_16k[:,0]
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
    #dataset.set_output_keys(["id","deg_audio","ref_audio","mos"])
    dataset.set_output_keys(["id","deg_audio","mos"])

    return dataset


if __name__ =="__main__":
    dataset1= get_dataset("cv-d-listening_tests_stereo","mushra")
    dataloader = sb.dataio.dataloader.make_dataloader(dataset1, batch_size=1)
    for batch in dataloader:
        print(batch.id)
        print(batch.deg_audio)
        print(batch.mos)
        break