from dataset import get_dataset as get_dataset_nisqa
from dataset_pstn import get_dataset as get_dataset_pstn
from dataset_tencent import get_dataset as get_dataset_tencent
from dataset_iub import get_dataset as get_dataset_iub
from torch.utils.data import ConcatDataset
from NISQA_lib import biasLoss
import pandas as pd
from speechbrain.processing.speech_augmentation import Resample
from speechbrain.dataio.dataset import DynamicItemDataset

import speechbrain as sb
def get_dataset_combined(seed=2712,sets=["tencent","iub","pstn","nisqa"]):
    

    train_dfs = []
    val_dfs = []
    if "tencent" in sets:
        tencent_data,tencent_dict= get_dataset_tencent("tencent_data/",["withoutReverberationTrainDev_16k","withReverberationTrainDev_16k"])
        tencent_df = pd.DataFrame.from_dict(tencent_dict,orient='index')
        tencent_df_val = tencent_df.sample(frac=0.2,random_state=seed)
        tencent_df_train = tencent_df.drop(tencent_df_val.index)
        train_dfs.append(tencent_df_train)
        val_dfs.append(tencent_df_val)
    if "iub" in sets:
        iub_data,iub_dict = get_dataset_iub("iub_data/",["audio_scaled_mos_cosine","audio_scaled_mos_voices"])
        iub_df = pd.DataFrame.from_dict(iub_dict,orient='index')
        iub_df_val = iub_df.sample(frac=0.2,random_state=seed)
        iub_df_train = iub_df.drop(iub_df_val.index)
        train_dfs.append(iub_df_train)
        val_dfs.append(iub_df_val)
    if "pstn" in sets:
        pstn_data,ptsn_dict = get_dataset_pstn("pstn_data/","pstn_train_16k")
        pstn_df = pd.DataFrame.from_dict(ptsn_dict,orient='index')
        pstn_df_val = pstn_df.sample(frac=0.2,random_state=seed)
        pstn_df_train = pstn_df.drop(pstn_df_val.index)
        train_dfs.append(pstn_df_train)
        val_dfs.append(pstn_df_val)
    if "nisqa" in sets:
        nisqa_data,nisqa_dict = get_dataset_nisqa("data",["NISQA_TRAIN_SIM","NISQA_TRAIN_LIVE"])
        nisqa_df_train = pd.DataFrame.from_dict(nisqa_dict,orient='index')
        nisqa_dict_val= get_dataset_nisqa("data",["NISQA_VAL_SIM","NISQA_VAL_LIVE"])[1]
        nisqa_df_val = pd.DataFrame.from_dict(nisqa_dict_val,orient='index')
        train_dfs.append(nisqa_df_train)
        val_dfs.append(nisqa_df_val)    
    #check if there are any datasets which have been requested but not loaded
    if len(train_dfs) != len(sets):
        raise Exception("Not all datasets have been loaded! Check the spelling of the dataset names")
    
    

    #handle the case where there is only one dataset
    if len(sets) == 1:
        #print(type(train_dfs[0]))
        #print(train_dfs[0]) 
        combined_df_train = train_dfs[0]
    else:   
        combined_df_train = pd.concat(train_dfs)

    #shuffle the dataframe
    combined_df_train = combined_df_train.sample(frac=1,random_state=seed)
    #reset the index
    combined_df_train.reset_index(inplace=True)
    #remove the old index column
    combined_df_train.drop(columns=["index"],inplace=True)

    if len(sets) == 1:
        combined_df_val = val_dfs[0]
    else:
        combined_df_val = pd.concat(val_dfs)
    
    #reset the index
    combined_df_val.reset_index(inplace=True)
    #remove the old index column
    combined_df_val.drop(columns=["index"],inplace=True)
    #create a biasLoss object
    bias_loss = biasLoss(combined_df_train["db"])

    combined_dataset_train = get_dataset(combined_df_train)
    combined_dataset_val = get_dataset(combined_df_val)

    return combined_dataset_train,combined_dataset_val,bias_loss,combined_df_train,combined_df_val


def convert_dataframe_to_dict(in_df):
    """
    convert a dataframe to a dictionary of lists
    """
    out_dict = {}
    for row in in_df.itertuples():
        #print(row)
        out_dict[row.Index] = {"deg_path":row.deg_path,"db":row.db,"mos":row.mos}
        #sanity check mos between 0 and 1
        assert row.mos >= 0 and row.mos <= 1
    return out_dict

#resampler_iub = Resample(orig_freq=32000, new_freq=16000)
#resampler_ptsn = Resample(orig_freq=8000, new_freq=16000)
#resampler_tencent32_16 = Resample(orig_freq=32000, new_freq=16000)
#resampler_tencent441_16 = Resample(orig_freq=44100, new_freq=16000)

#resampler_nisqa = Resample(orig_freq=48000, new_freq=16000)
def load_audio(path):
    #print("db",db)
    """load the audio and resample to 16kHz"""
    # if db == "iub":
    #     audio_16k = sb.dataio.dataio.read_audio(path)
    # elif db == "pstn":
    #     #audio_16k = resampler_ptsn(sb.dataio.dataio.read_audio(path).unsqueeze(0)).squeeze(0)
    #     audio_16k = sb.dataio.dataio.read_audio(path)
    # elif db == "tencent":
    #         # sr = sb.dataio.dataio.read_audio_info(path).sample_rate
    #         # print("sr",sr)
    #         # if sr != 16000:
    #         #     if sr == 32000:
    #         #         audio_16k = resampler_tencent32_16(sb.dataio.dataio.read_audio(path).unsqueeze(0)).squeeze(0)
    #         #     elif sr == 48000:
    #         #         audio_16k = resampler_nisqa(sb.dataio.dataio.read_audio(path).unsqueeze(0)).squeeze(0)
    #         #     elif sr == 44100:
    #         #         audio_16k = resampler_tencent441_16(sb.dataio.dataio.read_audio(path).unsqueeze(0)).squeeze(0)
    #         #     else:
    #         #         raise Exception("tencent audio is not 32kHz or 16kHz")
    #         # else:
    #         audio_16k = sb.dataio.dataio.read_audio(path)
    # elif db == "nisqa":
    #     #audio_16k = resampler_nisqa(sb.dataio.dataio.read_audio(path).unsqueeze(0)).squeeze(0)
    #     audio_16k = sb.dataio.dataio.read_audio(path)
    #print("audio_16k",audio_16k.shape  )
    audio_16k = sb.dataio.dataio.read_audio(path)
    return audio_16k


def get_dataset(dataframe):

    data_dict = convert_dataframe_to_dict(dataframe)
    dynamic_items = [
        {"func": lambda l: load_audio(l),
         "takes": "deg_path",
        "provides": "deg_audio"},
    ]
    dataset = DynamicItemDataset(data_dict, dynamic_items)
    dataset.set_output_keys(["id","db","deg_audio","mos"])

    return dataset

if __name__ == "__main__":
    import numpy as np
    import torch
    from speechbrain.dataio.batch import PaddedBatch
    from speechbrain.dataio.dataloader import SaveableDataLoader
    from speechbrain.dataio.dataio import write_audio
    import matplotlib.pyplot as plt
    train_dataset,val_dataset,bias_loss,combined_df,combined_df_val = get_dataset_combined(sets=["tencent","iub","pstn","nisqa"])


    # #boxplot of mos versus dataset
    # plt.figure()
    # combined_df.boxplot(column="mos",by="db")
    # plt.ylabel("Normalized MOS")
    # plt.title("Training Sets")
    # #text of the mean MOS next to the boxplot for each dataset, rounded to 2 decimal places
    # plt.text(1.05,combined_df[combined_df["db"]=="tencent"]["mos"].mean(),combined_df[combined_df["db"]=="tencent"]["mos"].mean().round(2))
    # plt.text(2.05,combined_df[combined_df["db"]=="iub"]["mos"].mean(),combined_df[combined_df["db"]=="iub"]["mos"].mean().round(2))
    # plt.text(3.05,combined_df[combined_df["db"]=="pstn"]["mos"].mean(),combined_df[combined_df["db"]=="pstn"]["mos"].mean().round(2))
    # plt.text(4.05,combined_df[combined_df["db"]=="nisqa"]["mos"].mean(),combined_df[combined_df["db"]=="nisqa"]["mos"].mean().round(2))


    # plt.savefig("combined_dataset_train_box.png")
    # plt.figure()
    # combined_df_val.boxplot(column="mos",by="db")
    # plt.ylabel("Normalized MOS")
    # plt.title("Validation Sets")
    # #text of the mean MOS next to the boxplot for each dataset, rounded to 2 decimal places
    # plt.text(1.05,combined_df_val[combined_df_val["db"]=="tencent"]["mos"].mean(),combined_df_val[combined_df_val["db"]=="tencent"]["mos"].mean().round(2))
    # plt.text(2.05,combined_df_val[combined_df_val["db"]=="iub"]["mos"].mean(),combined_df_val[combined_df_val["db"]=="iub"]["mos"].mean().round(2))
    # plt.text(3.05,combined_df_val[combined_df_val["db"]=="pstn"]["mos"].mean(),combined_df_val[combined_df_val["db"]=="pstn"]["mos"].mean().round(2))
    # plt.text(4.05,combined_df_val[combined_df_val["db"]=="nisqa"]["mos"].mean(),combined_df_val[combined_df_val["db"]=="nisqa"]["mos"].mean().round(2))
    # plt.savefig("combined_dataset_val_box.png")

    # #violinplot of mos versus dataset in order iub, nisqa, ptsn, tencent
    # plt.figure()
    # plt.violinplot(combined_df[combined_df["db"]=="iub"]["mos"],positions=[1],showmeans=True)
    # plt.violinplot(combined_df[combined_df["db"]=="nisqa"]["mos"],positions=[2],showmeans=True)
    # plt.violinplot(combined_df[combined_df["db"]=="pstn"]["mos"],positions=[3],showmeans=True)
    # plt.violinplot(combined_df[combined_df["db"]=="tencent"]["mos"],positions=[4],showmeans=True)
    # #populate x ticks with the dataset names and number of samples
    # plt.xticks([1,2,3,4],["iub: %d" % len(combined_df[combined_df["db"]=="iub"]),\
    #     "nisqa: %d" % len(combined_df[combined_df["db"]=="nisqa"]),\
    #     "pstn: %d" % len(combined_df[combined_df["db"]=="pstn"]),\
    #     "tencent: %d" % len(combined_df[combined_df["db"]=="tencent"])])
    # plt.ylabel("Normalized MOS")
    # plt.title("Training Sets")
    # plt.savefig("combined_dataset_train_violin.png")

    # plt.figure()
    # plt.violinplot(combined_df_val[combined_df_val["db"]=="iub"]["mos"],positions=[1],showmeans=True)
    # plt.violinplot(combined_df_val[combined_df_val["db"]=="nisqa"]["mos"],positions=[2],showmeans=True)
    # plt.violinplot(combined_df_val[combined_df_val["db"]=="pstn"]["mos"],positions=[3],showmeans=True)
    # plt.violinplot(combined_df_val[combined_df_val["db"]=="tencent"]["mos"],positions=[4],showmeans=True)
    # plt.xticks([1,2,3,4],["iub: %d" % len(combined_df_val[combined_df_val["db"]=="iub"]),\
    #     "nisqa: %d" % len(combined_df_val[combined_df_val["db"]=="nisqa"]),\
    #     "pstn: %d" % len(combined_df_val[combined_df_val["db"]=="pstn"]),\
    #     "tencent: %d" % len(combined_df_val[combined_df_val["db"]=="tencent"])])
    
    # plt.ylabel("Normalized MOS")
    # plt.title("Validation Sets")
    # plt.savefig("combined_dataset_val_violin.png")

    # #get mean and variance of each dataset
    # for ds in ["iub","nisqa","pstn","tencent"]:
    #     print("-----",ds,"-----")
    #     print("train")
    #     print(combined_df[combined_df["db"]==ds]["mos"].mean())
    #     print(combined_df[combined_df["db"]==ds]["mos"].var())
    #     print(combined_df[combined_df["db"]==ds]["mos"].max())
    #     print(combined_df[combined_df["db"]==ds]["mos"].min())
    #     print("val")
    #     print(combined_df_val[combined_df_val["db"]==ds]["mos"].mean())
    #     print(combined_df_val[combined_df_val["db"]==ds]["mos"].var())
    #     print(combined_df_val[combined_df_val["db"]==ds]["mos"].max())
    #     print(combined_df_val[combined_df_val["db"]==ds]["mos"].min())
    #     print("-----------------")

    #violinplot of mos versus dataset in order iub, nisqa, ptsn, tencent combined train and val
    combined_df_all = pd.concat([combined_df,combined_df_val])
    
    iub_mos = combined_df_all[combined_df_all["db"]=="iub"]["mos"]
    nisqa_mos = combined_df_all[combined_df_all["db"]=="nisqa"]["mos"]
    pstn_mos = combined_df_all[combined_df_all["db"]=="pstn"]["mos"]
    tencent_mos = combined_df_all[combined_df_all["db"]=="tencent"]["mos"]
    all_mos = combined_df_all["mos"]

    mos_list = [iub_mos,nisqa_mos,pstn_mos,tencent_mos,all_mos]

    plt.figure()
    for i,m in enumerate(mos_list):
        plt.violinplot(m,positions=[i+1],showmeans=True,showextrema=True,showmedians=False,vert=False)

    plt.yticks([1,2,3,4,5],["iub: \n%d" % len(combined_df_all[combined_df_all["db"]=="iub"]),\
        "nisqa: \n%d" % len(combined_df_all[combined_df_all["db"]=="nisqa"]),\
        "pstn: \n%d" % len(combined_df_all[combined_df_all["db"]=="pstn"]),\
        "tencent: \n%d" % len(combined_df_all[combined_df_all["db"]=="tencent"]),\
        "all: \n%d" % len(combined_df_all)])
    plt.xlabel("Normalized MOS")
    plt.title("Dataset MOS Distributions")
    plt.savefig("combined_dataset_all_violin.png")
    plt.savefig("combined_dataset_all_violin.svg")

