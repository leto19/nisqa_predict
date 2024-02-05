import speechbrain as sb
import torch
from dataset import get_dataset
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.dataio.dataloader import SaveableDataLoader
import tqdm
import torchinfo
import argparse
import os
import scipy.stats
import numpy as np
def val(dataloader,model):
    total_loss = 0
    total_pcc = 0
    out_list = []
    model.eval()
    #create tqdm progress bar tracking the validation loss
    tqdm_dataloader = tqdm.tqdm(dataloader)
  
    for batch in tqdm_dataloader:
        batch = batch
        #print(batch.id)
        deg_audio,deg_lens = batch.deg_audio
        mos = batch.mos.unsqueeze(1).float()

        predict_mos = model(deg_audio).float()
        loss = torch.nn.functional.mse_loss(predict_mos,mos)
        total_loss += loss.item()
        for i in range(len(batch.id)):
            out_list.append((batch.id[i],predict_mos[i].item(),mos[i].item()))
        tqdm_dataloader.set_description("test loss: %f"%(loss.item()))
    
    return total_loss / len(dataloader),out_list

def main(args):
    #if model is not given, figure out from the load_path
    if args.model is None:
        args.model = args.load_path.split("/")[-1].split("_")[-2]
        print("model not given, assuming %s"%args.model)
    print("loading model %s"%args.model)

        # ---- WavLM MODELS ----
    if args.model == "wavLMEncoderTransformerSmall":
        from models.wavlm_ni_predictors import wavLMMetricPredictorEncoderTransformerSmall
        model = wavLMMetricPredictorEncoderTransformerSmall()
    elif args.model == "wavLMEncoderTransformerSmallT":
        from models.wavlm_ni_predictors import wavLMMetricPredictorEncoderTransformerSmallT
        model = wavLMMetricPredictorEncoderTransformerSmallT()
    elif args.model == "wavLMFullTransformerSmall":
        from models.wavlm_ni_predictors import wavLMMetricPredictorFullTransformerSmall
        model = wavLMMetricPredictorFullTransformerSmall()
    elif args.model == "wavLMFullTransformerSmallT":
        from models.wavlm_ni_predictors import wavLMMetricPredictorFullTransformerSmallT
        model = wavLMMetricPredictorFullTransformerSmallT()
    elif args.model == "wavLMFullLayersTransformerSmall":
        from models.wavlm_ni_predictors import wavLMMetricPredictorFullLayersTransformerSmall
        model = wavLMMetricPredictorFullLayersTransformerSmall()
    elif args.model == "wavLMFullLayersTransformerSmallT":
        from models.wavlm_ni_predictors import wavLMMetricPredictorFullLayersTransformerSmallT
        model = wavLMMetricPredictorFullLayersTransformerSmallT()
    # ---- WHISPER MODELS ----
    elif args.model == "whisperEncoderTransformerSmall":
        from models.whisper_ni_predictors import whisperMetricPredictorEncoderTransformerSmall
        model = whisperMetricPredictorEncoderTransformerSmall()
    elif args.model == "whisperEncoderLayersTransformerSmall":
        from models.whisper_ni_predictors import whisperMetricPredictorEncoderLayersTransformerSmall
        model = whisperMetricPredictorEncoderLayersTransformerSmall()
    elif args.model == "whisperEncoderTransformerSmallT":
        from models.whisper_ni_predictors import whisperMetricPredictorEncoderTransformerSmallT
        model = whisperMetricPredictorEncoderTransformerSmallT()
    elif args.model == "whisperEncoderLayersTransformerSmallT":
        from models.whisper_ni_predictors import whisperMetricPredictorEncoderLayersTransformerSmallT
        model = whisperMetricPredictorEncoderLayersTransformerSmallT()
    elif args.model == "whisperFullTransformerSmall":
        from models.whisper_ni_predictors import whisperMetricPredictorFullTransformerSmall
        model = whisperMetricPredictorFullTransformerSmall()
    elif args.model == "whisperFullTransformerSmallT":
        from models.whisper_ni_predictors import whisperMetricPredictorFullTransformerSmallT
        model = whisperMetricPredictorFullTransformerSmallT()
    elif args.model == "whisperFullLayersTransformerSmall":
        from models.whisper_ni_predictors import whisperMetricPredictorFullLayersTransformerSmall
        model = whisperMetricPredictorFullLayersTransformerSmall()
    elif args.model == "whisperFullLayersTransformerSmallT":
        from models.whisper_ni_predictors import whisperMetricPredictorFullLayersTransformerSmallT
        model = whisperMetricPredictorFullLayersTransformerSmallT()
    elif args.model == "whisperMel":
        from models.whisper_ni_predictors import whisperMetricPredictorMelTransformerSmall
        model = whisperMetricPredictorMelTransformerSmall()
    elif args.model == "whisperMelT":
        from models.whisper_ni_predictors import whisperMetricPredictorMelTransformerSmallT
        model = whisperMetricPredictorMelTransformerSmallT()
    else:
        raise NotImplementedError("Model %s not implemented"%args.model)
    print("--- MODEL SUMMARY ---")
    torchinfo.summary(model)

    if args.load_mode == "recent":
    #find the most recent model in the directory
        if os.path.isdir(args.load_path):
            dir_list = os.listdir(args.load_path)
            #find the most recently created model using os.path.getctime
            #get only .pt files
            dir_list = [x for x in dir_list if x.endswith(".pt")]
            most_recent_path = os.path.join(args.load_path,max(dir_list,key=lambda x: os.path.getctime(os.path.join(args.load_path,x))))
            best_checkpoint = most_recent_path.split("/")[-1]
            print("loading most recent model %s"%most_recent_path)
            model.load_state_dict(torch.load(most_recent_path))
        
        else:
            model.load_state_dict(torch.load(args.load_path))
    
    else:
        #find the checkpoint with the lowest validation loss
        best_checkpoint = None
        best_val_loss = 1000
        for checkpoint in os.listdir(args.load_path):
            if checkpoint.endswith(".pt"):
                val_loss = float(checkpoint.split("_")[-1].replace(".pt",""))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_checkpoint = checkpoint
        print("BEST CHECKPOINT %s"%best_checkpoint)
        model.load_state_dict(torch.load(args.load_path +"/" + best_checkpoint,map_location=torch.device("cpu")),strict=False)

    print("------ TESTING ------")
    test_dataset,_ = get_dataset("data",args.testset)
    print("TESTSET: %s N_SAMPLES: %d"%(args.testset,len(test_dataset)))
    test_dataloader = SaveableDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=PaddedBatch,num_workers=args.n_jobs)
    test_loss,out_list = val(test_dataloader,model)
    print("test loss: %f"%test_loss)

    if os.path.isdir(args.load_path):
        csv_write_path = "%s/%s_%s_%s.csv"%(args.load_path,args.model,best_checkpoint.split("/")[-1].strip(".pt "),args.testset)
        print("writing results to %s"%csv_write_path)
        with open(csv_write_path,"w") as f:
            f.write("id,predicted_mos,mos\n")
            for id,predicted_mos,mos in out_list:
                f.write("%s,%f,%f\n"%(id,predicted_mos*5,mos*5))

    else:
        with open("results_nisqa_only/%s_%s_%s.csv"%(args.model,args.load_path.split("/")[-1].strip(".pt "),args.testset),"w") as f:
            f.write("id,predicted_mos,mos\n")
            for id,predicted_mos,mos in out_list:
                f.write("%s,%f,%f\n"%(id,predicted_mos*5,mos*5))
if __name__ == "__main__":
    import datetime
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--load_path", type=str, default="results/%s"%datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),required=True)
    parser.add_argument("--debug", action="store_true",default=False)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--testset", type=str, default="NISQA_TEST_P501")
    parser.add_argument("--load_mode", type=str, default="best")
    parser.add_argument("--n_jobs", type=int, default=len(os.sched_getaffinity(0)))
    args = parser.parse_args()
    if args.debug:
        print("!!!DEBUG MODE!!!")
    main(args)