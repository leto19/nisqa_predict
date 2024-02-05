import speechbrain as sb
import torch
from dataset import get_dataset
from dataset_combine import get_dataset_combined
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.dataio.dataloader import SaveableDataLoader
import tqdm
import torchinfo
import argparse
import os
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
class earlyStopper(object):
    '''
    Early stopping class. 
    
    Training is stopped if neither RMSE or Pearson's correlation
    is improving after "patience" epochs.
    '''            
    def __init__(self, patience):
        self.best_rmse = 1e10
        self.best_r_p = -1e10
        self.cnt = -1
        self.patience = patience
        self.best = False
        
    def step(self, val_rmse, val_r_p):
        self.best = False
        if val_r_p> self.best_r_p:
            self.best_r_p = val_r_p
            self.cnt = -1   
        if val_rmse< self.best_rmse:
            self.best_rmse = val_rmse
            self.cnt = -1    
            self.best = True
        self.cnt += 1 

        if self.cnt >= self.patience:
            stop_early = True
            return stop_early
        else:
            stop_early = False
            return stop_early
        

def warmup(dataloader,model,optimizer,bias_loss,train_len=117292):
    total_loss = 0
    model.train()

    print("initial lr",optimizer.param_groups[0]['lr'])
    print("number of batches",len(dataloader))
    # set learning rate to the initial learning rate divided by number of batches
    lr_step = optimizer.param_groups[0]['lr']/len(dataloader)
    print("lr_step",lr_step)
    optimizer.param_groups[0]['lr'] = lr_step
    
    print("WARM UP INITIAL LR",optimizer.param_groups[0]['lr'])
    #create tqdm progress bar tracking the training loss
    y_train_hat = np.zeros((train_len, 1))
    #print("y_train_hat",y_train_hat.shape)
    tqdm_dataloader = tqdm.tqdm(dataloader)
    for batch in tqdm_dataloader:
        batch = batch.to("cuda")

        deg_audio,deg_lens = batch.deg_audio
        target_mos = batch.mos.unsqueeze(1).float()
        
        optimizer.zero_grad()

        predict_mos = model(deg_audio).float()
        
        if args.debug:
            print("TRUE MOS",target_mos.T)   
            print("PRED MOS",predict_mos.T)
        #print(predict_mos.shape,target_mos.shape)
        predict_mos_numpy = predict_mos.detach().cpu().numpy()
        batch_id = batch.id.cpu().numpy()
        #print("predict_mos_numpy",predict_mos_numpy.shape)
        #print("batch.id",batch.id.shape)
        #print("y_train_hat[batch.id]",y_train_hat[batch_id].shape)
        y_train_hat[batch_id] = predict_mos_numpy
        loss = bias_loss.get_loss(target_mos,predict_mos,batch.id)
        if args.debug:
            print("LOSS",loss)
        total_loss += loss.item()
        tqdm_dataloader.set_description("loss: %f lr: %f" % (total_loss / (tqdm_dataloader.n + 1),optimizer.param_groups[0]['lr']))
        loss.backward()
        #increase learning rate
        optimizer.param_groups[0]['lr'] += lr_step
        if args.debug:
            print("LR",optimizer.param_groups[0]['lr'])
        optimizer.step()

    return y_train_hat, total_loss / len(dataloader)




def train(dataloader,model,optimizer,bias_loss,train_len=117292):
    total_loss = 0
    model.train()
    #create tqdm progress bar tracking the training loss
    y_train_hat = np.zeros((train_len, 1))
    #print("y_train_hat",y_train_hat.shape)
    tqdm_dataloader = tqdm.tqdm(dataloader)
    for batch in tqdm_dataloader:
        batch = batch.to("cuda")

        deg_audio,deg_lens = batch.deg_audio
        target_mos = batch.mos.unsqueeze(1).float()
        
        optimizer.zero_grad()

        predict_mos = model(deg_audio).float()
        
        if args.debug:
            print("TRUE MOS",target_mos.T)   
            print("PRED MOS",predict_mos.T)
        #print(predict_mos.shape,target_mos.shape)
        predict_mos_numpy = predict_mos.detach().cpu().numpy()
        batch_id = batch.id.cpu().numpy()
        #print("predict_mos_numpy",predict_mos_numpy.shape)
        #print("batch.id",batch.id.shape)
        #print("y_train_hat[batch.id]",y_train_hat[batch_id].shape)
        y_train_hat[batch_id] = predict_mos_numpy
        loss = bias_loss.get_loss(target_mos,predict_mos,batch.id)
        if args.debug:
            print("LOSS",loss)
        total_loss += loss.item()
        tqdm_dataloader.set_description("loss: %f" % (total_loss / (tqdm_dataloader.n + 1)))
        loss.backward()
        optimizer.step()
    return y_train_hat, total_loss / len(dataloader)

def val(dataloader,model,bias_loss):
    with torch.no_grad():
        total_loss = 0
        model.eval()
        #create tqdm progress bar tracking the validation loss
        tqdm_dataloader = tqdm.tqdm(dataloader)
        out_val_list = np.zeros((len(dataloader.dataset), 3))
        for batch in tqdm_dataloader:
            batch = batch.to("cuda")
            deg_audio,_ = batch.deg_audio
            mos = batch.mos.unsqueeze(1).float()
            batch_id = batch.id.cpu().numpy()
            predict_mos = model(deg_audio).float()
            loss = bias_loss._nan_mse(mos,predict_mos)
            predict_mos_numpy = predict_mos.detach().cpu().numpy().squeeze()
            mos_numpy= mos.detach().cpu().numpy().squeeze()
            out_val_list[batch_id, 0] = batch_id
            out_val_list[batch_id, 1] = mos_numpy
            out_val_list[batch_id, 2] = predict_mos_numpy

            total_loss += loss.item()
            
            tqdm_dataloader.set_description("test/val loss: %f"%(loss.item()))          
    return total_loss / len(dataloader), out_val_list

def make_plots(y_train,y_train_hat,combined_df,i):
    for db in combined_df["db"].unique():
        plt.scatter(y_train[combined_df["db"]==db],y_train_hat[combined_df["db"]==db],label=db,alpha=0.4)
    plt.legend()
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel("y_train")
    plt.ylabel("y_train_hat")
    plt.title("train set performance @ epoch %d"%i) 
    plt.savefig(args.save_path + "train_%d.png"%i)
    plt.close()

def main(args):

    #append seed  and model type to save_path
    args.save_path = args.save_path + "_%s_%s_%d/"%(args.model_code_name,args.model,args.seed)
    if args.debug:
        args.save_path = args.save_path.replace("/","_debug/")

    #make sure the save path exists

    if not os.path.exists(args.save_path):
        print("Creating save path %s"%args.save_path)
        os.makedirs(args.save_path)
    #write the hyperparameters to a file in the save path
    with open(args.save_path + "hyperparams.txt","w") as f:
        #iterate over all arguments and write them to the file
        for arg in vars(args):
            f.write("%s: %s\n"%(arg,getattr(args, arg)))
         
    print("loading model")
    torch.manual_seed(args.seed)
    # ---- WavLM MODELS ----
    if args.model == "wavLMEncoderTransformerSmall":
        from models.wavlm_ni_predictors import wavLMMetricPredictorEncoderTransformerSmall
        model = wavLMMetricPredictorEncoderTransformerSmall().to("cuda")
    elif args.model == "wavLMEncoderTransformerSmallT":
        from models.wavlm_ni_predictors import wavLMMetricPredictorEncoderTransformerSmallT
        model = wavLMMetricPredictorEncoderTransformerSmallT().to("cuda")
    elif args.model == "wavLMFullTransformerSmall":
        from models.wavlm_ni_predictors import wavLMMetricPredictorFullTransformerSmall
        model = wavLMMetricPredictorFullTransformerSmall().to("cuda")
    elif args.model == "wavLMFullTransformerSmallT":
        from models.wavlm_ni_predictors import wavLMMetricPredictorFullTransformerSmallT
        model = wavLMMetricPredictorFullTransformerSmallT().to("cuda")
    elif args.model == "wavLMFullLayersTransformerSmall":
        from models.wavlm_ni_predictors import wavLMMetricPredictorFullLayersTransformerSmall
        model = wavLMMetricPredictorFullLayersTransformerSmall().to("cuda")
    elif args.model == "wavLMFullLayersTransformerSmallT":
        from models.wavlm_ni_predictors import wavLMMetricPredictorFullLayersTransformerSmallT
        model = wavLMMetricPredictorFullLayersTransformerSmallT().to("cuda")
    # ---- WHISPER MODELS ----
    elif args.model == "whisperEncoderTransformerSmall":
        from models.whisper_ni_predictors import whisperMetricPredictorEncoderTransformerSmall
        model = whisperMetricPredictorEncoderTransformerSmall().to("cuda")
    elif args.model == "whisperEncoderLayersTransformerSmall":
        from models.whisper_ni_predictors import whisperMetricPredictorEncoderLayersTransformerSmall
        model = whisperMetricPredictorEncoderLayersTransformerSmall().to("cuda")
    elif args.model == "whisperEncoderTransformerSmallT":
        from models.whisper_ni_predictors import whisperMetricPredictorEncoderTransformerSmallT
        model = whisperMetricPredictorEncoderTransformerSmallT().to("cuda")
    elif args.model == "whisperEncoderLayersTransformerSmallT":
        from models.whisper_ni_predictors import whisperMetricPredictorEncoderLayersTransformerSmallT
        model = whisperMetricPredictorEncoderLayersTransformerSmallT().to("cuda")
    elif args.model == "whisperFullTransformerSmall":
        from models.whisper_ni_predictors import whisperMetricPredictorFullTransformerSmall
        model = whisperMetricPredictorFullTransformerSmall().to("cuda")
    elif args.model == "whisperFullTransformerSmallT":
        from models.whisper_ni_predictors import whisperMetricPredictorFullTransformerSmallT
        model = whisperMetricPredictorFullTransformerSmallT().to("cuda")
    elif args.model == "whisperFullLayersTransformerSmall":
        from models.whisper_ni_predictors import whisperMetricPredictorFullLayersTransformerSmall
        model = whisperMetricPredictorFullLayersTransformerSmall().to("cuda")
    elif args.model == "whisperFullLayersTransformerSmallT":
        from models.whisper_ni_predictors import whisperMetricPredictorFullLayersTransformerSmallT
        model = whisperMetricPredictorFullLayersTransformerSmallT().to("cuda")
    elif args.model == "whisperMel":
        from models.whisper_ni_predictors import whisperMetricPredictorMelTransformerSmall
        model = whisperMetricPredictorMelTransformerSmall().to("cuda")
    elif args.model == "whisperMelT":
        from models.whisper_ni_predictors import whisperMetricPredictorMelTransformerSmallT
        model = whisperMetricPredictorMelTransformerSmallT().to("cuda")
    else:
        raise NotImplementedError("Model %s not implemented"%args.model)

    print("--- MODEL INFO ---")
    torchinfo.summary(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                'min',
                verbose=True,
                threshold=0.003,
                patience=15)
    #scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    stopper = earlyStopper(patience=20)
 


    train_dataset,val_dataset,bias_loss,combined_df,combined_df_val = get_dataset_combined(sets=args.datasets.split(","))  


    #acess the first 1000 samples of the dataset (for debugging)
    if args.debug:
        combined_df = combined_df.iloc[:1000]
        combined_df_val = combined_df_val.iloc[:100]
        train_dataset = train_dataset.filtered_sorted(select_n=1000)
        val_dataset = torch.utils.data.Subset(val_dataset,range(100))


    print("--- CPU INFO ---")
    #get the number of available CPUs
    num_cpus = args.num_cpus
    print("recommended num_cpus",len(os.sched_getaffinity(0)))
    print("num_cpus",num_cpus)
    y_train = combined_df["mos"].to_numpy()

    train_dataloader = SaveableDataLoader(train_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        collate_fn=PaddedBatch,
                                        num_workers=num_cpus,
                                        )
    
    val_dataloader = SaveableDataLoader(val_dataset,
                                        batch_size=args.batch_size//2,
                                        shuffle=False,
                                        collate_fn=PaddedBatch,
                                        num_workers=num_cpus,
                                   )
    print("------ DATA INFO ------")
    print("trainset has %d elements (%s batches)"%(len(train_dataset),len(train_dataloader)))
    print("combined_df has %d elements"%len(combined_df)    )
    print("valset has %d elements (%s batches)"%(len(val_dataset),len(val_dataloader)))
    
    #if the model has 'transformer' in the name, do some warmup epochs  
    with open(args.save_path + "stats.csv","a") as f:
        f.write("epoch,train_loss,val_loss,train_pcc,val_pcc\n")
    
    
    #warmup
    print("------ WARMUP ------")
    warmup(train_dataloader,model,optimizer,bias_loss,train_len=len(combined_df))

    print("LEARNING RATE:",optimizer.param_groups[0]['lr'])
    print("------ TRAINING ------")
    val_pccs = []
    val_losses = []

    
  
   

    for i in range(args.n_epochs):
        
        #if args.debug:
        #    y_train_hat = train(train_dataloader,model,optimizer,bias_loss,train_len=1000)
        #else:
        y_train_hat,train_loss = train(train_dataloader,model,optimizer,bias_loss,train_len=len(combined_df))

        #print(y_train.shape,y_train_hat[:,0].shape)
        #print(y_train,y_train_hat)
        train_pcc = scipy.stats.pearsonr(y_train,y_train_hat[:,0])[0]
        #update bias
        if args.debug or args.plot:
            print(len(y_train),len(y_train_hat))
            if len(y_train) != len(y_train_hat):
                print("ERROR: y_train and y_train_hat have different lengths")
            else:
                make_plots(y_train,y_train_hat,combined_df,i)
        
        if not args.debug:
            bias_loss.update_bias(y_train,y_train_hat)

        val_loss,val_list = val(val_dataloader,model,bias_loss)
        val_id = [x[0] for x in val_list]
        val_true = [x[1] for x in val_list]
        val_true = np.array(val_true)
        val_pred = [x[2] for x in val_list]
        val_pred = np.array(val_pred)
        val_pcc = scipy.stats.pearsonr(val_true,val_pred)[0]
        if args.debug or args.plot:
            for db in combined_df_val["db"].unique():
                #print(db)
                #print(len(val_true[combined_df_val["db"]==db]))
                #print(len(val_pred[combined_df_val["db"]==db]))
                plt.scatter(val_true[combined_df_val["db"]==db],val_pred[combined_df_val["db"]==db],label=db,alpha=0.4)
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.legend()
            plt.xlabel("true")
            plt.ylabel("pred")
            plt.title("val set performance @ epoch %d"%i)
            plt.savefig(args.save_path + "val_%d.png"%i)
            plt.close()
        val_losses.append(val_loss)
        val_pccs.append(val_pcc)
        scheduler.step(val_loss)
        #scheduler.step()
        print("EPOCH %d VAL LOSS %f PCC %f"%(i,val_loss,val_pcc))
        #save model
        torch.save(model.state_dict(), args.save_path + "model_%d_%s.pt"%(i,val_loss))
        # write the epoch stats to a file
        with open(args.save_path + "stats.csv","a") as f:
            f.write("%d,%f,%f,%f,%f\n"%(i,train_loss,val_loss,train_pcc,val_pcc))
        if stopper.step(val_loss,val_pcc):
            print("EARLY STOPPING AT EPOCH %d"%i)
            break
    #find the checkpoint with the lowest validation loss
    # best_checkpoint = None
    # best_val_loss = 1000
    # for checkpoint in os.listdir(args.save_path):
    #     if checkpoint.endswith(".pt"):
    #         val_loss = float(checkpoint.split("_")[-1].replace(".pt",""))
    #         if val_loss < best_val_loss:
    #             best_val_loss = val_loss
    #             best_checkpoint = checkpoint
    # print("BEST CHECKPOINT %s"%best_checkpoint)
    # model.load_state_dict(torch.load(args.save_path + best_checkpoint))


    print("------ TESTING ------")
    test_dataset,_ = get_dataset("data",["NISQA_TEST_P501","NISQA_TEST_FOR","NISQA_TEST_LIVETALK"])
    test_dataloader = SaveableDataLoader(test_dataset, batch_size=args.batch_size//2, shuffle=False, collate_fn=PaddedBatch)
    print("testset has %d elements (%s batches)"%(len(test_dataset),len(test_dataloader)))
    test_loss,test_list = val(test_dataloader,model,bias_loss)
    test_pcc = scipy.stats.pearsonr([x[1] for x in test_list],[x[2] for x in test_list])[0]
    print("TEST LOSS %f PCC %f"%(test_loss,test_pcc))
    print("------ DONE ------")


if __name__ == "__main__":
    import datetime
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2712)
    parser.add_argument("--model_code_name", type=str, default="")
    parser.add_argument("--save_path", type=str, default="results_combined_bias/%s"%datetime.datetime.now().strftime("%Y%m%d/%H%M"))
    parser.add_argument("--debug", action="store_true",default=False)
    parser.add_argument("--model", type=str, default="wavLMEncoder")
    parser.add_argument("--num_cpus", type=int, default=len(os.sched_getaffinity(0)))
    parser.add_argument("--plot", action="store_true",default=True)
    parser.add_argument("--datasets", type=str, default="tencent,pstn,nisqa,iub")
    args = parser.parse_args()
    if args.debug:
        print("!!!DEBUG MODE!!!")
    main(args)