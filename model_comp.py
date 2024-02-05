import torchinfo

# model_list = {wavLMEncoderTransformerSmall,\
# wavLMFullTransformerSmall,\
# wavLMFullLayersTransformerSmall,\
# whisperEncoderTransformerSmall,\
# whisperEncoderLayersTransformerSmall,\
# whisperFullTransformerSmall,\
# whisperFullLayersTransformerSmall]
model_list = ["wavLMEncoderTransformerSmall", "wavLMFullTransformerSmall", "wavLMFullLayersTransformerSmall", "whisperEncoderTransformerSmall","whisperEncoderLayersTransformerSmall","whisperFullTransformerSmall","whisperFullLayersTransformerSmall"]
print(model_list)
for m in model_list:
    print(m)

    if m == "wavLMEncoder":
        from models.wavlm_ni_predictors import wavLMMetricPredictorEncoder
        model = wavLMMetricPredictorEncoder().to("cuda")
    elif m == "wavLMFull":
        from models.wavlm_ni_predictors import wavLMMetricPredictorFull
        model = wavLMMetricPredictorFull().to("cuda")
    
    elif m == "wavLMFullLayers":
        from models.wavlm_ni_predictors import wavLMMetricPredictorFullLayers
        model = wavLMMetricPredictorFullLayers().to("cuda")
    elif m == "wavLMASREncoder":
        from models.wavlm_asr_ni_predictors import wavLMASRMetricPredictorEncoder
        model = wavLMASRMetricPredictorEncoder().to("cuda")
    elif m == "wavLMEncoderTransformerSmall":
        from models. wavlm_ni_predictors import wavLMMetricPredictorEncoderTransformerSmall
        model = wavLMMetricPredictorEncoderTransformerSmall().to("cuda")
    elif m == "wavLMFullTransformerSmall":
        from models.wavlm_ni_predictors import wavLMMetricPredictorFullTransformerSmall
        model = wavLMMetricPredictorFullTransformerSmall().to("cuda")
    elif m == "wavLMFullLayersTransformerSmall":
        from models.wavlm_ni_predictors import wavLMMetricPredictorFullLayersTransformerSmall
        model = wavLMMetricPredictorFullLayersTransformerSmall().to("cuda")
    # ---- WHISPER MODELS ----
    elif m == "whisperEncoder":
        from models.whisper_ni_predictors import whisperMetricPredictorEncoder
        model = whisperMetricPredictorEncoder().to("cuda")
    elif m == "whisperEncoderLayers":
        from models.whisper_ni_predictors import whisperMetricPredictorEncoderLayers
        model = whisperMetricPredictorEncoderLayers().to("cuda")
    elif m == "whisperEncoderTransformer":
        from models.whisper_ni_predictors import whisperMetricPredictorEncoderTransformer
        model = whisperMetricPredictorEncoderTransformer().to("cuda")
    elif m == "whisperEncoderTransformerMedium":
        from models.whisper_ni_predictors import whisperMetricPredictorEncoderTransformerMedium
        model = whisperMetricPredictorEncoderTransformerMedium().to("cuda")
    elif m == "whisperEncoderTransformerSmall":
        from models.whisper_ni_predictors import whisperMetricPredictorEncoderTransformerSmall
        model = whisperMetricPredictorEncoderTransformerSmall().to("cuda")
    elif m == "whisperEncoderTransformerSmall2":
        from models.whisper_ni_predictors import whisperMetricPredictorEncoderTransformerSmall2
        model = whisperMetricPredictorEncoderTransformerSmall2().to("cuda")
    elif m == "whisperEncoderTransformerSmall2noSig":
        from models.whisper_ni_predictors import whisperMetricPredictorEncoderTransformerSmall2noSig
        model = whisperMetricPredictorEncoderTransformerSmall2noSig().to("cuda")
    elif m == "whisperEncoderTransformerSmall2T":
        from models.whisper_ni_predictors import whisperMetricPredictorEncoderTransformerSmall2_T
        model = whisperMetricPredictorEncoderTransformerSmall2_T().to("cuda")
    elif m == "whisperEncoderLayersTransformerSmall":
        from models.whisper_ni_predictors import whisperMetricPredictorEncoderLayersTransformerSmall
        model = whisperMetricPredictorEncoderLayersTransformerSmall().to("cuda")
    elif m == "whisperFullLayers":
        from models.whisper_ni_predictors import whisperMetricPredictorFullLayers
        model = whisperMetricPredictorFullLayers().to("cuda")
    elif m == "whisperFullTransformerSmall":
        from models.whisper_ni_predictors import whisperMetricPredictorFullTransformerSmall
        model = whisperMetricPredictorFullTransformerSmall().to("cuda")
    elif m == "whisperFullLayersTransformerSmall":
        from models.whisper_ni_predictors import whisperMetricPredictorFullLayersTransformerSmall
        model = whisperMetricPredictorFullLayersTransformerSmall().to("cuda")
    elif m == "whisperMel":
        from models.whisper_ni_predictors import whisperMetricPredictorMelTransformerSmall
        model = whisperMetricPredictorMelTransformerSmall().to("cuda")
    else:
        raise NotImplementedError("Model %s not implemented"%m)
    #print("--- MODEL SUMMARY ---")
    torchinfo.summary(model,[16,16000*8])
    print("---------------------------------")    