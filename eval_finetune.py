import torch
import torch.nn as nn
import torchaudio
from torchaudio import transforms as T
from models import ResNet
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from util import finetune_proto, get_distance, finetune_ce
import glob
import os
from augmentations import RandomCrop, Resize, GaussNoise, Compander, FreqShift
from evaluation import evaluate
from args import args
from utils_eval import evaluate_with_ce, evaluate_with_protoloss


# General params
val_dir = args.valdir
out_dir = os.path.join(args.valdir, '../outputs')
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, f'{args.csv_file}.csv')
ckpt_dir = os.path.join(args.valdir, '../model/')
TARGET_SR = args.sr
N_MELS = args.nmels
HOP_MEL = args.hoplen
N_SHOT = args.nshot
BATCH_SIZE = args.qbs
fps = TARGET_SR/HOP_MEL
emb_dim = 2048 # dimension of the latent space, check architecture in models.py

# Spectrogram
mel = T.MelSpectrogram(sample_rate=args.sr, n_fft=args.nfft, hop_length=args.hoplen, f_min=args.fmin, f_max=args.fmax, n_mels=args.nmels)
power_to_db = T.AmplitudeToDB()
transform = nn.Sequential(mel, power_to_db)

name_arr = np.array([])
onset_arr = np.array([])
offset_arr = np.array([])
filenames = [file for file in glob.glob(os.path.join(val_dir,'*/*.csv'))]
for filename in filenames:
    print(f"processing file {filename}")
    feat_name = filename.split('/')[-1]
    audio_name = feat_name.replace('csv','wav')
    csv_file = os.path.join(val_dir, filename)
    wav_file = csv_file.replace('csv','wav')
    df = pd.read_csv(csv_file)
    df.loc[:,'Starttime'] = df['Starttime']
    df.loc[:,'Endtime'] = df['Endtime']
    Q_list = df['Q'].to_numpy()
    index_sup = np.where(Q_list == 'POS')[0][:N_SHOT]
    start_time = [int(np.floor(start * fps)) for start in df['Starttime']]
    end_time = [int(np.floor(end * fps)) for end in df['Endtime']]
    difference = []
    for index in index_sup:
        difference.append(end_time[index] - start_time[index])

    # changed this to the 2022 implementation of using Adaptive segment length
    # tmax_frames = int(np.max(difference))
    # tmax_sec = tmax_frames / fps

    # if tmax_sec <= 0.1:
    #     win_len_sec = 8 
    # elif tmax_sec <= 0.4:
    #     win_len_sec = tmax_sec
    # elif tmax_sec <= 0.8:
    #     win_len_sec = tmax_sec / 2
    # elif tmax_sec <= 3.0:
    #     win_len_sec = tmax_sec / 4
    # else:
    #     win_len_sec = tmax_sec / 8

    # win_len = int(round(win_len_sec * fps))
    # seg_hop = win_len // 3

    max_len = int(round(np.mean(difference)))

    if max_len <= 17:
        win_len = 17
    elif max_len > 17 and max_len <= 100 :
        win_len = max_len
    elif max_len > 100 and max_len <= 200 :
        win_len = max_len//2
    elif max_len > 200 and max_len <= 400 :
        win_len = max_len//4
    else:
        win_len = max_len//8
    seg_hop = win_len//2

    wav, sr = torchaudio.load(wav_file)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    resample = T.Resample(sr, TARGET_SR)
    wav = resample(wav)
    melspec = transform(wav)

    features_pos = []
    features_neg = []

    for index in range(len(index_sup)):
        # pos features
        start_idx = start_time[index_sup[index]] 
        if end_time[index_sup[index]] - start_idx > win_len:
            while end_time[index_sup[index]] - start_idx > win_len:
                spec = melspec[...,start_idx:start_idx+win_len]
                start_idx += seg_hop
                features_pos.append(spec)  
            if end_time[index_sup[index]] - start_idx > win_len//2:
                spec = melspec[...,start_idx:end_time[index_sup[index]]]
                repeat_num = int(win_len / (spec.shape[-1])) + 1
                spec = spec.repeat(1,1,repeat_num)
                spec = spec[...,:int(win_len)]
                features_pos.append(spec)  
        else:
            if end_time[index_sup[index]] - start_idx > 0:
                spec = melspec[...,start_idx:end_time[index_sup[index]]]
                repeat_num = int(win_len / (spec.shape[-1])) + 1
                spec = spec.repeat(1,1,repeat_num)
                spec = spec[...,:int(win_len)]
                features_pos.append(spec)  

        # neg features
        start_idx = end_time[index_sup[index]]
        if index < len(index_sup)-1:
            if start_time[index_sup[index+1]] - start_idx > win_len:
                while start_time[index_sup[index+1]] - start_idx > win_len:
                    spec = melspec[...,start_idx:start_idx+win_len]
                    start_idx += seg_hop
                    features_neg.append(spec)
                if start_time[index_sup[index+1]] - start_idx > win_len//2:
                    spec = melspec[...,start_idx:start_time[index_sup[index+1]]]
                    repeat_num = int(win_len / (spec.shape[-1])) + 1
                    spec = spec.repeat(1,1,repeat_num)
                    spec = spec[...,:int(win_len)]
                    features_neg.append(spec)
            else:
                if start_time[index_sup[index+1]] - start_idx > 0:
                    spec = melspec[...,start_idx:start_time[index_sup[index+1]]]
                    repeat_num = int(win_len / (spec.shape[-1])) + 1
                    spec = spec.repeat(1,1,repeat_num)
                    spec = spec[...,:int(win_len)]
                    features_neg.append(spec)   
        
        # add event before first pos to negatives
        end_t0 = start_time[index_sup[0]] 
        start_t0 = 0
        curr_t0 = start_t0
        if end_t0 - curr_t0 > win_len : 
            while end_t0 - curr_t0 > win_len :
                spec = melspec[...,curr_t0:curr_t0+win_len]
                curr_t0 += seg_hop
                features_neg.append(spec)
            if end_t0 - curr_t0 > win_len//2 :
                spec = melspec[...,curr_t0:curr_t0+win_len]
                repeat_num = int(win_len / (spec.shape[-1])) + 1
                spec = spec.repeat(1,1,repeat_num)
                spec = spec[...,:int(win_len)]
                features_neg.append(spec)
        else:
            if end_t0 - curr_t0 > 0 :
                spec = melspec[...,curr_t0:curr_t0+win_len]
                repeat_num = int(win_len / (spec.shape[-1])) + 1
                spec = spec.repeat(1,1,repeat_num)
                spec = spec[...,:int(win_len)]
                features_neg.append(spec)

    features_pos = torch.stack(features_pos)
    features_neg = torch.stack(features_neg)
    nb_positives = features_pos.shape[0]
    nb_negatives = features_neg.shape[0]

    # query features
    labels_q = []
    features_q = []

    last_frame = melspec.shape[-1]
    curr_frame = end_time[index_sup[-1]]
    if last_frame - curr_frame > win_len:
        while last_frame - curr_frame > win_len:
            spec = melspec[...,curr_frame:curr_frame+win_len]
            features_q.append(spec)
            curr_frame += seg_hop
        if last_frame - curr_frame > win_len//2:
            spec = melspec[...,curr_frame:last_frame]
            repeat_num = int(win_len / (spec.shape[-1])) + 1
            spec = spec.repeat(1,1,repeat_num)
            spec = spec[...,:int(win_len)]
            features_q.append(spec)   
    else:
        if last_frame - curr_frame > 0: #win_len//2
            spec = melspec[...,curr_frame:last_frame]
            repeat_num = int(win_len / (spec.shape[-1])) + 1
            spec = spec.repeat(1,1,repeat_num)
            spec = spec[...,:int(win_len)]
            features_q.append(spec)

    features_q = torch.stack(features_q)

    # Light DA to create different views
    rc = RandomCrop(n_mels=128, time_steps=features_q.shape[-1], tcrop_ratio=0.9)
    resize = Resize(n_mels=128, time_steps=features_q.shape[-1])
    comp = Compander(comp_alpha=0.9)
    makeview = nn.Sequential(rc, resize, comp)

    # Loading model
    if args.ftmethod == 'ce':
        encoder = ResNet(method='ce', num_classes=2)
    else:
        encoder = ResNet()

    ckpt = torch.load(os.path.join(ckpt_dir, f'{args.model_name}.pth'), map_location=torch.device('cpu'))
    encoder.load_state_dict(ckpt['encoder'], strict=False)
    encoder = encoder.to(args.device)

    # Preparing loader
    ds_pos = TensorDataset(features_pos)
    ds_neg = TensorDataset(features_neg)
    ds_q = TensorDataset(features_q)

    loader_pos = DataLoader(ds_pos, batch_size=BATCH_SIZE)
    loader_neg = DataLoader(ds_neg, batch_size=BATCH_SIZE)
    loader_q = DataLoader(ds_q, batch_size=BATCH_SIZE)

    # Finetuning
    data_finetune = torch.cat([features_pos, features_neg], dim=0)
    label_finetune = torch.cat([torch.ones(features_pos.shape[0], dtype=torch.long), torch.zeros(features_neg.shape[0], dtype=torch.long)], dim=0)
    bs_finetune = args.ftbs
    ds_finetune = TensorDataset(data_finetune, label_finetune)
    dl_finetune = DataLoader(ds_finetune, batch_size=bs_finetune, shuffle=True)

    if args.ftmethod == 'ce':
        encoder = finetune_ce(encoder, dl_finetune, makeview, args)
        labels_pred = evaluate_with_ce(loader_q, encoder, args)
    else:
        encoder = finetune_proto(encoder, dl_finetune, makeview, args)
        labels_pred = evaluate_with_protoloss(encoder, loader_pos, loader_neg, loader_q, makeview, args, emb_dim)

    # create output csv with onset/offset times
    labels_comb = []
    labels_comb.append(labels_pred)

    labels_comb = torch.tensor(labels_comb)

    labs_pred, _ = torch.mode(labels_comb, axis=0)

    labs_pred = np.array(labs_pred)

    krn = np.array([1, -1])

    changes = np.convolve(krn, labs_pred)

    onset_frames = np.where(changes == 1)[0]
    offset_frames = np.where(changes == -1)[0]

    str_time_query = end_time[index_sup[-1]] * HOP_MEL / TARGET_SR

    onset = (onset_frames ) * (seg_hop) * HOP_MEL / TARGET_SR
    onset = onset + str_time_query

    offset = (offset_frames ) * (seg_hop) * HOP_MEL / TARGET_SR
    offset = offset + str_time_query

    assert len(onset) == len(offset)
    name = np.repeat(audio_name,len(onset))
    name_arr = np.append(name_arr,name)
    onset_arr = np.append(onset_arr,onset)
    offset_arr = np.append(offset_arr,offset)

df_out = pd.DataFrame({'Audiofilename': name_arr,'Starttime': onset_arr,'Endtime': offset_arr})
df_out.to_csv(csv_path, index=False)

