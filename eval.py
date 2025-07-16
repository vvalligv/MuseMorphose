
import sys, os, time
sys.path.append('./model')

from model.musemorphose import MuseMorphose
from dataloader import REMIFullSongTransformerDataset
from torch.utils.data import DataLoader

from utils import pickle_load
from torch import nn, optim
import torch
import numpy as np

import yaml
config_path = sys.argv[1]
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

device = config['training']['device']

pretrained_params_path = config['model']['pretrained_params_path']
pretrained_optim_path = config['model']['pretrained_optim_path']
max_lr, min_lr = config['training']['max_lr'], config['training']['min_lr']


def validate(model, dloader, n_rounds=8, use_attr_cls=True):
  model.eval()
  loss_rec = []
  kl_loss_rec = []

  print ('[info] validating ...')
  with torch.no_grad():
    for i in range(n_rounds):
      print ('[round {}]'.format(i+1))

      for batch_idx, batch_samples in enumerate(dloader):
        model.zero_grad()

        batch_enc_inp = batch_samples['enc_input'].permute(2, 0, 1).to(device)
        batch_dec_inp = batch_samples['dec_input'].permute(1, 0).to(device)
        batch_dec_tgt = batch_samples['dec_target'].permute(1, 0).to(device)
        batch_inp_bar_pos = batch_samples['bar_pos'].to(device)
        batch_padding_mask = batch_samples['enc_padding_mask'].to(device)
        if use_attr_cls:
          batch_rfreq_cls = batch_samples['rhymfreq_cls'].permute(1, 0).to(device)
          batch_polyph_cls = batch_samples['polyph_cls'].permute(1, 0).to(device)
        else:
          batch_rfreq_cls = None
          batch_polyph_cls = None

        mu, logvar, dec_logits = model(
          batch_enc_inp, batch_dec_inp, 
          batch_inp_bar_pos, batch_rfreq_cls, batch_polyph_cls,
          padding_mask=batch_padding_mask
        )

        losses = model.compute_loss(mu, logvar, 0.0, 0.0, dec_logits, batch_dec_tgt)
        if not (batch_idx + 1) % 10:
          print ('batch #{}:'.format(batch_idx + 1), round(losses['recons_loss'].item(), 3))

        loss_rec.append(losses['recons_loss'].item())
        kl_loss_rec.append(losses['kldiv_raw'].item())
    
  return loss_rec, kl_loss_rec

if __name__ == "__main__":

  dset_val = REMIFullSongTransformerDataset(
    config['data']['data_dir'], config['data']['vocab_path'], 
    do_augment=False, 
    model_enc_seqlen=config['data']['enc_seqlen'], 
    model_dec_seqlen=config['data']['dec_seqlen'], 
    model_max_bars=config['data']['max_bars'],
    pieces=pickle_load(config['data']['val_split']),
    pad_to_same=True
  )

  dloader_val = DataLoader(dset_val, batch_size=config['data']['batch_size'], shuffle=True, num_workers=8)

  mconf = config['model']
  model = MuseMorphose(
    mconf['enc_n_layer'], mconf['enc_n_head'], mconf['enc_d_model'], mconf['enc_d_ff'],
    mconf['dec_n_layer'], mconf['dec_n_head'], mconf['dec_d_model'], mconf['dec_d_ff'],
    mconf['d_latent'], mconf['d_embed'], dset_val.vocab_size,
    d_polyph_emb=mconf['d_polyph_emb'], d_rfreq_emb=mconf['d_rfreq_emb'],
    cond_mode=mconf['cond_mode']
  ).to(device)
  if pretrained_params_path:
    model.load_state_dict( torch.load(pretrained_params_path) )


  opt_params = filter(lambda p: p.requires_grad, model.parameters())
  optimizer = optim.Adam(opt_params, lr=max_lr)
  if pretrained_optim_path:
    optimizer.load_state_dict( torch.load(pretrained_optim_path) )

  vallosses = validate(model, dloader_val)
  with open('valloss.txt', 'a') as f:
    f.write('[val] | RC: {:.4f} | KL: {:.4f}\n'.format(
        np.mean(vallosses[0]),
        np.mean(vallosses[1])
    ))
