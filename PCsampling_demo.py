import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import sampling as sampling
from sampling import ReverseDiffusionPredictor,LangevinCorrector,AnnealedLangevinDynamics ,EulerMaruyamaPredictor,AncestralSamplingPredictor

import aapm_sin_ncsnpp_up as configs_up
import aapm_sin_ncsnpp_down as configs_down
import aapm_sin_ncsnpp_middle as configs_middle

sys.path.append('..')
from losses import get_optimizer
from models.ema import ExponentialMovingAverage

import numpy as np
from utils import restore_checkpoint

import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
from sde_lib import VESDE, VPSDE, subVPSDE
import os.path as osp
if len(sys.argv) > 1:
  start = int(sys.argv[1])
  end = int(sys.argv[2])



checkpoint_num = [[24,24,24]]

def get_predict(num):
  if num == 0:
    return None
  elif num == 1:
    return EulerMaruyamaPredictor
  elif num == 2:
    return ReverseDiffusionPredictor

def get_correct(num):
  if num == 0:
    return None
  elif num == 1:
    return LangevinCorrector
  elif num == 2:
    return AnnealedLangevinDynamics


predicts = [2]
corrects = [1]
for predict in predicts:
  for correct in corrects:
    for check_num in checkpoint_num:
      sde = 'VESDE' #@param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
      if sde.lower() == 'vesde':
        # ckpt_filename_up = './exp50_up/checkpoints/checkpoint_{}.pth'.format(check_num[0])
        # ckpt_filename_down = './exp50_down/checkpoints/checkpoint_{}.pth'.format(check_num[1])
        # ckpt_filename_middle = './exp50_middle/checkpoints/checkpoint_{}.pth'.format(check_num[2])
        ckpt_filename_up = './exp1_up/checkpoints/checkpoint_{}.pth'.format(check_num[0])
        ckpt_filename_down = './exp1_down/checkpoints/checkpoint_{}.pth'.format(check_num[1])
        ckpt_filename_middle = './exp1_middle/checkpoints/checkpoint_{}.pth'.format(check_num[2])
        assert os.path.exists(ckpt_filename_up)
        assert os.path.exists(ckpt_filename_down)
        assert os.path.exists(ckpt_filename_middle)
        config_up = configs_up.get_config()
        config_down = configs_down.get_config()
        config_middle = configs_middle.get_config()
        sde_up = VESDE(sigma_min=config_up.model.sigma_min, sigma_max=config_up.model.sigma_max,N=config_up.model.num_scales)
        sde_down = VESDE(sigma_min=config_down.model.sigma_min, sigma_max=config_down.model.sigma_max,N=config_down.model.num_scales)
        sde_middle = VESDE(sigma_min=config_middle.model.sigma_min, sigma_max=config_middle.model.sigma_max,N=config_middle.model.num_scales)
        sampling_eps = 1e-5

      batch_size = 1  # @param {"type":"integer"}
      config_up.training.batch_size = batch_size
      config_up.eval.batch_size = batch_size
      config_down.training.batch_size = batch_size
      config_down.eval.batch_size = batch_size
      config_middle.training.batch_size = batch_size
      config_middle.eval.batch_size = batch_size

      random_seed = 0 #@param {"type": "integer"}

      # up
      sigmas = mutils.get_sigmas(config_up)
      score_model_up = mutils.create_model(config_up)

      optimizer = get_optimizer(config_up, score_model_up.parameters())
      ema = ExponentialMovingAverage(score_model_up.parameters(),
                                    decay=config_up.model.ema_rate)
      state = dict(step=0, optimizer=optimizer,
                  model=score_model_up, ema=ema)

      state = restore_checkpoint(ckpt_filename_up, state, config_up.device)
      ema.copy_to(score_model_up.parameters())

      # down
      sigmas = mutils.get_sigmas(config_down)
      score_model_down = mutils.create_model(config_down)

      optimizer = get_optimizer(config_down, score_model_down.parameters())
      ema = ExponentialMovingAverage(score_model_down.parameters(),
                                     decay=config_down.model.ema_rate)
      state = dict(step=0, optimizer=optimizer,
                   model=score_model_down, ema=ema)

      state = restore_checkpoint(ckpt_filename_down, state, config_down.device)
      ema.copy_to(score_model_down.parameters())

      # middle
      sigmas = mutils.get_sigmas(config_middle)
      score_model_middle = mutils.create_model(config_middle)

      optimizer = get_optimizer(config_middle, score_model_middle.parameters())
      ema = ExponentialMovingAverage(score_model_middle.parameters(),
                                     decay=config_middle.model.ema_rate)
      state = dict(step=0, optimizer=optimizer,
                   model=score_model_middle, ema=ema)

      state = restore_checkpoint(ckpt_filename_middle, state, config_middle.device)
      ema.copy_to(score_model_middle.parameters())

      predictor = get_predict(predict) #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
      corrector = get_correct(correct) #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}

      snr = 0.0025
      n_steps =  1#@param {"type": "integer"}
      probability_flow = False #@param {"type": "boolean"}
      sampling_fn = sampling.get_pc_sampler(sde_up, sde_down,sde_middle, predictor, corrector,
                                            None, snr, n_steps=n_steps,
                                            probability_flow=probability_flow,
                                            continuous=config_up.training.continuous,
                                            eps=sampling_eps, device=config_up.device)

      sampling_fn(score_model_up,score_model_down,score_model_middle,check_num,predict,correct)

