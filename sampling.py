import datetime
import os
from random import betavariate
import sys

import math
import pandas as pd
import scipy.io as sio
sys.path.append('..')
import functools
import matplotlib.pyplot as plt
import torch
import skimage
from skimage import io, img_as_float
import numpy as np
import abc
from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
import sde_lib
from models import utils as mutils
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim,mean_squared_error as compare_mse
import odl
import glob
import pydicom
from cv2 import imwrite, resize
from func_test import WriteInfo
from scipy.io import loadmat, savemat
from radon_utils import (create_sinogram, bp, filter_op,
                         fbp, reade_ima, write_img, sinogram_2c_to_img,
                         padding_img, unpadding_img, indicate)
from time import sleep
import cv2
import odl
from tvdenoise import tvdenoise

import scipy.io


Fan_angle_partition = odl.uniform_partition(0, 2 * np.pi, 720)
Fan_detector_partition = odl.uniform_partition(-360, 360, 720)
Fan_geometry = odl.tomo.FanBeamGeometry(Fan_angle_partition, Fan_detector_partition,src_radius=500, det_radius=500)
Fan_reco_space = odl.uniform_discr(min_pt=[-128, -128], max_pt=[128, 128], shape=[512, 512], dtype='float32')

reco_space = odl.uniform_discr(min_pt=[-128, -128], max_pt=[128, 128], shape=[720, 720], dtype='float32')
Fan_ray_trafo = odl.tomo.RayTransform(Fan_reco_space, Fan_geometry)

Fan_ray_trafobp = odl.tomo.RayTransform(reco_space, Fan_geometry)
Fan_FBP = odl.tomo.fbp_op(Fan_ray_trafo)
Fan_filter = odl.tomo.fbp_filter_op(Fan_ray_trafo)


A = (Fan_ray_trafo(Fan_ray_trafo.domain.one())).data
ATA = Fan_ray_trafo.adjoint(Fan_ray_trafo(Fan_ray_trafo.domain.one()))
AAT = Fan_ray_trafo(Fan_ray_trafo.adjoint(Fan_ray_trafobp.domain.one()))


pseudoinverse = odl.tomo.fbp_op(Fan_ray_trafo)

_CORRECTORS = {}
_PREDICTORS = {}


def set_predict(num):
    if num == 0:
        return 'None'
    elif num == 1:
        return 'EulerMaruyamaPredictor'
    elif num == 2:
        return 'ReverseDiffusionPredictor'

def set_correct(num):
    if num == 0:
        return 'None'
    elif num == 1:
        return 'LangevinCorrector'
    elif num == 2:
        return 'AnnealedLangevinDynamics'

def padding_img(img):
    w, h = 720, 720
    h1 = 768
    tmp = np.zeros([h1, h1])
    x_start = int((h1 - w) // 2)
    y_start = int((h1 - h) // 2)
    tmp[x_start:x_start + w, y_start:y_start + h] = img
    return tmp


def unpadding_img(img):
    w, h = 720, 720
    h1 = 768
    x_start = int((h1 - w) // 2)
    y_start = int((h1 - h) // 2)
    return img[x_start:x_start + w, y_start:y_start + h]



photons = 1e5

def init_ct_op(img):
    global photons
    photons_per_pixel = photons
    mu_water = 0.02
    epsilon = 0.0001
    nonlinear_operator = odl.ufunc_ops.exp(Fan_ray_trafo.range) * (-mu_water * Fan_ray_trafo)
    phantom = Fan_reco_space.element(img)


    proj_trans = Fan_ray_trafo(phantom)

    proj_data = nonlinear_operator(phantom)
    proj_ideal = np.copy(proj_trans)
    proj_data = odl.phantom.poisson_noise(proj_data * photons_per_pixel) / photons_per_pixel
    proj_data = -np.log(epsilon + proj_data) / mu_water


    sinogram_input = proj_data
    sinogram_input = sinogram_input.asarray()

    img_ldct = Fan_FBP(sinogram_input)

    return sinogram_input


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


def get_corrector(name):
    return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps):
    """Create a sampling function.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.
sparse
  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """

    sampler_name = config.sampling.method  # pc

    if sampler_name.lower() == 'ode':
        sampling_fn = get_ode_sampler(sde=sde,
                                      shape=shape,
                                      inverse_scaler=inverse_scaler,
                                      denoise=config.sampling.noise_removal,
                                      eps=eps,
                                      device=config.device)

    elif sampler_name.lower() == 'pc':
        predictor = get_predictor(config.sampling.predictor.lower())
        corrector = get_corrector(config.sampling.corrector.lower())
        sampling_fn = get_pc_sampler(sde=sde,
                                     shape=shape,
                                     predictor=predictor,
                                     corrector=corrector,
                                     inverse_scaler=inverse_scaler,
                                     snr=config.sampling.snr,
                                     n_steps=config.sampling.n_steps_each,
                                     probability_flow=config.sampling.probability_flow,
                                     continuous=config.training.continuous,
                                     denoise=config.sampling.noise_removal,
                                     eps=eps,
                                     device=config.device)
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")

    return sampling_fn


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the corrector.

    Args:
      x: A PyTorch tensor representing the current state
      t: A PyTorch tensor representing the current time step.x0

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
        pass


@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        dt = -1. / self.rsde.N
        z = torch.randn_like(x)
        drift, diffusion = self.rsde.sde(x, t)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean


# ===================================================================== ReverseDiffusionPredictor
@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        f, G = self.rsde.discretize(x, t)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + G[:, None, None, None] * z
        return x, x_mean


# =====================================================================

@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
    """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)
        if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
        assert not probability_flow, "Probability flow not supported by ancestral sampling"

    def vesde_update_fn(self, x, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        sigma = sde.discrete_sigmas[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
        score = self.score_fn(x, t)
        x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[:, None, None, None]
        std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
        noise = torch.randn_like(x)
        x = x_mean + std[:, None, None, None] * noise
        return x, x_mean

    def vpsde_update_fn(self, x, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        beta = sde.discrete_betas.to(t.device)[timestep]
        score = self.score_fn(x, t)
        x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
        return x, x_mean

    def update_fn(self, x, t):
        if isinstance(self.sde, sde_lib.VESDE):
            return self.vesde_update_fn(x, t)
        elif isinstance(self.sde, sde_lib.VPSDE):
            return self.vpsde_update_fn(x, t)


@register_predictor(name='none')
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde, score_fn, probability_flow=False):
        pass

    def update_fn(self, x, t):
        return x, x


# ==========================================================
@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, sde_lib.VPSDE) \
                and not isinstance(sde, sde_lib.VESDE) \
                and not isinstance(sde, sde_lib.subVPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for i in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

        return x, x_mean


# ==================================================================================================

@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
    """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

  We include this corrector only for completeness. It was not directly used in our paper.
  """

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, sde_lib.VPSDE) \
                and not isinstance(sde, sde_lib.VESDE) \
                and not isinstance(sde, sde_lib.subVPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        std = self.sde.marginal_prob(x, t)[1]

        for i in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            step_size = (target_snr * std) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

        return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, score_fn, snr, n_steps):
        pass

    def update_fn(self, x, t):
        return x, x


# ========================================================================================================

def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous):
    """A wrapper that configures and returns the update function of predictors."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(sde, score_fn, probability_flow)
    else:
        predictor_obj = predictor(sde, score_fn, probability_flow)
    return predictor_obj.update_fn(x, t)


def shared_corrector_update_fn(x, t, sde, model, corrector, continuous, snr, n_steps):
    """A wrapper tha configures and returns the update function of correctors."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
    else:
        corrector_obj = corrector(sde, score_fn, snr, n_steps)
    return corrector_obj.update_fn(x, t)


# ========================================================================================================

def get_pc_sampler(sde_up, sde_down,sde_middle, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda'):
    """Create a Predictor-Corrector (PC) sampler.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
    # Create predictor & corrector update functions
    # up
    predictor_update_fn_up = functools.partial(shared_predictor_update_fn,
                                               sde=sde_up,
                                               predictor=predictor,
                                               probability_flow=probability_flow,
                                               continuous=continuous)
    corrector_update_fn_up = functools.partial(shared_corrector_update_fn,
                                               sde=sde_up,
                                               corrector=corrector,
                                               continuous=continuous,
                                               snr=snr,
                                               n_steps=n_steps)
    # down
    predictor_update_fn_down = functools.partial(shared_predictor_update_fn,
                                                 sde=sde_down,
                                                 predictor=predictor,
                                                 probability_flow=probability_flow,
                                                 continuous=continuous)
    corrector_update_fn_down = functools.partial(shared_corrector_update_fn,
                                                 sde=sde_down,
                                                 corrector=corrector,
                                                 continuous=continuous,
                                                 snr=snr,
                                                 n_steps=n_steps)
    # middle
    predictor_update_fn_middle = functools.partial(shared_predictor_update_fn,
                                                 sde=sde_middle,
                                                 predictor=predictor,
                                                 probability_flow=probability_flow,
                                                 continuous=continuous)
    corrector_update_fn_middle = functools.partial(shared_corrector_update_fn,
                                                 sde=sde_middle,
                                                 corrector=corrector,
                                                 continuous=continuous,
                                                 snr=snr,
                                                 n_steps=n_steps)
####################################################################################

    def im2row(im, winSize):
        size = (im).shape
        out = np.zeros(
            ((size[0] - winSize[0] + 1) * (size[1] - winSize[1] + 1), winSize[0] * winSize[1], size[2]),
            dtype=np.float32)
        count = -1
        for y in range(winSize[1]):
            for x in range(winSize[0]):
                count = count + 1
                temp1 = im[x:(size[0] - winSize[0] + x + 1), y:(size[1] - winSize[1] + y + 1), :]
                temp2 = np.reshape(temp1,
                                   [(size[0] - winSize[0] + 1) * (size[1] - winSize[1] + 1), 1, size[2]],
                                   order='F')

                out[:, count, 0] = np.squeeze(temp2)

        return out

    def svd_zl(input, cutSize=64):
        if useSvd == False:
            return np.array(input, dtype=np.float32)
        # SVD ===============================================
        svd_input = torch.tensor(input, dtype=torch.float32)
        U, S, V = torch.svd(svd_input)
        S = torch.diag(S)

        U = np.array(U, dtype=np.float32)
        S = np.array(S, dtype=np.float32)
        V = np.array(V, dtype=np.float32)
        uu = U[:, 0:math.floor(cutSize)]
        ss = S[0:math.floor(cutSize),
             0:math.floor(cutSize)]
        vv = V[:, 0:math.floor(cutSize)]

        A_svd = np.dot(np.dot(uu, ss), vv.T)

        return A_svd

    def row2im(mtx, size_data, winSize):
        size_mtx = mtx.shape
        sx = size_data[0]
        sy = size_data[1]
        sz = size_mtx[2]

        res = np.zeros((sx, sy, sz), dtype=np.float32)
        W = np.zeros((sx, sy, sz), dtype=np.float32)
        out = np.zeros((sx, sy, sz), dtype=np.float32)
        count = -1

        for y in range(winSize[1]):
            for x in range(winSize[0]):
                count = count + 1
                res[x: sx - winSize[0] + x + 1, y: sy - winSize[1] + y + 1, :] = res[
                                                                                 x: sx - winSize[0] + x + 1,
                                                                                 y: sy - winSize[1] + y + 1,
                                                                                 :] + np.reshape(
                    np.squeeze(mtx[:, count, :]), [sx - winSize[0] + 1, sy - winSize[1] + 1, sz], order='F')
                W[x: sx - winSize[0] + x + 1, y: sy - winSize[1] + y + 1, :] = W[x: sx - winSize[0] + x + 1,
                                                                               y: sy - winSize[1] + y + 1,
                                                                               :] + 1

        out = np.multiply(res, 1. / W)
        return out

    def hankel(input, ksize):
        hankel = im2row(input, ksize)
        size_temp = hankel.shape
        A = np.reshape(hankel, [size_temp[0], size_temp[1] * size_temp[2]], order='F')
        return A, size_temp

    def hankel_ni_zl(input, ksize, size_temp, size_last=None):
        if size_last is None:
            size_last = [768, 768, 1]
        input = np.reshape(input, [size_temp[0], size_temp[1], size_temp[2]], order='F')
        B = row2im(input, size_last, ksize)
        return B


####################################################################################

    def pc_sampler(up_model,down_model,middle_model,check_num,predict,correct):
        with torch.no_grad():
            def gyh(inputs):

                maxvalue1 = np.max(inputs)
                minvalue1 = np.min(inputs)
                result_gyh = (inputs - minvalue1) / (maxvalue1 - minvalue1)
                return result_gyh
            max_psnr = 0

            img_List = np.load('./Test_CT/batch_img.npy') # test

            cut0 = 46
            ksize = [8,8]
            batch_result = np.zeros([12, 512, 512])
            path = ["../Test_CT/L067_FD_1_1.CT.0001.0001.2015.12.22.18.09.40.840353.358074219.IMA"]
            file_name = ['0319', '0160', '0001', '0354', '0405', '0292', '0462', '0137', '0521', '0062', '0248', '0361']
            picNum = img_List.shape[0]

            for picNO in range(picNum):
                print(f"picNO is : {picNO}")
                img = img_List[picNO,...]

                batch_size = 1
                max_psnr = np.zeros(batch_size)
                max_ssim = np.zeros(batch_size)
                min_mse = 999 * np.ones(batch_size)
                rec_img = np.zeros_like(img)
                best_img = np.zeros_like(img)

                sinogram = Fan_ray_trafo(img).data
                ori_img = Fan_FBP(sinogram).data


                sinogram_input = init_ct_op(img)
                write_img(sinogram_input, 'sinogram_input.png')
                img_input = Fan_FBP(sinogram_input).data
                write_img(img_input, 'img_input.png')
                sinogram_input = padding_img(sinogram_input)
                sinogram_ldct = unpadding_img(sinogram_input)


                x = np.copy(sinogram_ldct)
                z0 = np.copy(x)

                x0 = sinogram_input



                def toNumpy(tensor):
                    if isinstance(tensor, torch.Tensor):
                        return tensor.cpu().numpy()
                    else:
                        return tensor

                def toTensor(array):
                    return torch.from_numpy(array).cuda()

                def im2row(im, winSize):
                    size = (im).shape
                    out = np.zeros(
                        ((size[0] - winSize[0] + 1) * (size[1] - winSize[1] + 1), winSize[0] * winSize[1], size[2]),
                        dtype=np.float32)
                    count = -1
                    for y in range(winSize[1]):
                        for x in range(winSize[0]):
                            count = count + 1
                            temp1 = im[x:(size[0] - winSize[0] + x + 1), y:(size[1] - winSize[1] + y + 1), :]
                            temp2 = np.reshape(temp1,
                                               [(size[0] - winSize[0] + 1) * (size[1] - winSize[1] + 1), 1, size[2]],
                                               order='F')
                            out[:, count, 0] = np.squeeze(temp2)

                    return out

                def svd_zl(input, cutSize=64):
                    # SVD ===============================================
                    svd_input = torch.tensor(input, dtype=torch.float32)
                    U, S, V = torch.svd(svd_input)
                    S = torch.diag(S)

                    U = np.array(U, dtype=np.float32)
                    S = np.array(S, dtype=np.float32)
                    V = np.array(V, dtype=np.float32)
                    # zero jie duan
                    uu = U[:, 0:math.floor(cutSize)]
                    ss = S[0:math.floor(cutSize),
                         0:math.floor(cutSize)]
                    vv = V[:, 0:math.floor(cutSize)]

                    A_svd = np.dot(np.dot(uu, ss), vv.T)
                    return A_svd

                def row2im(mtx, size_data, winSize):
                    size_mtx = mtx.shape
                    sx = size_data[0]
                    sy = size_data[1]
                    sz = size_mtx[2]

                    res = np.zeros((sx, sy, sz), dtype=np.float32)
                    W = np.zeros((sx, sy, sz), dtype=np.float32)
                    out = np.zeros((sx, sy, sz), dtype=np.float32)
                    count = -1

                    for y in range(winSize[1]):
                        for x in range(winSize[0]):
                            count = count + 1
                            res[x: sx - winSize[0] + x + 1, y: sy - winSize[1] + y + 1, :] = res[
                                                                                             x: sx - winSize[0] + x + 1,
                                                                                             y: sy - winSize[1] + y + 1,
                                                                                             :] + np.reshape(
                                np.squeeze(mtx[:, count, :]), [sx - winSize[0] + 1, sy - winSize[1] + 1, sz], order='F')
                            W[x: sx - winSize[0] + x + 1, y: sy - winSize[1] + y + 1, :] = W[x: sx - winSize[0] + x + 1,
                                                                                           y: sy - winSize[1] + y + 1,
                                                                                           :] + 1

                    out = np.multiply(res, 1. / W)
                    return out

                def hankel(input, ksize):
                    '''

                    Args:
                        input: [x*y*c] c is channel
                        ksize:

                    Returns:

                    '''
                    hankel = im2row(input, ksize)
                    size_temp = hankel.shape
                    A = np.reshape(hankel, [size_temp[0], size_temp[1] * size_temp[2]], order='F')
                    return A, size_temp

                def hankel_ni_zl(input, ksize, size_temp, size_last=None):
                    '''

                    Args:
                        input: [9999,99]
                        ksize: [8,8]
                        size_temp:
                        size_last: for example, [768, 768, 1]

                    Returns:

                    '''
                    if size_last is None:
                        size_last = [768, 768, 1]
                    input = np.reshape(input, [size_temp[0], size_temp[1], size_temp[2]], order='F')
                    B = row2im(input, size_last, ksize)
                    return B

                def unfoldTo2D(thing_3D, index):
                    # A's size -- 9*256*256
                    '''
                    useage:
                        A_new = unfoldTo2D(A_complex, 1)
                        A_new = svd_zl(A_new, cut1)
                        A_new = pileTo3D(A_new, 1)
                    Args:
                        thing_3D:
                        index: 0,1,2
                    Returns:
                    '''

                    size_3D = thing_3D.shape
                    thing_2D = np.zeros((size_3D[0] * size_3D[1], size_3D[2]), dtype=np.float32)
                    if index == 0:
                        for i in range(size_3D[0]):
                            thing_2D[size_3D[1] * i:size_3D[1] * (i + 1), :] = thing_3D[i, :, :]
                    elif index == 1:
                        for i in range(size_3D[1]):
                            thing_2D[size_3D[0] * i:size_3D[0] * (i + 1), :] = thing_3D[:, i, :]
                    elif index == 2:
                        for i in range(size_3D[2]):
                            thing_2D[size_3D[0] * i:size_3D[0] * (i + 1), :] = thing_3D[:, :, i]
                    else:
                        return -996

                    return thing_2D

                def pileTo3D(thing_2D, index, size_3D=(121, 512, 512)):

                    size_2D = thing_2D.shape
                    thing_3D = np.zeros((size_3D[0], size_3D[1], size_3D[2]), dtype=np.float32)
                    if index == 0:
                        for i in range(size_3D[0]):
                            thing_3D[i, :, :] = thing_2D[size_3D[1] * i:size_3D[1] * (i + 1), :]
                    elif index == 1:
                        for i in range(size_3D[1]):
                            thing_3D[:, i, :] = thing_2D[size_3D[0] * i:size_3D[0] * (i + 1), :]
                    elif index == 2:
                        for i in range(size_3D[2]):
                            thing_3D[:, :, i] = thing_2D[size_3D[0] * i:size_3D[0] * (i + 1), :]
                    else:
                        return -996

                    return thing_3D


                x0 = (x0[..., None])
                A, size_temp = hankel(x0, ksize)
                x0_up = pileTo3D(A[0:289560, ...], 0, [4524, 64, 64])
                x0_down = pileTo3D(A[289560:, ...], 0, [4524, 64, 64])
                x0_middle = pileTo3D(A[144780:434340, ...], 0, [4524, 64, 64])

                x0_up = toTensor(x0_up[None, ...])
                x0_down = toTensor(x0_down[None, ...])
                x0_middle = toTensor(x0_middle[None, ...])

                timesteps_up = torch.linspace(sde_up.T, eps, sde_up.N, device=device)
                timesteps_down = torch.linspace(sde_down.T, eps, sde_down.N, device=device)
                timesteps_middle = torch.linspace(sde_middle.T, eps, sde_middle.N, device=device)


                stepStart = 1400
                stepEnd = 2000
                for i in range(stepStart, stepEnd):
                    t_up = timesteps_up[i]
                    t_down = timesteps_down[i]
                    t_middle = timesteps_middle[i]

                    vec_t_up = torch.ones(x0_up.shape[0], device=t_up.device) * t_up
                    vec_t_down = torch.ones(x0_down.shape[0], device=t_down.device) * t_down
                    vec_t_middle = torch.ones(x0_middle.shape[0], device=t_middle.device) * t_middle

                    x01, x0_up = predictor_update_fn_up(x0_up, vec_t_up, model=up_model)
                    x02, x0_down = predictor_update_fn_down(x0_down, vec_t_down, model=down_model)
                    x03, x0_middle = predictor_update_fn_middle(x0_middle, vec_t_middle, model=middle_model)

                    # up
                    x0_up = unfoldTo2D(toNumpy(x0_up).squeeze(), 0)
                    A_fix_up = A[4524 * 64:289560, :]
                    x0_up = np.concatenate((x0_up, A_fix_up), 0)
                    x0_up = svd_zl(x0_up, cut0)

                    # down
                    x0_down = unfoldTo2D(toNumpy(x0_down).squeeze(), 0)
                    A_fix_down = A[289560 + 4524 * 64:, :]
                    x0_down = np.concatenate((x0_down, A_fix_down), 0)
                    x0_down = svd_zl(x0_down, cut0)

                    # middle
                    x0_middle = unfoldTo2D(toNumpy(x0_middle).squeeze(), 0)
                    A_fix_middle = A[144780 + 4524 * 64:434340, :]
                    x0_middle = np.concatenate((x0_middle, A_fix_middle), 0)
                    x0_middle = svd_zl(x0_middle, cut0)


                    x0 = np.concatenate((x0_up, x0_down), 0) 

                    average_result = np.mean([x0[144780:434340, :], x0_middle], axis=0)
                    x0[144780:434340, :] = average_result


                    x0 = hankel_ni_zl(x0, ksize, size_temp, [768, 768, 1])
                    #x0 = toTensor(x0)
                    x0 = unpadding_img(x0).squeeze()

                    ########################################################
                    n = sinogram_ldct
                    # ********** PWLS ********* ##
                    w = 1 / (100 * (np.exp(x / 220000)))
                    w = np.diag(w)
                    hyper = 150
                    sum_diff = x - x0
                    norm_diff = x - n
                    x_new = z0 - (w * norm_diff + 2 * hyper * sum_diff) / (2 * hyper + w)
                    z0 = x_new + 0.3 * (x_new - x)
                    x = x_new
                    x0 = x

                    x0 = tvdenoise(x0, 150, 2)
                    x0=toNumpy(x0)

                    x0 = padding_img(x0)


                    x0 = (x0[..., None])
                    A, size_temp = hankel(x0, ksize)
                    x0_up = pileTo3D(A[0:289560, ...], 0, [4524, 64, 64])
                    x0_down = pileTo3D(A[289560:, ...], 0, [4524, 64, 64])
                    x0_middle = pileTo3D(A[144780:434340, ...], 0, [4524, 64, 64])

                    x0_up = toTensor(x0_up[None, ...])
                    x0_down = toTensor(x0_down[None, ...])
                    x0_middle = toTensor(x0_middle[None, ...])


                    x01, x0_up = corrector_update_fn_up(x0_up, vec_t_up, model=up_model)
                    x02, x0_down = corrector_update_fn_down(x0_down, vec_t_down, model=down_model)
                    x03, x0_middle = corrector_update_fn_middle(x0_middle, vec_t_middle, model=middle_model)

                    # up
                    x0_up = unfoldTo2D(toNumpy(x0_up).squeeze(), 0)
                    A_fix_up = A[4524 * 64:289560, :]
                    x0_up = np.concatenate((x0_up, A_fix_up), 0)
                    x0_up = svd_zl(x0_up, cut0)

                    # down
                    x0_down = unfoldTo2D(toNumpy(x0_down).squeeze(), 0)
                    A_fix_down = A[289560 + 4524 * 64:, :]
                    x0_down = np.concatenate((x0_down, A_fix_down), 0)
                    x0_down = svd_zl(x0_down, cut0)

                    # middle
                    x0_middle = unfoldTo2D(toNumpy(x0_middle).squeeze(), 0)
                    A_fix_middle = A[144780 + 4524 * 64:434340, :]
                    x0_middle = np.concatenate((x0_middle, A_fix_middle), 0)
                    x0_middle = svd_zl(x0_middle, cut0)

                    x0 = np.concatenate((x0_up, x0_down), 0)

                    average_result = np.mean([x0[144780:434340, :], x0_middle], axis=0)
                    x0[144780:434340, :] = average_result
                    
                    x0 = hankel_ni_zl(x0, ksize, size_temp, [768, 768, 1])
                    #x0 = toTensor(x0)
                    x0 = unpadding_img(x0).squeeze()  # ([720, 720])

                    # ********** PWLS ********* ##
                    w = 1 / (100 * (np.exp(x / 220000)))
                    w = np.diag(w)
                    hyper = 150
                    sum_diff = x - x0
                    norm_diff = x - sinogram_ldct
                    x_new = z0 - (w * norm_diff + 2 * hyper * sum_diff) / (2 * hyper + w)
                    z0 = x_new + 0.3 * (x_new - x)
                    x = x_new
                    x0 = x

                    tmp = np.copy(x0)

                    tmp = tvdenoise(tmp, 150, 2)
                    tmp = toNumpy(tmp)
                    
                    x0 = padding_img(x0)

                    x0 = (x0[..., None])
                    A, size_temp = hankel(x0, ksize)
                    x0_up = pileTo3D(A[0:289560, ...], 0, [4524, 64, 64])
                    x0_down = pileTo3D(A[289560:, ...], 0, [4524, 64, 64])
                    x0_middle = pileTo3D(A[144780:434340, ...], 0, [4524, 64, 64])

                    x0_up = toTensor(x0_up[None, ...])
                    x0_down = toTensor(x0_down[None, ...])
                    x0_middle = toTensor(x0_middle[None, ...])


                    rec_img = Fan_FBP(tmp).data
                    write_img(rec_img, 'rec_img.png')

                    rec_img = np.squeeze(rec_img)
                    rec_img = rec_img[None,...]
                    ori_img = np.squeeze(ori_img)
                    ori_img = ori_img[None,...]

                    psnr0,ssim0,mse0 = indicate(rec_img, ori_img)

                    c = max_psnr < psnr0
                    if np.sum(c) > 0.01:
                        max_psnr = max_psnr * (1 - c) + psnr0 * c
                        max_ssim = max_ssim * (1 - c) + ssim0 * c
                        min_mse = min_mse * (1 - c) + mse0 * c
                        min_mse = min_mse * (1 - c) + mse0 * c
                        c = c[..., None, None]
                        best_img = best_img * (1 - c) + rec_img * c


                    print(f"Step: {i}  PSNR:{np.round(psnr0[:4], 3)}  PSNR:{np.round(max_psnr[0],3)}  MSE:{np.round(1000 * mse0[:4], 3)}  pic: {picNO}   cut0: {cut0} ")

                    path_dir = f"./result_up_down_hankel/cut_{cut0}"
                    path_filename = f"P_{picNO}"
                    path_all_step = f"{path_dir}/{path_filename}_all.csv"
                    path_best_step = f"{path_dir}/{path_filename}.csv"
                    path_best_step_img = f"{path_dir}/{path_filename}.png"
                    path_best_npy = f"{path_dir}/{path_filename}.npy"
                    path_best_mat = f"{path_dir}/{path_filename}.mat"

                    # delete all step when get a new start
                    if i == stepStart:
                        t = 0
                        if os.path.exists(path_best_step):
                            os.remove(path_best_step)
                            t = t + 1
                        if os.path.exists(path_all_step):
                            os.remove(path_all_step)
                            t = t + 1
                        if t == 2:
                            print("Delete old files succeed ... Now it is a New start....")

                        # save all step

                    def WriteInfo_zl(path, **args):
                        ppp = os.path.split(path)
                        if not os.path.isdir(ppp[0]):
                            os.makedirs(ppp[0])


                        try:
                            args = args['args']
                        except:
                            pass
                        args['Time'] = [str(datetime.datetime.now())[:-7]]
                        try:
                            df = pd.read_csv(path, encoding='utf-8', engine='python')
                        except:
                            df = pd.DataFrame()

                        df2 = pd.DataFrame(pd.DataFrame.from_dict(args, orient='index').values.T,
                                          columns=list(args.keys()))

                        df = df.append(df2)
                        df.to_csv(path, index=False)

                    if i >= stepStart:
                        WriteInfo_zl(path_all_step,
                                     PSNR=psnr0[0], SSIM=ssim0[0], MSE=mse0[0],
                                     step=i, Check_num=check_num)
                    # save best step
                    if i == stepEnd-1:
                        print("fuck")
                        WriteInfo_zl(path_best_step,
                                     PSNR=max_psnr[0], SSIM=max_ssim[0], MSE=min_mse[0],
                                     step=i, Check_num=check_num)
                        np.save(path_best_npy, best_img)
                        savemat(path_best_mat, mdict={'data': best_img})

                        batch_result[picNO, ...] = best_img

                        plt.imshow(best_img[0, :, :], cmap=plt.get_cmap('gray'))
                        plt.savefig(path_best_step_img)

                        break



                    # endregion

                    proj_ideal = Fan_ray_trafo(Fan_reco_space.element(img)).data
                    write_img(proj_ideal, 'ideal_sinogram.png')
                    write_img(sinogram_ldct, 'sinogram_ldct.png')
                    savemat('ideal_sinogram.mat', mdict={'data': proj_ideal})
                    savemat('sinogram_ldct.mat', mdict={'data': sinogram_ldct})

                print("MAX:  PSNR:{} MSE:{}".format(np.round(max_psnr[:4], 3), np.round(1000 * min_mse[:4], 3)))

            global photons
            photons2 = int(photons)
            isExists = os.path.exists(f'./result_all/noise_{photons2}/')
            if not isExists:
                os.makedirs(f'./result_all/noise_{photons2}/')


    return pc_sampler


def get_ode_sampler(sde, shape, inverse_scaler,
                    denoise=False, rtol=1e-5, atol=1e-5,
                    method='RK45', eps=1e-3, device='cuda'):
    #"""Probability flow ODE sampler with the black-box ODE solver.

  #Args:
    #sde: An `sde_lib.SDE` object that represents the forward SDE.
    #shape: A sequence of integers. The expected shape of a single sample.
    #inverse_scaler: The inverse data normalizer.
    #denoise: If `True`, add one-step denoising to final samples.
    #rtol: A `float` number. The relative tolerance level of the ODE solver.
    #atol: A `float` number. The absolute tolerance level of the ODE solver.
    #method: A `str`. The algorithm used for the black-box ODE solver.
      #See the documentation of `scipy.integrate.solve_ivp`.
    #eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
    #device: PyTorch device.

  #Returns:
    #A sampling function that returns samples and the number of function evaluations during sampling.
  #"""

    def denoise_update_fn(model, x):
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        # Reverse diffusion predictor for denoising
        predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
        vec_eps = torch.ones(x.shape[0], device=x.device) * eps
        _, x = predictor_obj.update_fn(x, vec_eps)
        return x

    def drift_fn(model, x, t):
        """Get the drift function of the reverse-time SDE."""
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]

    def ode_sampler(model, z=None):
        #"""The probability flow ODE sampler with black-box ODE solver.

    #Args:
      #model: A score model.
      #z: If present, generate samples from latent code `z`.
    #Returns:
      #samples, number of function evaluations.
    #"""
        with torch.no_grad():
            # Initial sample
            if z is None:
                # If not represent, sample the latent code from the prior distibution of the SDE.
                x = sde.prior_sampling(shape).to(device)
            else:
                x = z

            def ode_func(t, x):
                x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
                vec_t = torch.ones(shape[0], device=x.device) * t
                drift = drift_fn(model, x, vec_t)
                return to_flattened_numpy(drift)

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                           rtol=rtol, atol=atol, method=method)
            nfe = solution.nfev
            x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

            # Denoising is equivalent to running one predictor step without adding noise
            if denoise:
                x = denoise_update_fn(model, x)

            x = inverse_scaler(x)
            return x, nfe

    return ode_sampler
