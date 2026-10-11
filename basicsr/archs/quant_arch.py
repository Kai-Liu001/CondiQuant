from typing import Any
from torch.nn import Module, Linear, Parameter, Conv2d, ReLU
import torch
from torch import Tensor, FloatTensor
from torch.autograd import Function
from torch.autograd.function import _ContextMethodMixin
from functools import partial
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging

# 全局变量记录总时间
total_time = 0.0

calibrated_num = 0
total_num = 0

def quant(input:torch.Tensor, lb:float, ub:float, bit:int):
    input = input.clamp(lb, ub)
    s = (ub - lb).clamp_min(torch.finfo(input.dtype).eps) / (2 ** bit -1)
    input = (input - lb)/s
    input = input.round()
    input = input * s + lb
    return input

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(out_chn), requires_grad=True)
    def forward(self, x):
        # x:Tensor
        out = x + self.bias.expand_as(x)

        return out

def cal_mse(input:torch.Tensor, lb:float, ub:float, bit:int):
    quant_input = quant(input, lb, ub, bit)
    res = float(torch.norm(input - quant_input))
    return res

def change_singular(weight, x, lambda_reg=0.003, learning_rate=0.01, num_iterations=50):
    with torch.enable_grad():

        device = weight.device

        #t = 1
        U_, S_, Vh_ = torch.linalg.svd(weight, full_matrices=False)
        mean_value = torch.mean(S_)

        #median_value = torch.median(S_)

        #quantile_40 = torch.quantile(S_, 0.4)

        #quantile_60 = torch.quantile(S_, 0.6)
        t = mean_value

        weight = weight.detach().T.clone().to(device)

        X = x.clone().detach().to(device)
        Y = X @ weight.clone().detach().to(device)

        W = nn.Parameter(weight)
        F_norm = torch.norm(weight, p='fro')

        optimizer = optim.Adam([W], lr=learning_rate)

        def loss_fn(X, Y, W):
            data_loss = F.mse_loss(Y, X @ W)
            return data_loss

        for iter in range(1, num_iterations + 1):
            optimizer.zero_grad()

            loss = loss_fn(X, Y, W)

            loss.backward()

            optimizer.step()

            with torch.no_grad():
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)

                S_new = (S + 2 * lambda_reg * t) / (1 + 2 * lambda_reg)


                W.copy_(U @ torch.diag(S_new) @ Vh)
                #if f_norm <= epsilon:
                #    print(iter)
                #    break


            if iter % 10 == 0 or iter == 1:
                cond_num = S.max() / S.min()
                logging.getLogger("basicsr").debug("SVD step %d: MSE %.6g, condition %.6g", iter, loss.item(), cond_num.item())

        U_final, S_final, Vh_final = torch.linalg.svd(W, full_matrices=False)
        final_cond_num = S_final.max() / S_final.min()
        logging.getLogger("basicsr").debug("Final condition number: %.6g", final_cond_num.item())

    return W.detach().T.contiguous()


def DOBI(x, bit:int, one_direction = False, num:int=10, channel_wise: bool = False):

    if channel_wise:
        x_clone = x.clone().detach()
        n_channels = x_clone.shape[-1] if len(x.shape) == 3 else x_clone.shape[0]
        if len(x.shape) == 4:
            n_channels = x_clone.shape[1] * x_clone.shape[-1]
            x_max = x_clone.abs().max(dim=0)[0].max(dim=1)[0].reshape(-1)
        elif len(x.shape) == 2:
            x_max = x_clone.abs().max(dim=-1)[0]
        elif len(x.shape) == 3:
            x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
        else:
            raise NotImplementedError

        lb = x_max.clone()
        ub = x_max.clone()

        for c in range(n_channels):
            if len(x.shape) == 3:
                lb[c], ub[c] = DOBI(x_clone[:,:,c], bit, one_direction=one_direction, channel_wise=False)
            elif len(x.shape) == 4:
                lb[c], ub[c] = DOBI(x_clone[:, c // x_clone.shape[-1], :, c % x_clone.shape[-1]], bit, one_direction=one_direction, channel_wise=False)
            else:
                lb[c], ub[c] = DOBI(x_clone[c], bit, one_direction=one_direction, channel_wise=False)
        if len(x.shape) == 4:
            lb = lb.view(1, x_clone.shape[1], 1, -1)
            ub = ub.view(1, x_clone.shape[1], 1, -1)
        elif len(x.shape) == 2:
            lb = lb.view(-1, 1)
            ub = ub.view(-1, 1)
        elif len(x.shape) == 3:
            lb = lb.view(1, 1, -1)
            ub = ub.view(1, 1, -1)
        else:
            raise NotImplementedError
    else:
        x_clone = x.clone().detach()
        x_max = x_clone.max()
        x_min = x_clone.min()
        history_min = float('inf')
        try:
            new_max = torch.quantile(x_clone.reshape(-1), 0.9)
            new_min = torch.quantile(x_clone.reshape(-1), 0.1)
        except:
            new_max = torch.tensor(np.percentile(
                x_clone.reshape(-1).cpu(), 90),
                device=x_clone.device,
                dtype=torch.float32)
            new_min = torch.tensor(np.percentile(
                x_clone.reshape(-1).cpu(), 10),
                device=x_clone.device,
                dtype=torch.float32)

        if one_direction:
            diff = (x_max - new_max) / num
            for i in range(num):
                now_lb = x_min
                now_ub = x_max - diff * i
                cur_value = cal_mse(x, now_lb, now_ub, bit)
                if cur_value < history_min:
                    lb = now_lb
                    ub = now_ub
                    history_min = cur_value
        else:
            diff1 = (x_max - new_max) / (num)
            diff2 = (new_min - x_min) / (num)
            for i in range(num):
                now_lb = x_min + diff2 * i
                now_ub = x_max - diff1 * i
                cur_value = cal_mse(x, now_lb, now_ub, bit)
                if cur_value < history_min:
                    lb = now_lb
                    ub = now_ub
                    history_min = cur_value


    return lb, ub

def init_bound(x):
    x_clone = x.clone().detach()
    n_channels = x_clone.shape[-1] if len(x.shape) == 3 else x_clone.shape[0]
    if len(x.shape) == 4:
        n_channels = x_clone.shape[1] * x_clone.shape[-1]
        x_max = x_clone.abs().max(dim=0)[0].max(dim=1)[0].reshape(-1)
    elif len(x.shape) == 2:
        x_max = x_clone.abs().max(dim=-1)[0]
    elif len(x.shape) == 3:
        x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
    else:
        raise NotImplementedError

    lb = -x_max.clone().detach().clamp_min(torch.finfo(x.dtype).eps)
    ub = x_max.clone().detach().clamp_min(torch.finfo(x.dtype).eps)

    if len(x.shape) == 4:
        lb = lb.view(1, x_clone.shape[1], 1, -1)
        ub = ub.view(1, x_clone.shape[1], 1, -1)
    elif len(x.shape) == 2:
        lb = lb.view(-1, 1)
        ub = ub.view(-1, 1)
    elif len(x.shape) == 3:
        lb = lb.view(1, 1, -1)
        ub = ub.view(1, 1, -1)

    return lb, ub

class Differentiable_Round(Function):
    @staticmethod
    def forward(ctx: _ContextMethodMixin, x: Tensor):
        return x.round()

    @staticmethod
    def backward(ctx: _ContextMethodMixin, grad_outputs):
        return grad_outputs


class Differentiable_Clip(Function):
    @staticmethod
    def forward(
        ctx: _ContextMethodMixin,
        input: Tensor,
        min_val: Tensor,
        max_val: Tensor,
    ) -> Any:
        assert isinstance(max_val, Tensor), 'nononno'
        ctx.save_for_backward(input, min_val, max_val)
        # if isinstance(min_val, Tensor):
        #     min_val = min_val.item()
        # if isinstance(max_val, Tensor):
        #     max_val = max_val.item()

        # output = input.clamp(min_val, max_val)
        output = torch.where(input < min_val, min_val, input)
        output = torch.where(output > max_val, max_val, output)
        return output

    @staticmethod
    def backward(ctx: _ContextMethodMixin, grad_outputs: Tensor) -> Any:
        input, min_val, max_val = ctx.saved_tensors

        grad_input = grad_outputs.clone()
        grad_input[(input < min_val) | (input > max_val)] = 0

        grad_min = grad_outputs.clone()
        grad_min[input > min_val] = 0
        #grad_min = grad_min.sum().view(1)

        grad_max = grad_outputs.clone()
        grad_max[input < max_val] = 0
        #grad_max = grad_max.sum().view(1)

        reduce_dims_min = [dim for dim, size in enumerate(min_val.shape) if size == 1]
        reduce_dims_max = [dim for dim, size in enumerate(max_val.shape) if size == 1]

        grad_min = grad_min.sum(dim=reduce_dims_min, keepdim=True)
        grad_max = grad_max.sum(dim=reduce_dims_max, keepdim=True)

        return grad_input, grad_min, grad_max


class FakeQuantizerBase(Module):
    def __init__(self, int_quant: bool = True, bit:int=4) -> None:
        super().__init__()
        self.lower_bound = nn.Parameter()
        self.upper_bound = nn.Parameter()
        self.n_bit = nn.Parameter(torch.empty(0), requires_grad=False)
        # self.n_bit = bit
        self.set_n_bit_manually(bit)

        self.bit2bound = {}
        self.use_bit2bound = False
        self.size_of_input = None

        self.int_quant = int_quant

        self.clip = Differentiable_Clip.apply
        self.round = Differentiable_Round.apply

        self.calibrated = False

        global total_num
        total_num += 1


    def set_int_quant(self, enable: bool):
        self.int_quant = enable

    def set_require_grad(self, enable_lb: bool, enable_up: bool, enable_nbit: bool):
        self.lower_bound.requires_grad = enable_lb
        self.upper_bound.requires_grad = enable_up
        # self.n_bit.requires_grad = enable_nbit

    def set_params_manually(self, lb: Tensor, ub: Tensor, n_bit: Tensor):
        device = self.lower_bound.device
        self.lower_bound.data = FloatTensor([lb]).data.clone().to(device)
        self.upper_bound.data = FloatTensor([ub]).data.clone().to(device)
        # self.n_bit.data = FloatTensor([n_bit]).data.clone().to(device)

    def set_params_lb_manually(self, lb: Tensor):
        device = self.lower_bound.device
        self.lower_bound.data = FloatTensor([lb]).data.clone().to(device)

    def set_params_ub_manually(self, ub: Tensor):
        device = self.upper_bound.device
        self.upper_bound.data = FloatTensor([ub]).data.clone().to(device)

    def set_n_bit_manually(self, n_bit):
        device = self.n_bit.device
        self.n_bit.data = FloatTensor([n_bit]).data.clone().to(device)

    def per_channel_bound(self, shape):

        if len(shape) == 2:
            self.lower_bound = nn.Parameter(self.lower_bound.repeat(shape[0], 1))
            self.upper_bound = nn.Parameter(self.upper_bound.repeat(shape[0], 1))
        elif len(shape) == 3:
            self.lower_bound = nn.Parameter(self.lower_bound.repeat(1, 1, shape[-1]))
            self.upper_bound = nn.Parameter(self.upper_bound.repeat(1, 1, shape[-1]))
        else:
            self.lower_bound = nn.Parameter(self.lower_bound.repeat(1, shape[1], 1, shape[-1]))
            self.upper_bound = nn.Parameter(self.upper_bound.repeat(1, shape[1], 1, shape[-1]))



class FakeQuantizerWeight(FakeQuantizerBase):
    def __init__(self,bit=4, bound_init="dobi") -> None:
        super(FakeQuantizerWeight, self).__init__(bit=bit)
        self.bound_init = bound_init

    def forward(self, x:torch.Tensor):
        if not self.calibrated:
            lb, rb = (DOBI(x, bit=self.n_bit, channel_wise=True)
                      if self.bound_init == "dobi" else init_bound(x))
            self.lower_bound = nn.Parameter(lb, requires_grad=True)
            self.upper_bound = nn.Parameter(rb, requires_grad=True)

            self.calibrated = True
            return x
        if self.size_of_input is None:
            self.size_of_input = x.numel()

        n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)

        if self.use_bit2bound:
            try:
                lb, ub = self.bit2bound[int(n_bits.item())]
                self.set_params_lb_manually(lb)
                self.set_params_ub_manually(ub)
            except Exception as e:
                print(f'use bit 2 bound.{int(n_bits.item())} not found.')

        # (u-l)/(2^n-1)
        s = (self.upper_bound - self.lower_bound).clamp_min(torch.finfo(x.dtype).eps) / (torch.pow(2, n_bits) - 1)

        # clip(x,l,u)
        c = self.clip(x, self.lower_bound, self.upper_bound)

        # int value \in [0,2^n-1]
        r = self.round((c - self.lower_bound) / s)

        return s * r + self.lower_bound

        # n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)

        # # (u-l)/(2^n-1)
        # s = (self.upper_bound - self.lower_bound) / torch.pow(2, n_bits)

        # # clip(x,l,u)
        # c = self.clip(x, self.lower_bound, self.upper_bound)

        # # int value \in [0,2^n-1]
        # # r = self.clip(self.round((c - self.lower_bound) / s - 0.5, 0, torch.pow(2, n_bits)-1))
        # r = self.clip(
        #     self.round((c - self.lower_bound) / s - 0.5),
        #     0, torch.pow(2, n_bits)-1
        #     )

        # return s * (r+0.5) + self.lower_bound


class FakeQuantizerAct(FakeQuantizerBase):
    def __init__(self, bit=4, one_direction=False, bound_init="dobi") -> None:
        """
        if dynamic, bound will be the minmax value of the input values.
        dynamic is only used for act.
        if running_stat, bound will be set by moving average.
        """
        super(FakeQuantizerAct, self).__init__(bit=bit)
        self.bound_init = bound_init

        self.running_stat = False # initial boundary
        self.first_iter = False # initial boundary
        self.dynamic = False
        self.beta = 0.995
        self.identity = False
        self.one_direction = one_direction
        self.move_b4 = nn.Module()
        self.move_aft = nn.Module()

    def forward(self, x):
        if not self.calibrated:
            lb, rb = (DOBI(x, bit=self.n_bit, one_direction=self.one_direction, channel_wise=True)
                      if self.bound_init == "dobi" else init_bound(x))
            self.lower_bound = nn.Parameter(lb, requires_grad=True)
            self.upper_bound = nn.Parameter(rb, requires_grad=True)
            self.move_b4 = LearnableBias(x.shape[-1]).to(device=x.device, dtype=x.dtype)
            self.move_aft = LearnableBias(x.shape[-1]).to(device=x.device, dtype=x.dtype)
            self.calibrated = True
            return x

        if self.size_of_input is None:
            self.size_of_input = x.numel()

        if self.identity:
            return x

        if self.dynamic:
            '''
                use min,max as clip boundary
                TODO: use other oberver later.
            '''
            n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)

            lb = torch.min(x).detach()
            ub = torch.max(x).detach()
            n_bits = n_bits.detach()

            # (u-l)/(2^n-1)
            s = (ub - lb) / (torch.pow(2, n_bits) - 1)

            # clip(x,l,u)
            c = self.clip(x, lb, ub)

            # int value \in [0,2^n-1]
            r = self.round((c - lb) / s)

            return s * r + lb

        if self.running_stat:
            if self.first_iter:
                self.lower_bound.data = torch.min(x).detach().clone()
                self.upper_bound.data = torch.max(x).detach().clone()
                self.first_iter = False
            else:
                self.lower_bound =  self.beta * self.lower_bound + (1-self.beta) *  torch.min(x)
                self.upper_bound =  self.beta * self.upper_bound + (1-self.beta) *  torch.max(x)
            return x


        n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)
        if self.use_bit2bound:
            try:
                lb, ub = self.bit2bound[int(n_bits.item())]
                self.set_params_lb_manually(lb)
                self.set_params_ub_manually(ub)
            except Exception as e:
                print(f'use bit 2 bound.{int(n_bits.item())} not found.')

        # (u-l)/(2^n-1)
        x = self.move_b4(x)
        #input = self.input_quant_fn(input)
        s = (self.upper_bound - self.lower_bound).clamp_min(torch.finfo(x.dtype).eps) / (torch.pow(2, n_bits) - 1)

        # clip(x,l,u)
        c = self.clip(x, self.lower_bound, self.upper_bound)

        # int value \in [0,2^n-1]
        r = self.round((c - self.lower_bound) / s)

        return self.move_aft(s * r + self.lower_bound)

        # n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)

        # # (u-l)/(2^n-1)
        # s = (self.upper_bound - self.lower_bound) / torch.pow(2, n_bits)

        # # clip(x,l,u)
        # c = self.clip(x, self.lower_bound, self.upper_bound)

        # # int value \in [0,2^n-1]
        #     r = self.clip(
        #     self.round((c - self.lower_bound) / s - 0.5),
        #     0, torch.pow(2, n_bits)-1
        #     )

        # return s * (r+0.5) + self.lower_bound

##########################################################
##########################################################


class FakeQuantizerWeightParam2(FakeQuantizerBase):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):

        # n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)

        # (u-l)/(2^n-1)
        s = (self.upper_bound - self.lower_bound) / self.n_bit

        # clip(x,l,u)
        c = self.clip(x, self.lower_bound, self.upper_bound)

        # int value \in [0,2^n-1]
        r = self.round((c - self.lower_bound) / s)

        return s * r + self.lower_bound


class FakeQuantizerActParam2(FakeQuantizerBase):
    def __init__(self) -> None:
        """
        if dynamic, bound will be the minmax value of the input values.
        dynamic is only used for act.
        if running_stat, bound will be set by moving average.
        """
        super().__init__()

        self.running_stat = True # initial boundary
        self.first_iter = True # initial boundary
        self.dynamic = False
        self.beta = 0.995

    def forward(self, x):
        if self.dynamic:
            '''
                use min,max as clip boundary
                TODO: use other oberver later.
            '''
            # n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)
            n_bits = self.n_bit
            lb = torch.min(x).detach()
            ub = torch.max(x).detach()
            n_bits = n_bits.detach()

            # (u-l)/(2^n-1)
            s = (ub - lb) / n_bits

            # clip(x,l,u)
            c = self.clip(x, lb, ub)

            # int value \in [0,2^n-1]
            r = self.round((c - lb) / s)

            return s * r + lb

        if self.running_stat:
            if self.first_iter:
                self.lower_bound.data = torch.min(x).detach().clone()
                self.upper_bound.data = torch.max(x).detach().clone()
                self.first_iter = False
            else:
                self.lower_bound =  self.beta * self.lower_bound + (1-self.beta) *  torch.min(x)
                self.upper_bound =  self.beta * self.upper_bound + (1-self.beta) *  torch.max(x)
            return x


        # n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)

        # (u-l)/(2^n-1)
        s = (self.upper_bound - self.lower_bound) / n_bits

        # clip(x,l,u)
        c = self.clip(x, self.lower_bound, self.upper_bound)

        # int value \in [0,2^n-1]
        r = self.round((c - self.lower_bound) / s)

        return s * r + self.lower_bound


class QuantBase(Module):
    def __init__(self,config, one_direction):
        super().__init__()
        self.quant = True
        self.bit = config['bit']
        self.weight_quantizer = FakeQuantizerWeight(self.bit, config.get('bound_init', 'dobi'))
        self.act_quantizer = FakeQuantizerAct(self.bit, one_direction, config.get('bound_init', 'dobi'))
        # if config['param'] == 1:
        #     self.weight_quantizer = FakeQuantizerWeight()
        #     self.act_quantizer = FakeQuantizerAct()
        # elif config['param'] == 2:
        #     self.weight_quantizer = FakeQuantizerWeightParam2()
        #     self.act_quantizer = FakeQuantizerActParam2()

    def get_weight_quantizer(self):
        return self.weight_quantizer

    def get_act_quantizer(self):
        return self.act_quantizer

    def set_quant_flag(self, enable: bool):
        self.quant = enable

    def set_require_grad(self, enable: bool):
        # 似乎没有必要分别设置.
        self.weight_quantizer.set_require_grad(enable,enable, enable)
        self.act_quantizer.set_require_grad(enable,enable, enable)

    def set_weight_bias_grad(self, enable: bool):
        self.weight.requires_grad = enable
        if self.bias:
            self.bias.requires_grad = enable

    def get_quant_weight_bias(self):
        quant_weight = self.weight_quantizer(self.weight)

        return (quant_weight, self.bias)


class QuantLinear(QuantBase):
    def __init__(self,config, one_direction=False):
        super().__init__(config, one_direction)
        self.cali = False
        self.svd_options = dict(config.get('svd', {}))
        self.svd_enabled = self.svd_options.pop('enabled', True)
        self.svd_options.pop('exclude_mlp', None)

    def load_values(self, value):
        min_value, max_value = value
        self.act_quantizer.set_params_lb_manually(min_value)
        self.act_quantizer.set_params_ub_manually(max_value)

    def set_param(self, linear: Linear):
        """
        must be called before forward.
        """

        self.in_feature = self.in_features = linear.in_features
        self.out_feature = self.out_features = linear.out_features

        self.weight = Parameter(linear.weight.data.clone())

        if linear.bias is not None:
            self.bias = Parameter(linear.bias.data.clone())
        else:
            self.bias = linear.bias

    def forward(self, x):
        if not self.quant:
            return F.linear(x, self.weight, self.bias)
        if not self.cali and self.svd_enabled:
            self.weight = nn.Parameter(change_singular(self.weight, x, **self.svd_options))
            self.cali = True

        quant_act = self.act_quantizer(x)
        quant_weight = self.weight_quantizer(self.weight)
        # bias don't need to be quant

        return F.linear(quant_act, quant_weight, self.bias)

class QuantLinearQKV(Module):
    def __init__(self,config):
        super().__init__()
        self.q = QuantLinear(config)
        self.k = QuantLinear(config)
        self.v = QuantLinear(config)

    def load_values(self, value):
        min_value, max_value = value
        self.q.act_quantizer.set_params_lb_manually(min_value)
        self.q.act_quantizer.set_params_ub_manually(max_value)
        self.k.act_quantizer.set_params_lb_manually(min_value)
        self.k.act_quantizer.set_params_ub_manually(max_value)
        self.v.act_quantizer.set_params_lb_manually(min_value)
        self.v.act_quantizer.set_params_ub_manually(max_value)

    def set_quant_flag(self, enable: bool):
        self.q.set_quant_flag(enable)
        self.k.set_quant_flag(enable)
        self.v.set_quant_flag(enable)

    def set_require_grad(self, enable: bool):
        # 似乎没有必要分别设置.
        self.q.set_require_grad(enable)
        self.k.set_require_grad(enable)
        self.v.set_require_grad(enable)


    def set_weight_bias_grad(self, enable: bool):
        self.q.set_weight_bias_grad(enable)
        self.k.set_weight_bias_grad(enable)
        self.v.set_weight_bias_grad(enable)


    def get_quant_weight_bias(self):
        w_q,b_q = self.q.get_quant_weight_bias()
        w_k,b_k = self.k.get_quant_weight_bias()
        w_v,b_v = self.v.get_quant_weight_bias()

        quant_weight = torch.cat([w_q, w_k, w_v],dim=0)
        if b_q is not None:
            bias = torch.cat([b_q,b_k,b_v])

        return (quant_weight, bias)

    def set_param(self, linear: Linear):
        """
        must be called before forward.
        """

        self.in_feature = self.in_features = linear.in_features
        self.out_feature = self.out_features = linear.out_features // 3

        linear_q = Linear(self.in_feature, self.out_feature,bias=linear.bias is not None)
        linear_k = Linear(self.in_feature, self.out_feature,bias=linear.bias is not None)
        linear_v = Linear(self.in_feature, self.out_feature,bias=linear.bias is not None)


        linear_q.weight.data,linear_k.weight.data,linear_v.weight.data = linear.weight.data.clone().reshape(3,self.in_feature,self.out_feature)


        if linear.bias is not None:
            linear_q.bias.data, linear_k.bias.data, linear_v.bias.data = linear.bias.data.clone().reshape(3,self.out_feature)

        self.q.set_param(linear_q)
        self.k.set_param(linear_k)
        self.v.set_param(linear_v)


    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        # print('qsize',q.size())
        # print('ksize',k.size())
        # print('vsize',v.size())
        return torch.cat([q,k,v], dim=-1)

        quant_act = self.act_quantizer(x)
        quant_weight = self.weight_quantizer(self.weight)
        # bias don't need to be quant

        return F.linear(quant_act, quant_weight, self.bias)


class QuantConv2d(QuantBase):
    def __init__(self,config):
        super().__init__(config)

    def set_param(self, conv: Conv2d):
        """
        must be called before forward.
        """

        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.conv_kwargs = {
            "stride": conv.stride,
            "padding": conv.padding,
            "dilation": conv.dilation,
            "groups": conv.groups,
        }
        self.weight = Parameter(conv.weight.data.clone())
        if conv.bias is not None:
            self.bias = Parameter(conv.bias.data.clone())
        else:
            self.bias = conv.bias

    def forward(self, x):
        if not self.quant:
            return F.conv2d(x, self.weight, self.bias, **self.conv_kwargs)

        quant_act = self.act_quantizer(x)
        quant_weight = self.weight_quantizer(self.weight)
        # bias doesn't need to be quant

        return F.conv2d(quant_act, quant_weight, self.bias, **self.conv_kwargs)
