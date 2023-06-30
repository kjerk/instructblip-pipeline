import collections.abc
import math
import warnings
from itertools import repeat

import torch

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
	if drop_prob == 0. or not training:
		return x
	keep_prob = 1 - drop_prob
	shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
	random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
	if keep_prob > 0.0 and scale_by_keep:
		random_tensor.div_(keep_prob)
	return x * random_tensor

def _ntuple(n):
	def parse(x):
		if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
			return tuple(x)
		return tuple(repeat(x, n))
	
	return parse

to_2tuple = _ntuple(2)

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
	with torch.no_grad():
		return _trunc_normal_(tensor, mean, std, a, b)

def _trunc_normal_(tensor, mean, std, a, b):
	def norm_cdf(x):
		return (1. + math.erf(x / math.sqrt(2.))) / 2.
	
	if (mean < a - 2 * std) or (mean > b + 2 * std):
		warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.", stacklevel=2)
	
	l = norm_cdf((a - mean) / std)
	u = norm_cdf((b - mean) / std)
	
	tensor.uniform_(2 * l - 1, 2 * u - 1)
	
	tensor.erfinv_()
	
	tensor.mul_(std * math.sqrt(2.))
	tensor.add_(mean)
	
	tensor.clamp_(min=a, max=b)
	return tensor

class LayerNorm(torch.nn.LayerNorm):
	"""Subclass torch's LayerNorm to handle fp16."""
	
	def forward(self, x: torch.Tensor):
		orig_type = x.dtype
		ret = super().forward(x.type(torch.float32))
		return ret.type(orig_type)

if __name__ == '__main__':
	print('__main__ not allowed in modules')
