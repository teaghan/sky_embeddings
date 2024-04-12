from torch import nn
import torch
import math
import torch.nn.functional as F
from einops import rearrange

# helpers
def exists(val):
    return val is not None

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)

# sin activation
class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

# siren layer
class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0=1., c=6., 
                 is_first=False, use_bias=True, activation=None, dropout=False):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        self.dim_out = dim_out
        self.dropout = dropout

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out =  F.linear(x, self.weight, self.bias)
        if self.dropout:
            out = F.dropout(out, training=self.training)
        out = self.activation(out)
        return out

# siren network

class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, 
                 w0=1., w0_initial=30., use_bias=True, 
                 final_activation=None, dropout=False):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first,
                dropout = dropout
            ))

        final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation, dropout = False)

    def forward(self, x, mods = None):

        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if exists(mod):
                x *= rearrange(mod, 'd -> () d')

        return self.last_layer(x)


"""
FCNEt used in the GeoPrior Paper
"""

class ResLayer(nn.Module):
    def __init__(self, linear_size):
        super(ResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y

        return out

class FCNet(nn.Module):
    def __init__(self, num_inputs, num_classes, dim_hidden):
        super(FCNet, self).__init__()
        self.inc_bias = False
        self.class_emb = nn.Linear(dim_hidden, num_classes, bias=self.inc_bias)

        self.feats = nn.Sequential(nn.Linear(num_inputs, dim_hidden),
                                    nn.ReLU(inplace=True),
                                    ResLayer(dim_hidden),
                                    ResLayer(dim_hidden),
                                    ResLayer(dim_hidden),
                                    ResLayer(dim_hidden))

    def forward(self, x):
        loc_emb = self.feats(x)
        class_pred = self.class_emb(loc_emb)
        return class_pred


####################### Spherical Harmonics utilities ########################
def associated_legendre_polynomial(l, m, x):
    pmm = torch.ones_like(x)
    if m > 0:
        somx2 = torch.sqrt((1 - x) * (1 + x))
        fact = 1.0
        for i in range(1, m + 1):
            pmm = pmm * (-fact) * somx2
            fact += 2.0
    if l == m:
        return pmm
    pmmp1 = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmmp1
    pll = torch.zeros_like(x)
    for ll in range(m + 2, l + 1):
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    return pll

def SH_renormalization(l, m):
    return math.sqrt((2.0 * l + 1.0) * math.factorial(l - m) / \
        (4 * math.pi * math.factorial(l + m)))

def SH_closed_form(m, l, phi, theta):
    if m == 0:
        return SH_renormalization(l, m) * associated_legendre_polynomial(l, m, torch.cos(theta))
    elif m > 0:
        return math.sqrt(2.0) * SH_renormalization(l, m) * \
            torch.cos(m * phi) * associated_legendre_polynomial(l, m, torch.cos(theta))
    else:
        return math.sqrt(2.0) * SH_renormalization(l, -m) * \
            torch.sin(-m * phi) * associated_legendre_polynomial(l, -m, torch.cos(theta))

class SphericalHarmonics(nn.Module):
    def __init__(self, legendre_polys: int = 10):
        """
        legendre_polys: determines the number of legendre polynomials.
                        more polynomials lead more fine-grained resolutions
        calculation of spherical harmonics:
            analytic uses pre-computed equations. This is exact, but works only up to degree 50,
            closed-form uses one equation but is computationally slower (especially for high degrees)
        """
        super(SphericalHarmonics, self).__init__()
        self.L, self.M = int(legendre_polys), int(legendre_polys)
        self.embedding_dim = self.L * self.M

        self.SH = SH_closed_form

    def forward(self, ra_dec):
        ra, dec = ra_dec[:, 0], ra_dec[:, 1]
    
        # RA is already in degrees, so we convert it directly to radians.
        phi = torch.deg2rad(ra)
    
        # Convert Dec to radians
        theta = torch.deg2rad(dec + 90)
    
        Y = []
        for l in range(self.L):
            for m in range(-l, l + 1):
                y = self.SH(m, l, phi, theta)
                if isinstance(y, float):
                    y = y * torch.ones_like(phi)
                Y.append(y)
    
        return torch.stack(Y, dim=-1)

def get_neural_network(name, input_dim, num_classes=10, dim_hidden=32, num_layers=2, dropout=False):
    if name == "linear":
        return nn.Linear(input_dim, num_classes)
    elif name ==  "siren":
        return SirenNet(
                dim_in=input_dim,
                dim_hidden=dim_hidden,
                num_layers=num_layers,
                dim_out=num_classes,
                dropout=dropout
            )
    elif name == "fcnet":
        return FCNet(
                num_inputs=input_dim,
                num_classes=num_classes,
                dim_hidden=dim_hidden
            )
    else:
        raise ValueError(f"{name} not a known neural networks.")

class LocationEncoder(nn.Module):
    def __init__(self, neural_network_name, legendre_polys=10,
                 num_classes=10, dim_hidden=32, num_layers=2, dropout=False):
        super().__init__()

        self.positional_encoder = SphericalHarmonics(legendre_polys=legendre_polys)
        self.neural_network = get_neural_network(
            neural_network_name,
            input_dim=self.positional_encoder.embedding_dim,
            num_classes=num_classes, dim_hidden=dim_hidden, num_layers=num_layers, dropout=dropout
        )

    def forward(self, ra_dec):
        embedding = self.positional_encoder(ra_dec)
        return self.neural_network(embedding)

'''
model = LocationEncoder(neural_network_name="siren", 
                        legendre_polys=5,
                        dim_hidden=8,
                        num_layers=1,
                        num_classes=768)
'''