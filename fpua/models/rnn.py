import collections.abc
import math
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F

from fpua.distributions import straight_through_gumbel_sigmoid
from fpua.models.misc import straight_through_estimator, hard_sigmoid


class HMGRUCell(nn.Module):
    """Hierarchical Multiscale GRU cell."""
    def __init__(self, input_size, bottom_size, hidden_size, top_size=None, boundary_is_known=True,
                 update_boundary_method='straight-through', weight_initialisation='pytorch', bias=True):
        super(HMGRUCell, self).__init__()
        self.bias = bias
        self.is_last_layer = True if top_size is None else False
        self.bottom_size = bottom_size
        self.hidden_size = hidden_size
        self.top_size = top_size
        self.boundary_is_known = boundary_is_known
        self.update_boundary_method = update_boundary_method
        # Parameters
        self._create_parameters(input_size, bottom_size, hidden_size, top_size, name='z')
        self._create_parameters(input_size, bottom_size, hidden_size, top_size, name='r')
        self._create_parameters(input_size, bottom_size, hidden_size, top_size, name='g')
        W_boundary = None if self.boundary_is_known else nn.Parameter(torch.FloatTensor(hidden_size, 1))
        self.register_parameter('W_boundary', W_boundary)
        bias_boundary = nn.Parameter(torch.FloatTensor(1)) if self.bias else None
        self.register_parameter('bias_boundary', bias_boundary)
        self._initialise_parameters(hidden_size, weight_initialisation=weight_initialisation)

    def forward(self, x, h_bottom=None, h=None, h_top=None, d_bottom=None, d=None, temperature=1.0, slope=1.0):
        """Process input through the HMGRUCell.

        The hidden state for the current time step is computed using equations similar to the original GRU formulation
        with a few twists.

        First, there are three 'operations' that can happen: COPY, FLUSH, and UPDATE. The COPY
        operation copies the previous hidden state (i.e. h_{t} = h_{t-1}) and occurs when d_bottom = d = 0. The FLUSH
        operation throws away the previous hidden state (h_{t-1}) and computes h_{t} based only on the candidate hidden
        state (i.e. h_{t} = (1 - z) * g, where g is the candidate hidden state). The FLUSH operations occurs when d = 1
        (i.e. the previous time step identified a boundary and now we should restart our hidden state). The UPDATE
        operation is similar to the usual hidden state update operation in a standard GRU and happens when
        d_bottom = 1 and d = 0. Note that a FLUSH operation at the current time step, means that d = 1, which in turn
        means that the d_bottom of the layer above at the previous time step was 1, which means that h_{t-1} was
        incorporated in the hidden state of the layer above at the previous time step before we threw it away.

        Second, the computation of z, r, and g, are slightly modified to incorporate or ignore the hidden state
        of the layer below at the current time step and the hidden state of the layer above at the previous time step.
        Incorporating the hidden state from the layer below means that we can 'save' it before the layer below performs
        a FLUSH. Incorporating the hidden state from the layer above at the previous time step means that we 'recover'
        information that we threw away at the previous time step due to a FLUSH.

        Due to the batched nature of this function, all GRU equations must be computed even when the operation is
        a COPY or a FLUSH.

        Arg(s):
            x - Input tensor at the current time step of shape (batch_size, input_size).
            h_bottom - Hidden state of the layer below at the current time step, represented as a tensor of
                shape (batch_size, bottom_size).
            h - Hidden state of the current layer at the previous time step, represented as a tensor of
                shape (batch_size, hidden_size).
            h_top - Hidden state of the layer above at the previous time step, represented as a tensor of
                shape (batch_size, top_size).
            d_bottom - Boundary detector of the layer below at the current time step, either represented as a tensor
                of shape (batch_size, 1) or a scalar.
            d - Boundary detector of the current layer at the previous time step, either represented as a
                tensor of shape (batch_size, 1) or a scalar.
            temperature - Temperature for the computation of the Gumbel-Softmax when estimating the boundary at
                the current step. Only meaningful if the boundary between the steps is not known (i.e.
                self.boundary_is_known is False) and the method selected for computation of the boundary
                is 'gumbel-sigmoid'.
            slope - Slope for computation of the hard sigmoid function when estimating the boundary at the current
                step. Only meaningful if the boundary between the steps is not known (i.e. self.boundary_is_known
                is False) and the method selected for computation of the boundary is 'straight-through'.
        Returns:
            The hidden state of the cell at the current time step (tensor of shape (batch_size, hidden_state)),
            the hard boundary value at the current time step (tensor of shape (batch_size, 1)), and the soft
            boundary value at the current time step (tensor of shape (batch_size, 1)). If the boundary between
            the steps is known (i.e. self.boundary_is_known is True), the returned boundary values are tensors
            of zeros.
        """
        batch_size, dtype, device = x.size(0), x.dtype, x.device
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, dtype=dtype, device=device)
        if h_bottom is None:
            h_bottom = torch.zeros(batch_size, self.bottom_size, dtype=dtype, device=device)
        if h_top is None and not self.is_last_layer:
            h_top = torch.zeros(batch_size, self.top_size, dtype=dtype, device=device)
        d_bottom = d_bottom if d_bottom is not None else 1
        d = d if d is not None else 0

        if self.is_last_layer:
            z_top, r_top, g_top = 0.0, 0.0, 0.0
        else:
            z_top, r_top, g_top = d * (h_top @ self.Wz_top), d * (h_top @ self.Wr_top), d * (h_top @ self.Wg_top)
        if self.bias:
            z_bias, r_bias, g_bias = self.bias_z, self.bias_r, self.bias_g
        else:
            z_bias, r_bias, g_bias = 0.0, 0.0, 0.0
        z = torch.sigmoid(x @ self.Wz_input + d_bottom * (h_bottom @ self.Wz_bottom) + h @ self.Uz + z_top + z_bias)
        r = torch.sigmoid(x @ self.Wr_input + d_bottom * (h_bottom @ self.Wr_bottom) + h @ self.Ur + r_top + r_bias)
        g = torch.tanh(x @ self.Wg_input + d_bottom * (h_bottom @ self.Wg_bottom) + (r * h) @ self.Ug + g_top + g_bias)

        # This equation correctly selects COPY, FLUSH, or UPDATE in a batched way.
        h = (1 - d) * (1 - d_bottom) * h + d * (1 - z) * g + (1 - d) * d_bottom * (z * h + (1 - z) * g)
        if self.boundary_is_known:
            d = torch.zeros_like(h)[:, -1:]
            d_soft = torch.zeros_like(h)[:, -1:]
        else:
            d, d_soft = self._update_boundary(h, slope, temperature)
        return h, d, d_soft

    def _update_boundary(self, h, slope=1.0, temperature=1.0):
        d = h @ self.W_boundary
        d = d + self.bias_boundary if self.bias else d
        if self.update_boundary_method == 'straight-through':
            d_soft = hard_sigmoid(d, a=slope)
            d = straight_through_estimator(d_soft)
        elif self.update_boundary_method == 'gumbel-sigmoid':
            d = F.logsigmoid(d)
            d, d_soft = straight_through_gumbel_sigmoid(d, temperature=temperature, both=True)
        else:
            raise ValueError('update_boundary_method must be either \'straight-through\' or \'gumbel-sigmoid\'.')
        return d, d_soft

    def _initialise_parameters(self, hidden_size, weight_initialisation='pytorch'):
        if weight_initialisation == 'pytorch':
            self._initialise_parameters_pytorch_style(hidden_size)
        elif weight_initialisation == 'keras':
            self._initialise_parameters_keras_style()

    def _initialise_parameters_pytorch_style(self, hidden_size):
        stddev = 1 / math.sqrt(hidden_size)
        for parameter in self.parameters():
            parameter.data.uniform_(-stddev, stddev)

    def _initialise_parameters_keras_style(self):
        for name, parameter in self.named_parameters():
            if name.startswith('W'):
                torch.nn.init.xavier_uniform_(parameter.data, gain=1.0)
            elif name.startswith('U'):
                torch.nn.init.orthogonal_(parameter.data, gain=1)
            else:
                torch.nn.init.zeros_(parameter.data)

    def _create_parameters(self, input_size, bottom_size, hidden_size, top_size, name):
        # Input
        W_input_name, W_input = 'W' + name + '_input', nn.Parameter(torch.FloatTensor(input_size, hidden_size))
        self.register_parameter(W_input_name, W_input)
        # Bottom
        W_bottom_name, W_bottom = 'W' + name + '_bottom', nn.Parameter(torch.FloatTensor(bottom_size, hidden_size))
        self.register_parameter(W_bottom_name, W_bottom)
        # Recurrent
        U_name, U = 'U' + name, nn.Parameter(torch.FloatTensor(hidden_size, hidden_size))
        self.register_parameter(U_name, U)
        # Top
        W_top_name = 'W' + name + '_top'
        W_top = nn.Parameter(torch.FloatTensor(top_size, hidden_size)) if not self.is_last_layer else None
        self.register_parameter(W_top_name, W_top)
        # Bias
        bias_name = 'bias_' + name
        bias = nn.Parameter(torch.FloatTensor(hidden_size)) if self.bias else None
        self.register_parameter(bias_name, bias)


class HMGRU(nn.Module):
    """Hierarchical Multiscale GRU.

    This module is a GRU adaptation of the Hierarchical Multiscale LSTM proposed by Chung et al. [1]. In addition
    to the use of a GRU instead of an LSTM, this implementation also adds an additional input tensor to each cell. For
    the first layer it avoids passing the input tensor through the bottom hidden state, and for deeper layers it is
    an additional feature. If there are no input tensors to be passed to deeper layers, it is still necessary to
    pass a tensor of zeroes.

    [1] Chung, J., Ahn, S., & Bengio, Y. (2016). Hierarchical Multiscale Recurrent Neural Networks. In 5th
    International Conference on Learning Representations.

    Arg(s):
        input_sizes - List containing the input size of each hidden layer of the network.
        hidden_size - Either a list of integers containing the hidden size of each hidden layer of the network or
            a single integer specifying the same hidden size for each hidden layer of the network.
        known_boundaries - List specifying whether the boundaries for each layer are known or not. If None, it assumes
            that the boundaries for each layer are known. It makes more sense for None to represent the absence of
            information, but to keep backwards compatibility with older models, None represents the knowledge. This
            might change in future.
        update_boundary_method - How to update the boundaries in case the boundaries are unknown. Either
            'straight-through' or 'gumbel-sigmoid'.
        bias - Whether to include a bias term or not.
    """
    def __init__(self, input_sizes, hidden_size, known_boundaries=None, update_boundary_method='straight-through',
                 weight_initialisation='pytorch', bias=True):
        super(HMGRU, self).__init__()
        self.num_layers = len(input_sizes)
        if not isinstance(hidden_size, collections.abc.Sequence):
            hidden_size = [hidden_size] * self.num_layers
        self.hidden_sizes = list(hidden_size)
        self.known_boundaries = [True] * self.num_layers if known_boundaries is None else known_boundaries
        self.cells = nn.ModuleList()
        for layer, (input_size, hidden_size, boundary_is_known) in \
                enumerate(zip(input_sizes, self.hidden_sizes, self.known_boundaries)):
            top_size = None if layer == (self.num_layers - 1) else self.hidden_sizes[layer + 1]
            bottom_size = 1 if layer == 0 else self.hidden_sizes[layer - 1]
            cell = HMGRUCell(input_size, bottom_size=bottom_size, hidden_size=hidden_size, top_size=top_size,
                             boundary_is_known=boundary_is_known, update_boundary_method=update_boundary_method,
                             weight_initialisation=weight_initialisation, bias=bias)
            self.cells.append(cell)

    def forward(self, xs, hx=None, dx=None, temperature=1.0, slope=1.0):
        """Process input through the HMGRU for a number of time steps.

        For the HMGRU cell, the output coincides with the hidden state. Therefore, we accumulate the hidden states in
        the output list and copy the hidden states at the last time step to the hx_out variable. We do that in order
        to mimic the output of PyTorch's GRU module.

        Arg(s):
            xs - List of input tensors, where each input tensor is of shape (batch_size, seq_len, input_size*). Note
                that the input_size can vary for each input in the list. Also, for layers without an actual
                input tensor to be passed, a tensor of zeroes must be passed (i.e. be included in the list passed).
            hx - Either a list of tensors containing the initial hidden state of each layer of the network or None. If
                a list of tensors, each tensor in the list is of shape (batch_size, hidden_size*). If None, the
                initial hidden state of each layer is initialised to zero.
            dx - Tensor containing the boundary information of each layer. It is a tensor of shape (batch_size,
                num_layers, seq_len). dx must contain values for every layer, but for the layers that the boundaries
                are unknown, these values are ignored.
            temperature - Temperature for the computation of the Gumbel-Softmax when estimating the boundary at
                the current step. Only meaningful if the boundary between the steps is not known (i.e.
                self.known_boundaries[layer] is False) and the method selected for computation of the boundary is
                'gumbel-sigmoid'.
            slope - Slope for computation of the hard sigmoid function when estimating the boundary at the current
                step. Only meaningful if the boundary between the steps is not known (i.e. self.boundary_is_known
                is False) and the method selected for computation of the boundary is 'straight-through'.
        Returns:
            A list of tensors of length num_layers, where each tensor is of shape (batch_size, seq_len, hidden_size*)
            containing the output of each hidden layer at each time step; a list of tensors of length num_layers where
            each tensor is of shape (batch_size, hidden_size) containing the hidden states of all layers at the last
            time step; a tensor of shape (batch_size, num_layers, seq_len) containing the hard boundaries of each
            layer at each time step, and another tensor of shape (batch_size, num_layers, seq_len) containing the
            soft boundaries.
        """
        assert len(xs) == self.num_layers, 'Input tensors must match the number of hidden layers.'
        if hx is not None:
            assert len(hx) == self.num_layers, 'Initial hidden states must match the number of hidden layers.'
        seq_len = xs[0].size(1)
        output, ds, ds_soft = [], [], []
        for t in range(seq_len):
            output.append([])
            ds.append([])
            ds_soft.append([])
            for layer, (boundary_is_known, cell) in enumerate(zip(self.known_boundaries, self.cells)):
                x = xs[layer][:, t, :]
                h_bottom = None if layer == 0 else output[t][layer - 1]
                if t == 0:
                    if hx is None:
                        h = h_top = None
                    else:
                        h = hx[layer]
                        h_top = None if layer == (self.num_layers - 1) else hx[layer + 1]

                    d_bottom = d = None
                    if layer > 0:
                        boundary_layer_below_is_known = self.known_boundaries[layer - 1]
                        d_bottom = dx[:, layer - 1, t:t + 1] if boundary_layer_below_is_known else None
                else:
                    h = output[t - 1][layer]
                    h_top = None if layer == (self.num_layers - 1) else output[t - 1][layer + 1]
                    d = dx[:, layer, t - 1:t] if boundary_is_known else ds[t - 1][layer]
                    if layer > 0:
                        boundary_layer_below_is_known = self.known_boundaries[layer - 1]
                        d_bottom = dx[:, layer - 1, t:t + 1] if boundary_layer_below_is_known else ds[t][layer - 1]
                    else:
                        d_bottom = 1
                h, d, d_soft = cell(x, h_bottom, h, h_top, d_bottom, d, temperature, slope)
                output[t].append(h)
                ds[t].append(d)
                ds_soft[t].append(d_soft)
        output = [torch.stack([out_t[layer] for out_t in output], dim=1) for layer in range(self.num_layers)]
        hx_out = [out_l[:, -1, :] for out_l in output]
        ds_per_time_step = [torch.stack(time_step_out, dim=1) for time_step_out in ds]
        ds = torch.stack(ds_per_time_step, dim=2).squeeze()
        ds_soft_per_time_step = [torch.stack(time_step_out, dim=1) for time_step_out in ds_soft]
        ds_soft = torch.stack(ds_soft_per_time_step, dim=2).squeeze()
        return output, hx_out, ds, ds_soft


class HMGRUCellV2(nn.Module):
    def __init__(self, input_size, hidden_size, bottom_size=None, top_size=None, reset_after_flush=True,
                 always_include_parent_state=False, bias=True):
        super(HMGRUCellV2, self).__init__()
        self.hidden_size = hidden_size
        self.bottom_size = bottom_size
        self.top_size = top_size
        self.reset_after_flush = reset_after_flush
        self.always_include_parent_state = always_include_parent_state
        if bottom_size is not None:
            input_size += bottom_size
        if top_size is not None:
            input_size += top_size
        self.cell = nn.GRUCell(input_size=input_size, hidden_size=hidden_size, bias=bias)

    def forward(self, x, h=None, h_bottom=None, h_top=None, d_bottom=None, d=None):
        x, h, copy = self._preprocess_input(x, h=h, h_bottom=h_bottom, h_top=h_top, d_bottom=d_bottom, d=d)
        h = copy * h + (torch.ones_like(copy) - copy) * self.cell(x, hx=h)
        return h, None, None  # Mimic HMGRUCell return

    def _preprocess_input(self, x, h=None, h_bottom=None, h_top=None, d_bottom=None, d=None):
        batch_size, dtype, device = x.size(0), x.dtype, x.device
        if d_bottom is None:
            d_bottom = torch.ones(batch_size, 1, dtype=dtype, device=device)
        elif isinstance(d_bottom, numbers.Real):
            d_bottom = torch.full([batch_size, 1], fill_value=d_bottom, dtype=dtype, device=device)
        x = x * d_bottom
        if self.bottom_size is not None:
            if h_bottom is None:
                h_bottom = torch.zeros(batch_size, self.bottom_size, dtype=dtype, device=device)
            x = torch.cat([x, h_bottom * d_bottom], dim=-1)
        if d is None:
            d = torch.zeros(batch_size, 1, dtype=dtype, device=device)
        elif isinstance(d, numbers.Real):
            d = torch.full([batch_size, 1], fill_value=d, dtype=dtype, device=device)
        if self.top_size is not None:
            if h_top is None:
                h_top = torch.zeros(batch_size, self.top_size, dtype=dtype, device=device)
            if self.always_include_parent_state:
                x = torch.cat([x, h_top], dim=-1)
            else:
                x = torch.cat([x, h_top * d], dim=-1)
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, dtype=dtype, device=device)
        if self.reset_after_flush:
            h = h * (torch.ones_like(d) - d)
        copy = ((d_bottom + d) == 0.0).type(torch.float32)
        return x, h, copy


class HMGRUV2(nn.Module):
    def __init__(self, input_sizes, hidden_sizes, reset_after_flush=True, always_include_parent_state=False, bias=True):
        super(HMGRUV2, self).__init__()
        self.cells = nn.ModuleList()
        for layer, (input_size, hidden_size) in enumerate(zip(input_sizes, hidden_sizes)):
            bottom_size = None if layer == 0 else hidden_sizes[layer - 1]
            top_size = None if layer == len(hidden_sizes) - 1 else hidden_sizes[layer + 1]
            cell = HMGRUCellV2(input_size, hidden_size, bottom_size=bottom_size, top_size=top_size,
                               reset_after_flush=reset_after_flush,
                               always_include_parent_state=always_include_parent_state, bias=bias)
            self.cells.append(cell)

    def forward(self, x, hx=None, dx=None, dx_layer_zero=None, disable_gradient_from_child=False):
        output = []
        time_steps, num_layers = x[0].size(1), len(x)
        for t in range(time_steps):
            output.append([])
            for layer in range(num_layers):
                if t:
                    h = output[t - 1][layer]
                    h_bottom = output[t][layer - 1] if layer else None
                    h_top = output[t - 1][layer + 1] if layer + 1 < num_layers else None
                else:
                    h = hx[layer] if hx is not None else None
                    h_bottom = output[t][layer - 1] if output[t] else None
                    h_top = hx[layer + 1] if hx is not None and layer + 1 < num_layers else None
                if disable_gradient_from_child and h_top is not None:
                    h_top = h_top.detach()
                if layer:
                    d_bottom = dx[:, layer - 1, t:t + 1] if dx is not None else None
                else:
                    d_bottom = dx_layer_zero[:, t:t + 1] if dx_layer_zero is not None else None
                d = dx[:, layer, t - 1:t] if t and dx is not None else None
                h, _, _ = self.cells[layer](x[layer][:, t], h=h, h_bottom=h_bottom, h_top=h_top, d_bottom=d_bottom, d=d)
                output[t].append(h)
        output = [torch.stack([out_t[layer] for out_t in output], dim=1) for layer in range(num_layers)]
        hx = [out_l[:, -1, :] for out_l in output]
        return output, hx, None, None  # Mimic HMGRU return


class HMLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bottom_size=None, top_size=None, bias=True):
        super(HMLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.bottom_size = bottom_size
        self.top_size = top_size
        if bottom_size is not None:
            input_size += bottom_size
        if top_size is not None:
            input_size += top_size
        self.cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size, bias=bias)

    def forward(self, x, h=None, h_bottom=None, h_top=None, d_bottom=None, d=None):
        x, h, copy = self._preprocess_input(x, h, h_bottom, h_top, d_bottom, d)
        h_, c_ = self.cell(x, hx=h)
        h[0] = copy * h[0] + (torch.ones_like(copy) - copy) * h_
        h[1] = copy * h[1] + (torch.ones_like(copy) - copy) * c_
        return h, None, None  # Mimic HMGRUCell return

    def _preprocess_input(self, x, h, h_bottom=None, h_top=None, d_bottom=None, d=None):
        batch_size, dtype, device = x.size(0), x.dtype, x.device
        if d_bottom is None:
            d_bottom = torch.ones(batch_size, 1, dtype=dtype, device=device)
        elif isinstance(d_bottom, numbers.Real):
            d_bottom = torch.full([batch_size, 1], fill_value=d_bottom, dtype=dtype, device=device)
        x = x * d_bottom
        if self.bottom_size is not None:
            if h_bottom is None:
                h_bottom = torch.zeros(batch_size, self.bottom_size, dtype=dtype, device=device)
            x = torch.cat([x, h_bottom * d_bottom], dim=-1)
        if d is None:
            d = torch.zeros(batch_size, 1, dtype=dtype, device=device)
        elif isinstance(d, numbers.Real):
            d = torch.full([batch_size, 1], fill_value=d, dtype=dtype, device=device)
        if self.top_size is not None:
            if h_top is None:
                h_top = torch.zeros(batch_size, self.top_size, dtype=dtype, device=device)
            x = torch.cat([x, h_top * d], dim=-1)
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, dtype=dtype, device=device)
            c = torch.zeros(batch_size, self.hidden_size, dtype=dtype, device=device)
            h = [h, c]
        h[1] = h[1] * (torch.ones_like(d) - d)
        copy = ((d_bottom + d) == 0.0).type(torch.float32)
        return x, h, copy


class HMLSTM(nn.Module):
    def __init__(self, input_sizes, hidden_sizes, bias=True):
        super(HMLSTM, self).__init__()
        self.cells = nn.ModuleList()
        for layer, (input_size, hidden_size) in enumerate(zip(input_sizes, hidden_sizes)):
            bottom_size = None if layer == 0 else hidden_sizes[layer - 1]
            top_size = None if layer == len(hidden_sizes) - 1 else hidden_sizes[layer + 1]
            cell = HMLSTMCell(input_size, hidden_size, bottom_size=bottom_size, top_size=top_size, bias=bias)
            self.cells.append(cell)

    def forward(self, x, hx, dx, dx_layer_zero=None, disable_gradient_from_child=False):
        output_h, output_c = [], []
        time_steps, num_layers = x[0].size(1), len(x)
        for t in range(time_steps):
            output_h.append([])
            output_c.append([])
            for layer in range(num_layers):
                if t:
                    h = [output_h[t - 1][layer], output_c[t - 1][layer]]
                    h_bottom = output_h[t][layer - 1] if layer else None
                    h_top = output_h[t - 1][layer + 1] if layer + 1 < num_layers else None
                else:
                    h = hx[layer] if hx is not None else None
                    h_bottom = output_h[t][layer - 1] if output_h[t] else None
                    h_top = hx[layer + 1][0] if hx is not None and layer + 1 < num_layers else None
                if disable_gradient_from_child and h_top is not None:
                    h_top = h_top.detach()
                if layer:
                    d_bottom = dx[:, layer - 1, t:t + 1]
                else:
                    d_bottom = dx_layer_zero[:, t:t + 1] if dx_layer_zero is not None else None
                d = dx[:, layer, t - 1:t] if t else None
                hx_, _, _ = self.cells[layer](x[layer][:, t], h=h, h_bottom=h_bottom, h_top=h_top,
                                              d_bottom=d_bottom, d=d)
                output_h[t].append(hx_[0])
                output_c[t].append(hx_[1])
        output = [torch.stack([out_h_t[layer] for out_h_t in output_h], dim=1) for layer in range(num_layers)]
        hx = [[h, c] for h, c in zip([out_l[:, -1, :] for out_l in output], output_c[-1])]
        return output, hx, None, None  # Mimic HMGRU return
