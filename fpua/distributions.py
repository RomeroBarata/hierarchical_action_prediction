import torch


def sample_from_gumbel(shape, dtype=torch.float32, device='cpu', eps=1e-42):
    """Sample from a Gumbel(0, 1) distribution.

    Arg(s):
        shape - Shape of the sampled tensor.
        dtype - Data type of the sampled tensor.
        device - Device to put the sampled in.
        eps - Tolerance value to avoid numerical issues with the log function.
    Returns:
        A torch tensor with requested shape and dtype, containing samples from a Gumbel(0, 1) distribution.
    """
    uniform_sample = torch.rand(shape, dtype=dtype, device=device)
    return -torch.log(-torch.log(uniform_sample + eps) + eps)


def sigmoid_gumbel(logits, temperature=1.0):
    """Sigmoid version of the softmax_gumbel function.

    Arg(s):
        logits - A tensor of shape (batch_size, 1).
        temperature - Sigmoid temperature to approximate the results to the true binary distribution (temperature
            -> 0) or to smooth it out and make it uniform (temperature -> +Inf).
    Returns:
        A torch tensor of same shape as logits, containing the probabilities for each example.
    """
    logits = torch.cat((logits, torch.zeros_like(logits)), dim=-1)
    g = sample_from_gumbel(logits.size(), dtype=logits.dtype, device=logits.device)
    y = logits + g
    return torch.softmax(y / temperature, dim=-1)[:, 0:1]


def softmax_gumbel(logits, temperature=1.0):
    """Softmax function as a continuous approximation to the arg max function in the Gumbel-Max trick.

    Arg(s):
        logits - A tensor of shape (batch_size, number of categories). Number of categories must be > 1.
        temperature - Softmax temperature to approximate the results to the true categorical distribution
            (temperature -> 0) or to smooth it out the output and make it uniform (temperature -> +Inf).
    Returns:
        A torch tensor of the same shape as logits, containing the probabilities of each category.
    """
    g = sample_from_gumbel(logits.size(), dtype=logits.dtype, device=logits.device)
    y = logits + g
    return torch.softmax(y / temperature, dim=-1)


def straight_through_gumbel_sigmoid(logits, temperature=1.0, hard=True, both=False):
    """Straight-through estimator for binary variable using the Gumbel-Sigmoid distribution.

    Arg(s):
        logits - A tensor of shape (batch_size, 1).
        temperature - Sigmoid temperature to approximate the results to the true binary distribution
            (temperature -> 0) or to smooth it out the output and make it uniform (temperature -> +Inf).
        hard - Whether to return a hard estimation or a soft one.
        both - Return hard and soft estimates. If True, hard flag is ignored.
    Returns:
        A tensor of shape (batch_size, 1) containing the estimated hard or soft probabilities. If both is True, then
        return the hard and the soft estimates.
    """
    y = sigmoid_gumbel(logits, temperature=temperature)
    if both:
        z = (y > 0.5).float()
        z = (z - y).detach() + y
        return z, y
    else:
        if not hard:
            return y
        z = (y > 0.5).float()
        z = (z - y).detach() + y
        return z


def straight_through_gumbel_softmax(logits, temperature=1.0, hard=True):
    """Straight-through estimator for discrete variable using the Gumbel-Softmax distribution.

    This function sample values from a categorical distribution by using the Gumbel-Max trick with a softmax
    approximation for the arg max function. In the forward pass, it can either return the soft estimation by the
    Gumbel-Softmax or the hard estimation by applying arg max to the output of the Gumbel-Softmax. In either case,
    the graph is constructed in way that the backward pass always goes through the soft estimation, which is
    differentiable. The temperature parameter controls how good the continuous approximation is to the discrete
    representation. Temperatures -> 0.0 are a better approximation of the real categorical distribution but the
    gradients have higher variance; temperatures -> +Inf are worse approximations but the gradients have lower
    variance. A nice strategy is to start with a high temperature (e.g. 1.0) and anneal it to 0.0.

    Arg(s):
        logits - A tensor of shape (batch_size, number of categories). Number of categories must be > 1.
        temperature - Softmax temperature to approximate the results to the true categorical distribution
            (temperature -> 0) or to smooth it out the output and make it uniform (temperature -> +Inf).
        hard - Whether to return a hard estimation or a soft one.
    Returns:
        Either the smooth probability estimate of each category as computed by the Gumbel-Softmax function, or the
        one-hot encode of the probabilities. In either case, the computation graph is set up in way to always
        back propagate through the continuous estimate.
    """
    y = softmax_gumbel(logits, temperature=temperature)
    if not hard:
        return y
    z = torch.zeros_like(y)
    z.scatter_(dim=1, index=y.max(dim=-1)[1].view(-1, 1), value=1.0)
    z = (z - y).detach() + y  # Trick to force the backward pass to go through y (continuous) instead of z (discrete)
    return z
