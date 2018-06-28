import torch
from torch.distributions import Bernoulli


class myDropout(torch.autograd.Function):
    """
    This autograd function implements a Dropout regularization.
    """

    @staticmethod
    def forward(ctx, x, p, is_training):
        """
        The forward function.
        Zero out entries of x with probability p.

        Parameters
        ----------
        ctx: torch.autograd.function
            the context object in which to store stuff.
        x: torch.FloatTensor
            the input tensor.
        p: float
            the probability of zeroing out elements.
        is_training: bool
            whether the model is in training mode.

        Returns
        -------
        o: torch.FloatTensor
            the ReLU activated tensor.
        """

        if is_training:
            mask = Bernoulli(1 - p).sample(x.shape)
            mask = mask * (1 / (1 - p))
        else:
            mask = Bernoulli(1).sample(x.shape)

        if x.is_cuda:
            mask = mask.to('cuda')

        ctx.save_for_backward(mask)

        o = x * mask
        return o

    @staticmethod
    def backward(ctx, grad_outputs):
        """
        The backward function.

        Parameters
        ----------
        ctx: torch.autograd.function
            the context object with stored stuff.
        grad_outputs: torch.FloatTensor
            the gradient of the loss function with respect to the output of this function.

        Returns
        -------
        grad_x: torch.FloatTensor
            the gradient of the loss function with respect to the input of this function.
        grad_p: None
        grad_is_training: None
        """

        mask, = ctx.saved_tensors
        grad_x, grad_p, grad_is_training = None, None, None

        grad_x = grad_outputs * mask

        return grad_x, grad_p, grad_is_training
