import torch
import torch.nn.functional as F


def flip(x, dim):
    """
    Flips the input tensor along a given dimension.

    Parameters
    ----------
    x: torch.FloatTensor
        the input tensor.
    dim: int
        the dimension to flip.

    Returns
    -------
    torch.FloatTensor
        the flipped tensor.
    """

    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


class myConv(torch.autograd.Function):
    """
    This autograd function implements a 2d convolution.
    """

    @staticmethod
    def forward(ctx, x, weights, bias):
        """
        The forward function.
        Convolves x with weights and adds bias

        Parameters
        ----------
        ctx: torch.autograd.function
            the context object in which to store stuff.
        x: torch.FloatTensor
            the input tensor.
        weights: torch.FloatTensor
            the convolutional kernels.
        bias: torch.FloatTensor
            the bias term.

        Returns
        -------
        o: torch.FloatTensor
            the output of the convolution.
        """

        _, _, k_h, k_w = weights.shape
        p_h, p_w = k_h // 2, k_w // 2

        o = F.conv2d(x, weights, padding=(p_h, p_w)) + bias.view(1, -1, 1, 1)
        ctx.save_for_backward(x, weights, bias, o)

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
        grad_weights: torch.FloatTensor
            the gradient of the loss function with respect to weights.
        grad_bias: torch.FloatTensor
            the gradient of the loss function with respect to bias.
        """

        x, weights, bias, o = ctx.saved_tensors
        grad_x, grad_weights, grad_bias = None, None, None

        _, _, k_h, k_w = weights.shape
        p_h, p_w = k_h // 2, k_w // 2


        b, c_in, h, w = x.shape
        b, c_out, h, w = grad_outputs.shape

        # Gradient with respect to weights
        grad_weights = []
        for x_b, g_b in zip(x, grad_outputs):
            for g_o in g_b:
                for x_i in x_b:
                    grad_weights.append(F.conv2d(x_i.view(1, 1, h, w), g_o.view(1, 1, h, w), padding=(p_h, p_w)))

        grad_weights = torch.stack(grad_weights, dim=0)
        grad_weights = grad_weights.view(b, c_out, c_in, 3, 3)
        grad_weights = torch.sum(grad_weights, dim=0)

        # Gradient with respect to x
        rotated_weights = flip(weights, dim=2)
        rotated_weights = flip(rotated_weights, dim=3)

        rotated_weights = torch.transpose(rotated_weights, 1, 0)
        grad_x = F.conv2d(grad_outputs, rotated_weights, padding=(p_h, p_w))

        # Gradient with respect to bias
        grad_bias = torch.sum(grad_outputs, dim=0)
        grad_bias = torch.sum(grad_bias, dim=-1)
        grad_bias = torch.sum(grad_bias, dim=-1)

        return grad_x, grad_weights, grad_bias
