import torch


class myLinear(torch.autograd.Function):
    """
    This autograd function implements a Linear projection.
    """

    @staticmethod
    def forward(ctx, x, weights, bias):
        """
        The forward function.
        Compute x * weights + bias.

        Parameters
        ----------
        ctx: torch.autograd.function
            the context object in which to store stuff.
        x: torch.FloatTensor
            the input tensor.
        weights: torch.FloatTensor
            the matrix defining the linear projection.
        bias: torch.FloatTensor
            the bias term.

        Returns
        -------
        o: torch.FloatTensor
            the output tensor.
        """

        ctx.save_for_backward(x, weights, bias)

        o = x.mm(weights) + bias.unsqueeze(0)

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
            the gradient of the loss function with respect to the weights.
        grad_bias: torch.FloatTensor
            the gradient of the loss function with respect to the bias.
        """

        x, weights, bias = ctx.saved_tensors
        grad_x, grad_weights, grad_bias = None, None, None

        grad_x = grad_outputs.mm(weights.t())
        grad_weights = grad_outputs.t().mm(x).t()
        grad_bias = torch.sum(grad_outputs, dim=0)

        return grad_x, grad_weights, grad_bias
