import torch


class myRelu(torch.autograd.Function):
    """
    This autograd function implements a ReLU activation.
    """

    @staticmethod
    def forward(ctx, x):
        """
        The forward function.
        Zero out non positive entries of x.

        Parameters
        ----------
        ctx: torch.autograd.function
            the context object in which to store stuff.
        x: torch.FloatTensor
            the input tensor.

        Returns
        -------
        o: torch.FloatTensor
            the ReLU activated tensor.
        """

        ctx.save_for_backward(x)
        o = torch.clamp(x, min=0)
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
        """

        x, = ctx.saved_tensors

        grad_x = (x >= 0).float() * grad_outputs
        return grad_x
