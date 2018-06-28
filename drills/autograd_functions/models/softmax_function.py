import torch


class mySoftmax(torch.autograd.Function):
    """
    This autograd function implements a Softmax activation.
    """

    @staticmethod
    def forward(ctx, x, dim):
        """
        The forward function.
        Apply softmax to x along the dimension dim.

        Parameters
        ----------
        ctx: torch.autograd.function
            the context object in which to store stuff.
        x: torch.FloatTensor
            the input tensor.
        dim: int
            the dimension along which apply softmax.

        Returns
        -------
        o: torch.FloatTensor
            the ReLU activated tensor.
        """

        N = torch.exp(x)
        D = torch.sum(N, dim=dim, keepdim=True)

        o = N / D
        ctx.save_for_backward(o)

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
        grad_dim: None
        """

        output, = ctx.saved_tensors
        grad_x, grad_dim = None, None

        # Build identity matrix (useful later)
        _, d = output.shape
        I = torch.eye(d)
        if output.is_cuda:
            I = I.to('cuda')

        grad_x = []
        for o, grad in zip(output, grad_outputs):

            # Compute Jacobian for this batch example
            J = - torch.ger(o, o) * (1 - I) + torch.diag(o * (1-o))

            # Matrix-multiply
            grad_x.append(J.mm(grad.unsqueeze(1)).squeeze(1))
        grad_x = torch.stack(grad_x)

        return grad_x, grad_dim
