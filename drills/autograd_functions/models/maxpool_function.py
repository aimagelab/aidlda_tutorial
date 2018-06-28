import torch
import torch.nn.functional as F


class myMaxpool(torch.autograd.Function):
    """
    This autograd function implements a MaxPooling function.
    """

    @staticmethod
    def forward(ctx, x, kernel_size, stride):
        """
        The forward function.
        Performs max pooling on input x with proper kernel_size and stride.

        Parameters
        ----------
        ctx: torch.autograd.function
            the context object in which to store stuff.
        x: torch.FloatTensor
            the input tensor.
        kernel_size: int or tuple
            the size of the kernel window.
        stride: int or tuple
            the stride of the kernel window.

        Returns
        -------
        o: torch.FloatTensor
            the ReLU activated tensor.
        """

        assert kernel_size in [2, (2, 2)]
        assert stride in [2, (2, 2)]

        o, indexes = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, return_indices=True)

        ctx.save_for_backward(indexes)

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
        grad_kernel_size: None
        grad_stride: None
        """

        indexes, = ctx.saved_tensors
        grad_x, grad_kernel_size, grad_stride = None, None, None

        b, c, h, w = grad_outputs.shape

        grad_x = torch.zeros(b, c, h*2, w*2)
        grad_x = grad_x.view(b, c, -1)

        if indexes.is_cuda:
            grad_x = grad_x.to('cuda')

        indexes = indexes.view(b, c, -1)

        grad_outputs = grad_outputs.view(b, c, -1)
        grad_x.scatter_(2, indexes, grad_outputs)
        grad_x = grad_x.view(b, c, h*2, w*2)

        return grad_x, grad_kernel_size, grad_stride
