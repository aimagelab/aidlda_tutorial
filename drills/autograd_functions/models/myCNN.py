import torch
import torch.nn as nn
import torch.nn.init as init

from models.conv_function import myConv
from models.dropout_function import myDropout
from models.linear_function import myLinear
from models.maxpool_function import myMaxpool
from models.relu_function import myRelu
from models.softmax_function import mySoftmax


def get_linear_parameters(in_features, out_features):
    """
    Initializes parameters of a linear layer with xavier uniform.

    Parameters
    ----------
    in_features: int
        fan in of the layer.
    out_features: int
        fan out of the layer.

    Returns
    -------
    tuple
        weights: torch.nn.Parameter
            weights of the linear layer.
        bias: torch.nn.Parameter
            bias of the linear layer.
    """

    weights = nn.Parameter(torch.empty(in_features, out_features))
    bias = nn.Parameter(torch.empty(out_features))

    init.xavier_uniform_(weights)
    init.constant_(bias, 0.)

    return weights, bias


def get_conv_parameters(in_channels, out_channels, kernel_size):
    """
    Initializes parameters of a convolutional layer with xavier normal.

    Parameters
    ----------
    in_channels: int
        fan in of the layer.
    out_channels: int
        fan out of the layer.
    kernel_size: tuple
        the kernel size.

    Returns
    -------
    tuple
        weights: torch.nn.Parameter
            weights of the convolutional layer.
        bias: torch.nn.Parameter
            bias of the convolutional layer.
    """

    k_h, k_w  = kernel_size

    weights = nn.Parameter(torch.empty(out_channels, in_channels, k_h, k_w))
    bias = nn.Parameter(torch.empty(out_channels))

    init.xavier_normal_(weights)
    init.constant_(bias, 0.)

    return weights, bias


class ConvNet(nn.Module):
    """
    A Convolutional Neural Network.
    """

    def __init__(self, n_classes):
        """
        Model constructor.

        Parameters
        ----------
        n_classes: int
            number of output classes.
        """

        super(ConvNet, self).__init__()

        self.n_classes = n_classes

        self.init_fc_layers()
        self.init_conv_layers()

        self.conv = myConv.apply
        self.linear = myLinear.apply
        self.relu = myRelu.apply
        self.dropout = myDropout.apply
        self.softmax = mySoftmax.apply
        self.maxpool = myMaxpool.apply

    def init_fc_layers(self):

        # Create random weights
        fc1_w, fc1_b = get_linear_parameters(in_features=(15 * 4 * 4), out_features=512)
        fc2_w, fc2_b = get_linear_parameters(in_features=512, out_features=256)
        fc3_w, fc3_b = get_linear_parameters(in_features=256, out_features=self.n_classes)

        # Register
        self.register_parameter('fc1_w', fc1_w)
        self.register_parameter('fc1_b', fc1_b)
        self.register_parameter('fc2_w', fc2_w)
        self.register_parameter('fc2_b', fc2_b)
        self.register_parameter('fc3_w', fc3_w)
        self.register_parameter('fc3_b', fc3_b)

    def init_conv_layers(self):

        # Create random weights
        conv1_1_w, conv1_1_b = get_conv_parameters(in_channels=3, out_channels=5, kernel_size=(3, 3))
        conv1_2_w, conv1_2_b = get_conv_parameters(in_channels=5, out_channels=5, kernel_size=(3, 3))
        conv1_3_w, conv1_3_b = get_conv_parameters(in_channels=5, out_channels=5, kernel_size=(3, 3))
        conv2_1_w, conv2_1_b = get_conv_parameters(in_channels=5, out_channels=10, kernel_size=(3, 3))
        conv2_2_w, conv2_2_b = get_conv_parameters(in_channels=10, out_channels=10, kernel_size=(3, 3))
        conv2_3_w, conv2_3_b = get_conv_parameters(in_channels=10, out_channels=10, kernel_size=(3, 3))
        conv3_1_w, conv3_1_b = get_conv_parameters(in_channels=10, out_channels=15, kernel_size=(3, 3))
        conv3_2_w, conv3_2_b = get_conv_parameters(in_channels=15, out_channels=15, kernel_size=(3, 3))
        conv3_3_w, conv3_3_b = get_conv_parameters(in_channels=15, out_channels=15, kernel_size=(3, 3))

        self.register_parameter('conv1_1_w', conv1_1_w)
        self.register_parameter('conv1_1_b', conv1_1_b)
        self.register_parameter('conv1_2_w', conv1_2_w)
        self.register_parameter('conv1_2_b', conv1_2_b)
        self.register_parameter('conv1_3_w', conv1_3_w)
        self.register_parameter('conv1_3_b', conv1_3_b)
        self.register_parameter('conv2_1_w', conv2_1_w)
        self.register_parameter('conv2_1_b', conv2_1_b)
        self.register_parameter('conv2_2_w', conv2_2_w)
        self.register_parameter('conv2_2_b', conv2_2_b)
        self.register_parameter('conv2_3_w', conv2_3_w)
        self.register_parameter('conv2_3_b', conv2_3_b)
        self.register_parameter('conv3_1_w', conv3_1_w)
        self.register_parameter('conv3_1_b', conv3_1_b)
        self.register_parameter('conv3_2_w', conv3_2_w)
        self.register_parameter('conv3_2_b', conv3_2_b)
        self.register_parameter('conv3_3_w', conv3_3_w)
        self.register_parameter('conv3_3_b', conv3_3_b)


    def forward(self, x):
        """
        Forward function.

        Parameters
        ----------
        x: torch.FloatTensor
            a pytorch tensor having shape (batchsize, c, h, w).

        Returns
        -------
        o: torch.FloatTensor
            a pytorch tensor having shape (batchsize, n_classes).
        """


        # Extract Features
        h = x
        h = self.relu(self.conv(h, self.conv1_1_w, self.conv1_1_b))
        h = self.relu(self.conv(h, self.conv1_2_w, self.conv1_2_b))
        h = self.relu(self.conv(h, self.conv1_3_w, self.conv1_3_b))
        h = self.maxpool(h, 2, 2)
        h = self.relu(self.conv(h, self.conv2_1_w, self.conv2_1_b))
        h = self.relu(self.conv(h, self.conv2_2_w, self.conv2_2_b))
        h = self.relu(self.conv(h, self.conv2_3_w, self.conv2_3_b))
        h = self.maxpool(h, 2, 2)
        h = self.relu(self.conv(h, self.conv3_1_w, self.conv3_1_b))
        h = self.relu(self.conv(h, self.conv3_2_w, self.conv3_2_b))
        h = self.relu(self.conv(h, self.conv3_3_w, self.conv3_3_b))
        h = self.maxpool(h, 2, 2)

        # Flatten out
        h = h.view(len(x), 15 * 4 * 4)

        # Classify
        h = self.dropout(h, 0.5, self.training)
        h = self.relu(self.linear(h, self.fc1_w, self.fc1_b))
        h = self.dropout(h, 0.5, self.training)
        h = self.relu(self.linear(h, self.fc2_w, self.fc2_b))
        h = self.softmax(self.linear(h, self.fc3_w, self.fc3_b), 1)

        o = h

        return o


def crossentropy_loss(y_true, y_pred):
    """
    Crossentropy loss function for classification.

    Parameters
    ----------
    y_true: torch.LongTensor
        tensor holding groundtruth labels. Has shape (batchsize,).
    y_pred: torch.FloatTensor
        tensor holding model predictions. Has shape (batchsize, n_classes).

    Returns
    -------
    ce: torch.FloatTensor
        loss function value.
    """

    ce = - torch.gather(torch.log(y_pred + 1e-5), 1, y_true.unsqueeze(1))
    return torch.mean(ce)
