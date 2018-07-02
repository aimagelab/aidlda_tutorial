import torch
import torch.nn as nn


class ConvNet(nn.Module):
    """
    A Convolutional Neural Network.
    """

    def __init__(self, n_classes=10):
        """
        Model constructor.

        Parameters
        ----------
        n_classes: int
            number of output classes.
        """

        super(ConvNet, self).__init__()

        # Initialize feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
        )

        # Initialize classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=(128 * 4 * 4), out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=n_classes),
            nn.Softmax(dim=1)
        )

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
        x = self.features(x)

        # Flatten out
        x = x.view(len(x), 128 * 4 * 4)

        # Classify
        o = self.classifier(x)

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
