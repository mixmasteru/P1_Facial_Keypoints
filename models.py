## TODO: define the convolutional neural network architecture

import torch.nn as nn
import torch.nn.functional as f


# can use the below import should you choose to initialize the weights of your Net


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # TODO: Define all the layers of this CNN, the only requirements are:
        # 1. This network takes in a square (same width and height), grayscale image as input
        # 2. It ends with a linear layer that represents the keypoints
        # it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.max_pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 32, 5)
        self.drop1 = nn.Dropout(0.1)

        self.conv2 = nn.Conv2d(32, 64, 3)
        self.drop2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(64, 128, 2)
        self.drop3 = nn.Dropout(0.3)

        self.conv4 = nn.Conv2d(128, 256, 1)
        self.drop4 = nn.Dropout(0.4)

        self.dense1 = nn.Linear(256 * 13 * 13, 256 * 13)
        self.drop5 = nn.Dropout(0.5)

        self.dense2 = nn.Linear(256 * 13, 256 * 13)
        self.drop6 = nn.Dropout(0.6)

        self.dense3 = nn.Linear(256 * 13, 136)

        # Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

    def forward(self, x):
        # TODO: Define the feedforward behavior of this model
        # x is the input image and, as an example, here you may choose to include a pool/conv step:
        # x = self.pool(F.relu(self.conv1(x)))

        # a modified x, having gone through all the layers of your model, should be returned

        x = self.max_pool(f.elu(self.conv1(x)))
        x = self.drop1(x)

        x = self.max_pool(f.elu(self.conv2(x)))
        x = self.drop2(x)

        x = self.max_pool(f.elu(self.conv3(x)))
        x = self.drop3(x)

        x = self.max_pool(f.elu(self.conv4(x)))
        x = self.drop4(x)

        # Flatten layer
        x = x.view(x.size(0), -1)

        x = f.elu(self.dense1(x))
        x = self.drop5(x)

        x = f.relu(self.dense2(x))
        x = self.drop6(x)

        x = self.dense3(x)

        return x
