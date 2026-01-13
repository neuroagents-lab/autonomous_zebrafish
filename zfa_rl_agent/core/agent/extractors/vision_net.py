import torch 
import torch.nn as nn
import torch.nn.functional as F

class ConvReluNet(nn.Module):
    def __init__(self, output_size, training=True):
        super(ConvReluNet, self).__init__()

        # First Convolutional Layer followed by ReLU activation
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        # Second Convolutional Layer followed by ReLU activation
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        # Fully Connected Layer (for training)
        # after two conv layers with same padding, the size remains 25x25


        self.fc = nn.Linear(in_features=64*64*64, out_features=output_size)
        # State for adding the fully connected layer or not
        self.training_mode = training

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))        
        # If the network is in training mode, then add the dense layer
        if self.training_mode:
            x = x.reshape(x.size(0), -1)           
            x = self.fc(x)
        return x
    def _get_feature_size(self):
        return self.fc.out_features

class OpticFlowDecoder(nn.Module):
    def __init__(self, in_channels):
        super(OpticFlowDecoder, self).__init__()

        self.input_size = 64

        # First layer: input D=34 channels, output 8 channels
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=5, padding=2)
        self.batchnorm1 = nn.BatchNorm2d(8)

        # Second layer: input 8 channels, output 3 channels (optic flow + normalization)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=5, padding=2)
        self.batchnorm2 = nn.BatchNorm2d(3)

        # Activation function: Softplus
        self.softplus = nn.Softplus()

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=0.5)

        # Initialize weights homogeneously at 0.001
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.constant_(layer.weight, 0.001)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.001)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # First conv layer
        x = self.batchnorm1(self.conv1(x))
        x = self.softplus(x)
        x = self.dropout(x)

        # Second conv layer
        x = self.batchnorm2(self.conv2(x))
        x = self.softplus(x)
        x = self.dropout(x)

        # Split the output into the 2D flow (2 channels) and normalization (1 channel)
        flow = x[:, :2, :, :]  # First 2 channels are the flow (x and y)
        normalization = x[:, 2:3, :, :]  # Third channel is the normalization

        flow = flow / (normalization + 1e-8)

        # Scale
        scale_x = flow.size(3) / self.input_size
        scale_y = flow.size(2) / self.input_size

        # Resize flow to match original input size
        flow = F.interpolate(flow, size=self.input_size, mode='bilinear', align_corners=False)
        
        # Create scaling tensor for element-wise multiplication
        scale = torch.tensor([scale_x, scale_y], dtype=flow.dtype, device=flow.device).view(1, 2, 1, 1)
        # Apply scaling to flow vectors
        flow = flow * scale


        return flow

class Net_2_layers(nn.Module):
    def __init__(self, input_size):
        super(Net_2_layers, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # First residual block
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(64)

        # Second residual block
        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(128)
        
        # Adjust input channels to match the output channels (1x1 conv)
        self.adjust_channels = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0, bias=False)
        
        # OpticFlowDecoder
        self.optic_flow_decoder = OpticFlowDecoder(in_channels=128, input_size=input_size)

    def forward(self, x):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[1]
            x = torch.cat(x, dim=1)

        # First conv layer
        x = self.relu(self.bn1(self.conv1(x)))

        # First residual block
        residual = x  # Store residual
        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.bn2_2(self.conv2_2(x))
        x += residual  # Add skip connection
        x = self.relu(x)

        # Second residual block
        residual = x  # Store residual again
        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.bn3_2(self.conv3_2(x))
        
        # Adjust the dimensions of the residual if necessary
        residual = self.adjust_channels(residual)
        x += residual  # Add skip connection
        x = self.relu(x)

        # Pass through OpticFlowDecoder
        flow = self.optic_flow_decoder(x)

        return flow


class Net_3_layers(nn.Module):
    def __init__(self):
        super(Net_3_layers, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # First residual block
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(64)

        # Second residual block
        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(128)
        
        # Adjust input channels to match the output channels (1x1 conv)
        self.adjust_channels = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0, bias=False)
        
        # Third residual block
        self.conv4_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_2 = nn.BatchNorm2d(256)
        
        # Adjust input channels to match the output channels (1x1 conv)
        self.adjust_channels_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0, bias=False)
        self.flatten = nn.Flatten()
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        # First residual block
        residual = x  # Store residual
        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.bn2_2(self.conv2_2(x))
        x += residual  # Add skip connection
        x = self.relu(x)

        # Second residual block
        residual = x  # Store residual again
        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.bn3_2(self.conv3_2(x))
        
        # Adjust the dimensions of the residual if necessary
        residual = self.adjust_channels(residual)
        x += residual  # Add skip connection
        x = self.relu(x)

        # Third residual block
        residual = x
        x = self.relu(self.bn4_1(self.conv4_1(x)))
        x = self.bn4_2(self.conv4_2(x))
        
        # Adjust the dimensions of the residual if necessary
        residual = self.adjust_channels_2(residual)
        x += residual
        x = self.relu(x)
        x = self.flatten(x)
        return x
