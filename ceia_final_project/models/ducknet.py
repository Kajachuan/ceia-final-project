import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, filters, block_type, repeat=1, dilation_rate=1, size=3, padding='same'):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(repeat):
            if block_type == 'separated':
                self.blocks.append(SeparatedConv2DBlock(in_channels, filters, size, padding))
            elif block_type == 'duckv2':
                self.blocks.append(Duckv2Conv2DBlock(in_channels, filters, size))
            elif block_type == 'midscope':
                self.blocks.append(MidscopeConv2DBlock(in_channels, filters))
            elif block_type == 'widescope':
                self.blocks.append(WidescopeConv2DBlock(in_channels, filters))
            elif block_type == 'resnet':
                self.blocks.append(ResNetConv2DBlock(in_channels, filters, dilation_rate))
            elif block_type == 'conv':
                convBlock = nn.Conv2d(in_channels, filters, (size, size), padding=padding)
                init.kaiming_uniform_(convBlock.weight, mode='fan_in', nonlinearity='relu')
                self.blocks.append(convBlock)
                self.blocks.append(nn.ReLU())
            elif block_type == 'double_convolution':
                self.blocks.append(DoubleConvWithBN(in_channels, filters, dilation_rate))

            in_channels = filters

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
    
class SeparatedConv2DBlock(nn.Module):
    def __init__(self, in_channels, filters, size=3, padding='same'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, filters, (1, size), padding=padding)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, (size, 1), padding=padding)
        self.bn2 = nn.BatchNorm2d(filters)

        init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        return x
    
class MidscopeConv2DBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, filters, 3, padding=1, dilation=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm2d(filters)

        init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        return x
    
class WidescopeConv2DBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, filters, 3, padding=1, dilation=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm2d(filters)
        self.conv3 = nn.Conv2d(filters, filters, 3, padding=3, dilation=3)
        self.bn3 = nn.BatchNorm2d(filters)

        init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        return x
    
class ResNetConv2DBlock(nn.Module):
    def __init__(self, in_channels, filters, dilation_rate=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, filters, 1, padding='same')
        self.conv2 = nn.Conv2d(in_channels, filters, 3, padding=dilation_rate, dilation=dilation_rate)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv3 = nn.Conv2d(filters, filters, 3, padding=dilation_rate, dilation=dilation_rate)
        self.bn2 = nn.BatchNorm2d(filters)
        self.bn3 = nn.BatchNorm2d(filters)

        init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
        
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn1(x)
        x = F.relu(self.conv3(x))
        x = self.bn2(x)
        x = x + x1
        x = self.bn3(x)
        return x
    
class DoubleConvWithBN(nn.Module):
    def __init__(self, in_channels, filters, dilation_rate=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, filters, 3, padding=dilation_rate, dilation=dilation_rate)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=dilation_rate, dilation=dilation_rate)
        self.bn2 = nn.BatchNorm2d(filters)

        init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        return x
    
class Duckv2Conv2DBlock(nn.Module):
    def __init__(self, in_channels, filters, size):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.widescope = WidescopeConv2DBlock(in_channels, filters)
        self.midscope = MidscopeConv2DBlock(in_channels, filters)
        self.resnet1 = ConvBlock2D(in_channels, filters, 'resnet', repeat=1)
        self.resnet2 = ConvBlock2D(in_channels, filters, 'resnet', repeat=2)
        self.resnet3 = ConvBlock2D(in_channels, filters, 'resnet', repeat=3)
        self.separated = SeparatedConv2DBlock(in_channels, filters, size=6)
        self.bn_final = nn.BatchNorm2d(filters)
        
    def forward(self, x):
        x = self.bn(x)
        x1 = self.widescope(x)
        x2 = self.midscope(x)
        x3 = self.resnet1(x)
        x4 = self.resnet2(x)
        x5 = self.resnet3(x)
        x6 = self.separated(x)
        x = x1 + x2 + x3 + x4 + x5 + x6
        x = self.bn_final(x)
        return x
    
class DUCKNet(nn.Module):
    def __init__(self, input_channels: int, out_classes: int, starting_filters: int) -> None:
        super().__init__()
        
        # Downsampling Path
        self.pad1 = nn.ZeroPad2d((0, 1, 0, 1))
        self.p1 = nn.Conv2d(input_channels, starting_filters*2, kernel_size=2, stride=2)

        self.pad2 = nn.ZeroPad2d((0, 1, 0, 1))
        self.p2 = nn.Conv2d(starting_filters*2, starting_filters*4, kernel_size=2, stride=2)

        self.pad3 = nn.ZeroPad2d((0, 1, 0, 1))
        self.p3 = nn.Conv2d(starting_filters*4, starting_filters*8, kernel_size=2, stride=2)

        self.pad4 = nn.ZeroPad2d((0, 1, 0, 1))
        self.p4 = nn.Conv2d(starting_filters*8, starting_filters*16, kernel_size=2, stride=2)

        self.pad5 = nn.ZeroPad2d((0, 1, 0, 1))
        self.p5 = nn.Conv2d(starting_filters*16, starting_filters*32, kernel_size=2, stride=2)
        
        # Initial Block
        self.t0 = ConvBlock2D(input_channels, starting_filters, 'duckv2', repeat=1)
        
        # Encoder Blocks
        self.l1i_pad = nn.ZeroPad2d((0, 1, 0, 1))
        self.l1i = nn.Conv2d(starting_filters, starting_filters*2, kernel_size=2, stride=2)
        self.t1 = ConvBlock2D(starting_filters*2, starting_filters*2, 'duckv2', repeat=1)
        
        self.l2i_pad = nn.ZeroPad2d((0, 1, 0, 1))
        self.l2i = nn.Conv2d(starting_filters*2, starting_filters*4, kernel_size=2, stride=2)
        self.t2 = ConvBlock2D(starting_filters*4, starting_filters*4, 'duckv2', repeat=1)
        
        self.l3i_pad = nn.ZeroPad2d((0, 1, 0, 1))
        self.l3i = nn.Conv2d(starting_filters*4, starting_filters*8, kernel_size=2, stride=2)
        self.t3 = ConvBlock2D(starting_filters*8, starting_filters*8, 'duckv2', repeat=1)
        
        self.l4i_pad = nn.ZeroPad2d((0, 1, 0, 1))
        self.l4i = nn.Conv2d(starting_filters*8, starting_filters*16, kernel_size=2, stride=2)
        self.t4 = ConvBlock2D(starting_filters*16, starting_filters*16, 'duckv2', repeat=1)
        
        self.l5i_pad = nn.ZeroPad2d((0, 1, 0, 1))
        self.l5i = nn.Conv2d(starting_filters*16, starting_filters*32, kernel_size=2, stride=2)
        self.t51 = ConvBlock2D(starting_filters*32, starting_filters*32, 'resnet', repeat=2)
        self.t53 = ConvBlock2D(starting_filters*32, starting_filters*16, 'resnet', repeat=2)
        
        # Decoder Blocks
        self.l5o = nn.Upsample(scale_factor=2, mode='nearest')
        self.q4 = ConvBlock2D(starting_filters*16, starting_filters*8, 'duckv2', repeat=1)
        
        self.l4o = nn.Upsample(scale_factor=2, mode='nearest')
        self.q3 = ConvBlock2D(starting_filters*8, starting_filters*4, 'duckv2', repeat=1)
        
        self.l3o = nn.Upsample(scale_factor=2, mode='nearest')
        self.q6 = ConvBlock2D(starting_filters*4, starting_filters*2, 'duckv2', repeat=1)
        
        self.l2o = nn.Upsample(scale_factor=2, mode='nearest')
        self.q1 = ConvBlock2D(starting_filters*2, starting_filters, 'duckv2', repeat=1)
        
        self.l1o = nn.Upsample(scale_factor=2, mode='nearest')
        self.z1 = ConvBlock2D(starting_filters, starting_filters, 'duckv2', repeat=1)
        
        # Final Layer
        self.output = nn.Conv2d(starting_filters, out_classes, kernel_size=1)

        # Weight Initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Para capas de downsampling (p1, p2, l1i, l2i, etc.)
                if "p" in m._get_name() or "l" in m._get_name():
                    nn.init.xavier_uniform_(m.weight)  # glorot_uniform
                else:
                    nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')  # he_uniform
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        p1 = self.p1(self.pad1(x))
        p2 = self.p2(self.pad2(p1))
        p3 = self.p3(self.pad3(p2))
        p4 = self.p4(self.pad4(p3))
        p5 = self.p5(self.pad5(p4))
        
        t0 = self.t0(x)
        
        # Downsample blocks
        l1i = self.l1i(self.l1i_pad(t0))
        s1 = l1i + p1
        t1 = self.t1(s1)
        
        l2i = self.l2i(self.l2i_pad(t1))
        s2 = l2i + p2
        t2 = self.t2(s2)
        
        l3i = self.l3i(self.l3i_pad(t2))
        s3 = l3i + p3
        t3 = self.t3(s3)
        
        l4i = self.l4i(self.l4i_pad(t3))
        s4 = l4i + p4
        t4 = self.t4(s4)
        
        l5i = self.l5i(self.l5i_pad(t4))
        s5 = l5i + p5
        t51 = self.t51(s5)
        t53 = self.t53(t51)
        
        # Decoder
        l5o = self.l5o(t53)
        c4 = l5o + t4
        q4 = self.q4(c4)
        
        l4o = self.l4o(q4)
        c3 = l4o + t3
        q3 = self.q3(c3)
        
        l3o = self.l3o(q3)
        c2 = l3o + t2
        q6 = self.q6(c2)
        
        l2o = self.l2o(q6)
        c1 = l2o + t1
        q1 = self.q1(c1)
        
        l1o = self.l1o(q1)
        c0 = l1o + t0
        z1 = self.z1(c0)
        
        # Output
        return self.output(z1)