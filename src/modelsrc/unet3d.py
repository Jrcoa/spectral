import torch
import torch.nn as nn

class ResBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=False):
        super(ResBlock3d, self).__init__()
        
        if stride < 0:
            # use transpose convolution to upsample
            self.conv1 = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=-stride, padding=kernel_size//2, output_padding=1)
            self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size, stride=-stride, padding=kernel_size//2, output_padding=1)
            self.match_dimensions = nn.Sequential(nn.ConvTranspose3d(in_channels, out_channels, kernel_size=1, stride=-stride, padding=0, output_padding=1), nn.ConvTranspose3d(out_channels, out_channels, kernel_size=1, stride=-stride, padding=0, output_padding=1)) if in_channels != out_channels else None
        else:
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=padding)
            self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding=padding)
            self.match_dimensions = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride**2, padding=0) if in_channels != out_channels else None
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.leakrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.dropout = nn.Dropout(p=0.2) if dropout else None
        
    def forward(self, x):
        # <batch_size>, <channels>, <depth> (bands), <height>, <width>
        # First convolution is to match the dimensions of the residual and the output
        residual = x
        residual = self.match_dimensions(residual) if self.match_dimensions else residual
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.leakrelu(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = out + residual
        
        if self.dropout:
            out = self.dropout(out)
        return out

class Net3d(nn.Module):
    def __init__(self, in_channels_3d=1, out_channels_3d=1, bands=288, classes=60, base_block=None, dropout=False):
        super(Net3d, self).__init__()
        self.base_block = base_block
        self.encoder = nn.Sequential(
            base_block(in_channels_3d, 32, stride=2, dropout=dropout),  # Downsample
            base_block(32, 64, dropout=dropout),  # Downsample
            base_block(64, 128, dropout=dropout),  # Downsample
        )
        self.decoder = nn.Sequential(
            base_block(128, 64, dropout=dropout),
            base_block(64, 32, stride=-2, dropout=dropout),
            base_block(32, out_channels_3d, dropout=dropout)
        )
        
        #self.projection_layer = nn.Conv2d(bands, classes, kernel_size=1, stride=1, padding=0)
        self.projection_layer = nn.Conv2d(bands, classes, kernel_size=1, stride=1, padding=0) 
    def forward(self, x):
        original_shape = x.shape # <batch_size>, <channels>, <depth> (bands), <height>, <width>
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.squeeze(1)
        x = self.projection_layer(x)
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        return x

class UNet3dAdd(nn.Module):
    def __init__(self, in_channels_3d=1, out_channels_3d=1, bands=288, classes=60, base_block=None, projection_layer=None, dropout=False):
        super(UNet3dAdd, self).__init__()
        self.base_block = base_block
        
        self.e1 = base_block(in_channels_3d, 32, stride=2, dropout=dropout)  # outputs <N, 32, band/4, H/4, W/4>
        self.e2 = base_block(32, 64, dropout=dropout)  # Outputs <N, 64, band/4, H/4, W/4>
        self.e3 = base_block(64, 128, dropout=dropout)  # Outputs <N, 128, band/4, H/4, W/4>

        self.d1 = base_block(128, 64, dropout=dropout) # Outputs <N, 64, band/4, H/4, W/4>
        self.d2 = base_block(64, 32, dropout=dropout) # Outputs <N, 32, band/4, H/4, W/4>
        self.d3 = base_block(32, out_channels_3d, stride=-2, dropout=dropout) # Outputs <N, out_channels_3d, band, H, W>

        #self.projection_layer = nn.Conv2d(bands, classes, kernel_size=1, stride=1, padding=0)
        self.projection_layer = nn.Conv2d(bands, classes, kernel_size=1, stride=1, padding=0) if projection_layer is None else projection_layer
        self.classes = classes
    def forward(self, x):
        original_shape = x.shape # <batch_size>, <channels>, <depth> (bands), <height>, <width>
        
        x1 = self.e1(x)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        
        x = self.d1(x3) + x2
        x = self.d2(x) + x1
        x = self.d3(x)
        
        x = x.squeeze(1)
        x = self.projection_layer(x)
        x = x.reshape(original_shape[0], original_shape[3] * original_shape[4], self.classes)
        return x

class UNet3dConcat(nn.Module):
    def __init__(self, in_channels_3d=1, out_channels_3d=1, bands=288, classes=60, base_block=None, projection_layer=None, dropout=False):
        super(UNet3dConcat, self).__init__()
        self.base_block = base_block
        self.classes = classes 
        self.conv = nn.Conv3d(in_channels_3d, 16, kernel_size=7, stride=1, padding='same')   
        self.e1 = base_block(16, 32, dropout=dropout)  
        self.e2 = base_block(32, 64, dropout=dropout)  
        self.e3 = base_block(64, 128, dropout=dropout)  

        self.d1 = base_block(128 + 64, 64, dropout=dropout)
        self.d2 = base_block(64 + 32, 32, dropout=dropout) 
        self.d3 = base_block(32, out_channels_3d, dropout=dropout) 

        self.maxpool = nn.MaxPool3d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2)
        #self.projection_layer = nn.Conv2d(bands, classes, kernel_size=1, stride=1, padding=0)
        self.projection_layer = nn.Conv2d(bands, classes, kernel_size=1, stride=1, padding=0) if projection_layer is None else projection_layer

        self.apply(self.init_weights)
        
    def forward(self, x):
        original_shape = x.shape # <batch_size>, <channels>, <depth> (bands), <height>, <width>
        x1 = self.conv(x) # <batch_size>, 16, bands, H, W
        x1 = self.maxpool(x1) 
        x1 = self.e1(x1) # <batch_size>, 32, bands/2, H/2, W/2
        x2 = self.maxpool(x1) 
        x2 = self.e2(x2) # <batch_size>, 64, bands/4, H/4, W/4
        x3 = self.maxpool(x2) 
        x3 = self.e3(x3) # <batch_size>, 128, bands/8, H/8, W/8
        
        d1_in = torch.cat((self.upsample(x3), x2), dim=1) # <batch_size>, 128+64, bands/4, H/4, W/4
        d1_out = self.d1(d1_in) # <batch_size>, 64, bands/4, H/4, W/4
        
        d2_in = torch.cat((self.upsample(d1_out), x1), dim=1) # <batch_size>, 64+32, bands/2, H/2, W/2
        d2_out = self.d2(d2_in) # <batch_size>, 32, bands/2, H/2, W/2
        
        d3_in = self.upsample(d2_out) # <batch_size>, 32, bands, H, W
        x = self.d3(d3_in) # <batch_size>, out_channels_3d, bands, H, W
        
        x = x.squeeze(1)
        x = self.projection_layer(x)
        x = x.permute(0, 2, 3, 1) # <batch>, <height>, <width>, <classes>
        return x   
    
    def init_weights(self, m):
        # NOTE: I should consider using the leaky relu initialzation for the "conv1" children
        
        # if the parameter path name contains "conv1" then use the leaky relu initialization
        #if m.__class__.__name__ == 'Conv3d' and 'conv1' in m._parameters['weight'].path:
        #    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if isinstance(m, nn.ConvTranspose3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class Net3dResBlock(Net3d):
    def __init__(self, in_channels=1, out_channels=1, classes=60, bands=288):
        super(Net3dResBlock, self).__init__(in_channels, out_channels, classes=classes, base_block=ResBlock3d, bands=bands)

class UNet3dResBlock(UNet3dAdd):
    def __init__(self, in_channels=1, out_channels=1, classes=60, bands=288):
        super(UNet3dResBlock, self).__init__(in_channels, out_channels, classes=classes, base_block=ResBlock3d, bands=bands)

class UNet3dResBlockV2(UNet3dConcat):
    def __init__(self, in_channels=1, out_channels=1, classes=60, bands=288, dropout=False):
        super(UNet3dResBlockV2, self).__init__(in_channels, out_channels, classes=classes, base_block=ResBlock3d, dropout=dropout, bands=bands)

class UNet3dResBlockV3(UNet3dConcat):
    def __init__(self, in_channels=1, out_channels=1, classes=60, bands=288, dropout=False):
        super(UNet3dResBlockV3, self).__init__(in_channels, out_channels, classes=classes, base_block=ResBlock3d, dropout=dropout, bands=bands,
                                               projection_layer=nn.Sequential(nn.Conv2d(bands, bands, 1, 1), nn.Conv2d(bands, classes, 1, 1)))