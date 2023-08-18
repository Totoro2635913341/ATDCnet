import torch
import torch.nn as nn

class residualBlock(nn.Module):

    def __init__(self, inChannle=3, outChannle=64, kernelSize=3, padding=1, stride=1):
        super(residualBlock, self).__init__()
        self.kernelSize=kernelSize
        self.padding=padding
        self.stride=stride

        self.plainBlock = nn.Sequential(
            nn.Conv2d(inChannle, outChannle, kernel_size=self.kernelSize, padding=self.padding, stride=self.stride),
            nn.BatchNorm2d(outChannle),
            nn.LeakyReLU()
        )

        self.residualBlock1 = nn.Sequential(
            nn.Conv2d(outChannle, outChannle, kernel_size=self.kernelSize, padding=self.padding, stride=self.stride),
            nn.BatchNorm2d(outChannle),
            nn.LeakyReLU()
        )

        self.residualBlock2 = nn.Sequential(
            nn.Conv2d(outChannle, outChannle, kernel_size=self.kernelSize, padding=self.padding, stride=self.stride),
            nn.BatchNorm2d(outChannle),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Conv2d(outChannle, outChannle, kernel_size=self.kernelSize, padding=self.padding, stride=self.stride)
        self.leakyRelu = nn.LeakyReLU()

    def forward(self, data):

        data = self.plainBlock(data)
        residual1 = self.residualBlock1(data)
        residual2 = self.residualBlock2(residual1)
        residual2 = self.conv3(residual2)
        data = self.leakyRelu(data + residual2)
        return data


class channleAttentionBlock(nn.Module):

    def __init__(self, inChannle=64):
        super(channleAttentionBlock, self).__init__()
        self.inChannle = inChannle
        self.globalAveragePool = nn.AdaptiveAvgPool2d(1)
        self.ratio = 16

        self.denseConnct = nn.Sequential(
            nn.Linear(self.inChannle, int(self.inChannle/self.ratio)),
            nn.LeakyReLU(),
            nn.Linear(int(self.inChannle/self.ratio), self.inChannle),
            nn.Sigmoid()
        )

    def forward(self, data):
        # (B,channel,H,W) -> (B,channel,1,1)
        x1=self.globalAveragePool(data)
        # print("注意力输出尺寸:", x1.shape)
        # (B,channel)
        x2=torch.reshape(x1, x1.shape[0:2])
        attentionScore=self.denseConnct(x2)

        # # (B,channel) -> (B,channel,1,1)
        attentionScore=torch.reshape(attentionScore, [attentionScore.shape[0], attentionScore.shape[1], 1, 1])

        # print("data.shape=", data.shape)
        # print("attentionScore.shape=", attentionScore.shape)

        data = data + data*attentionScore

        return data



class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.residualBlock1 = residualBlock(3, 64)  # 对应output512
        self.residualBlock2 = residualBlock(64, 128)  # 对应output256
        self.residualBlock3 = residualBlock(128, 256)  # 对应output128

    def forward(self, data):

        dataX64 = self.residualBlock1(data)
        dataX128 = self.residualBlock2(dataX64)
        dataX256 = self.residualBlock3(dataX128)

        return dataX64, dataX128, dataX256


class transmissonEncoder(nn.Module):

    def __init__(self):
        super(transmissonEncoder, self).__init__()
        self.residualBlock1 = residualBlock(3, 64)  # 对应output512
        self.residualBlock2 = residualBlock(64, 64)  # 对应output256
        self.residualBlock3 = residualBlock(64, 1)  # 对应output128

    def forward(self, data):
        dataX64 = self.residualBlock1(data)
        dataX128 = self.residualBlock2(dataX64)
        transmission = self.residualBlock3(dataX128)

        return transmission



class ratioRgb(nn.Module):

    def __init__(self):
        super(ratioRgb, self).__init__()
        self.dense1 = nn.Sequential(
            nn.Linear(in_features=3, out_features=3),
            nn.LeakyReLU()
        )
        self.dense2 = nn.Sequential(
            nn.Linear(in_features=3, out_features=3),
            nn.LeakyReLU()
        )
        self.dense3 = nn.Sequential(
            nn.Linear(in_features=6, out_features=3),
            nn.LeakyReLU()
        )


    def forward(self, data):

        data1 = self.dense1(data)
        data2 = self.dense2(data1)
        data = torch.cat([data1, data2], dim=1)
        data = self.dense3(data)
        return data



class Model(nn.Module):
    def __init__(self, num=64):
        super(Model, self).__init__()

        self.ratioRgb = ratioRgb()

        self.encoder = Encoder()
        self.transmissonEncoder = transmissonEncoder()

        self.channleAttention256 = channleAttentionBlock(256)
        self.channleAttention128 = channleAttentionBlock(128)
        self.channleAttention64 = channleAttentionBlock(64)

        self.decoder1 = residualBlock(257, 128)
        self.decoder2 = residualBlock(257, 64)
        self.decoder3 = residualBlock(129, 64)
        self.decoder4 = nn.Conv2d(64, 3, kernel_size=3, padding=1, stride=1)

        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1, stride=1)
        self.decoder = nn.Conv2d(3, 3, kernel_size=3, padding=1, stride=1)

    def forward(self, data, input_RMT):

        dataX64, dataX128, dataX256 = self.encoder(data)

        # print('color改变前.....')
        color = torch.mean(data, dim=[2, 3])
        ratioRgb = self.ratioRgb(color)
        ratioRgb = ratioRgb.view(ratioRgb.shape[0], ratioRgb.shape[1], 1, 1)
        transmission = self.transmissonEncoder(input_RMT)

        attentionDataX256 = self.channleAttention256(dataX256)

        # attentionDataX256 = attentionDataX256 + attentionDataX256*(1 - transmission)
        attentionDataX256 = torch.cat([attentionDataX256, transmission], dim=1)

        attentionDataX256 = self.decoder1(attentionDataX256)

        attentionDataX128 = self.channleAttention128(dataX128)

        # attentionDataX128 = attentionDataX128 + attentionDataX128 * (1 - transmission)
        attentionDataX128 = torch.cat([attentionDataX128, transmission], dim=1)

        attentionDataX128 = torch.cat([attentionDataX256, attentionDataX128], dim=1)

        attentionDataX128 = self.decoder2(attentionDataX128)

        attentionDataX64 = self.channleAttention64(dataX64)

        # attentionDataX64 = attentionDataX64 + attentionDataX64*(1 - transmission)
        attentionDataX64 = torch.cat([attentionDataX64, transmission], dim=1)

        attentionDataX64 = torch.cat([attentionDataX128, attentionDataX64], dim=1)
        attentionDataX64 = self.decoder3(attentionDataX64)

        data = self.decoder4(attentionDataX64)


        data = data + data * ratioRgb
        data = self.conv(data)
        data = self.decoder(data)

        return data


