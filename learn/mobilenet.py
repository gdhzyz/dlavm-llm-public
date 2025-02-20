import torch
import torch.nn as nn

_conv2d = nn.Conv2d
_linear = nn.Linear

class DepthwiseBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(DepthwiseBlock, self).__init__()
        self.features = nn.Sequential(
            _conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            _conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        return x

class MobileNet(nn.Module):

    vision = "MobileNet v1"

    def __init__(self, num_classes=10, init_weights=True):
        super(MobileNet, self).__init__()
        conv1 = _conv2d(3, 32, 3, 2, 1)
        block2 = nn.Sequential(
            DepthwiseBlock(32, 64, 1),
            # DepthwiseBlock(64, 128, 2),
            # DepthwiseBlock(128, 128, 1),
            # DepthwiseBlock(128, 256, 2),
            # DepthwiseBlock(256, 256, 1),
            # DepthwiseBlock(256, 512, 2),
        )
        block3 = nn.Sequential(
            DepthwiseBlock(512, 512, 1),
            DepthwiseBlock(512, 512, 1),
            DepthwiseBlock(512, 512, 1),
            DepthwiseBlock(512, 512, 1),
            DepthwiseBlock(512, 512, 1),
        )
        block4 = nn.Sequential(
            DepthwiseBlock(512, 1024, 2),
            DepthwiseBlock(1024, 1024, 1),
        )
        pool5 = nn.AvgPool2d(7, 7)
        self.features = nn.Sequential(
            conv1,
            block2,
            # block3,
            # block4,
            pool5
        )
        self.classifier = _linear(16384, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = MobileNet()

    from frontend import DynamoCompiler, transform
    dynamo_compiler = DynamoCompiler()
    # Import the model into MLIR module and parameters.
    with torch.no_grad():
        data = torch.randn([1, 3, 224, 224])
        graphs = dynamo_compiler.importer(model, data)
    with open("mobilenet.prototxt", "w") as f:
        print(transform.export_prototxt(graphs[0]), file=f)