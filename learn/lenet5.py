import torch
import torch.nn as nn

_conv2d = nn.Conv2d
_linear = nn.Linear

class Lenet5(nn.Module):

    vision = "LeNet5"

    def __init__(self, init_weights=True):
        super(Lenet5, self).__init__()
        _relu = nn.ReLU(inplace=False)
        self.features = nn.Sequential(
            _conv2d(3, 64, 5, bias=False),
            _relu,
            nn.MaxPool2d(2, 2),
            _conv2d(64, 128, 3, bias=False),
            _relu,
            nn.MaxPool2d(2, 2),
            _conv2d(128, 128, 5, bias=False),
        )
        self.classifier = nn.Sequential(
            _linear(128*4, 128, bias=False),
            _relu,
            _linear(128, 64, bias=False),
            _relu,
            _linear(64, 10, bias=False),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = Lenet5()
    from frontend import DynamoCompiler, transform
    dynamo_compiler = DynamoCompiler()

    # Import the model into MLIR module and parameters.
    with torch.no_grad():
        data = torch.randn([1, 3, 32, 32])
        graphs = dynamo_compiler.importer(model, data)
    with open("lenet5.prototxt", "w") as f:
        print(transform.export_prototxt(graphs[0]), file=f)
