"""Load model with pretrained weights and initialise new layers."""

from overrides import overrides

import torch
import torch.nn as nn

import torchvision.models as models


class BilinearModel(nn.Module):
    """Load model with pretrained weights and initialise new layers."""

    def __init__(self, num_classes: int = 200) -> None:
        """Load pretrained model, set new layers with specified number of layers."""
        super(BilinearModel, self).__init__()
        model: nn.Module = models.vgg16(pretrained=True)
        self.features: nn.Module = nn.Sequential(*list(model.features)[:-1])
        self.classifier: nn.Module = nn.Linear(512 ** 2, num_classes)
        nn.init.kaiming_normal_(self.classifier.weight.data)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias.data, val=0)

    @overrides
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Extract input features, perform bilinear transform, project to # of classes and return."""
        outputs: torch.Tensor = self.features(inputs)               # extract features from pretrained base
        outputs = outputs.view(-1, 512, 28 ** 2)                    # reshape to batchsize * 512 * 28 ** 2
        outputs = torch.bmm(outputs, outputs.permute(0, 2, 1))      # bilinear product
        outputs = torch.div(outputs, 28 ** 2)                       # divide by 196 to normalize
        outputs = outputs.view(-1, 512 ** 2)                        # reshape to batchsize * 512 * 512
        outputs = torch.sign(outputs) * torch.sqrt(outputs + 1e-5)  # signed square root normalization
        outputs = nn.functional.normalize(outputs, p=2, dim=1)      # l2 normalization
        outputs = self.classifier(outputs)                          # linear projection
        return outputs
