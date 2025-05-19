import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50FeatureExtractor(nn.Module):
    def init(self):
        super().init()
	    resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
	    self.features = nn.Sequential(*list(resnet50.children())[:-1])
	    self.output_dim = resnet50.fc.in_features
    def forward(self, x_raw_batch):
        with torch.no_grad():
            features = self.features(x_raw_batch)
            features = torch.flatten(features, 1)
        return features