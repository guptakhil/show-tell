import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models

class ResNet(nn.Module):
	'''
	Encoding via pretrained ResNet.
	'''

	def __init__(self, resnet_version=101, embed_dim=256):
		'''
		Args:
			embed_dim (int) : Embedding dimension between CNN and RNN
		'''

		super(ResNet, self).__init__()

		if resnet_version == 18:
			resnet_model = models.resnet18(pretrained=True)
		elif resnet_version == 34:
			resnet_model = models.resnet34(pretrained=True)
		elif resnet_version == 50:
			resnet_model = models.resnet50(pretrained=True)
		elif resnet_version == 101:
			resnet_model = models.resnet101(pretrained=True)
		elif resnet_version == 152:
			resnet_model = models.resnet152(pretrained=True)
		else:
			raise ValueError('Please specify a valid ResNet version. %d doesn\'t exist.'%(resnet_version))
		self.model = nn.Sequential(*list(resnet_model.children())[:-1])

		#Training only the last 2 layers for the Resnet model i.e. linear and batchnorm layer
		self.linear_secondlast_layer = nn.Linear(resnet_model.fc.in_features, embed_dim)
		self.last_layer = nn.BatchNorm1d(embed_dim, momentum=0.01)

		#Initializing the weights using normal distribution
		self.linear_secondlast_layer.weight.data.normal_(0, 0.05)
		self.last_layer.bias.data.fill_(0)

	def forward(self, x):

		x = self.model(x)
		x = Variable(x.data) # Converting to a PyTorch variable
		x = x.view(x.size(0), -1) # Flattening the output of the CNN model
		x = self.last_layer(self.linear_secondlast_layer(x)) # Applying the linear layer

		return x