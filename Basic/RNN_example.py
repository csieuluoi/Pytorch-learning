import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Hyperparameters
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers,  num_classes = 10):
		super(RNN, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		# # RNN
		# self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True) #use batch_first when data have the shape of (batch, sequence_length,...)
		# # GRU
		# self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True) #use batch_first when data have the shape of (batch, sequence_length,...)
		# LSTM
		self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True) #use batch_first when data have the shape of (batch, sequence_length,...)

		self.fc = nn.Linear(hidden_size*sequence_length, num_classes)
		# when using only information of the last hidden state
		self.fc = nn.Linear(hidden_size, num_classes)

	def forward(self, x):
		# initialize hidden state 
		h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

		# initialize cell state for LSTM
		c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
		# Forward Prop
		# out, _ = self.rnn(x, h0) # RNN return out and "_" is the hidden state
		# for LSTM
		out, _ = self.rnn(x, (h0, c0)) # RNN return out and "_" is the hidden state

		# out = out.reshape(out.shape[0], -1)

		# when using only information of the last hidden state
		
		out = self.fc(out[:, -1, :])

		return out

# model = RNN(input_size, hidden_size, num_layers, num_classes)
# x = torch.randn(64, 1, 28, 28)
# print(model(x).shape)

# Set device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)




# Load Data
train_dataset = datasets.MNIST(root = 'dataset/', train = True, transform = transforms.ToTensor(), download = True)
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_dataset = datasets.MNIST(root = 'dataset/', train = False, transform = transforms.ToTensor(), download = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)


# Initialize network
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Train Network

for epoch in range(num_epochs):
	for batch_idx, (data, targets) in enumerate(train_loader):
		# Get data to cuda if possible
		data = data.to(device = device).squeeze(1)
		targets = targets.to(device = device)

		# forward
		scores = model(data)
		loss = criterion(scores, targets)

		# backward
		optimizer.zero_grad()
		loss.backward()

		# gradient descent or adam step
		optimizer.step()


# Check accuracy on training and test to see how good our model

def check_accuracy(loader, model):
	if loader.dataset.train:
		print("Checking accuracy on training data")
	else:
		print("Checking accuracy on test data")
	num_correct = 0
	num_samples = 0
	model.eval()

	with torch.no_grad():
		for x, y in loader:
			x = x.to(device = device).squeeze(1)
			y = y.to(device = device)

			scores = model(x)

			_, predictions = scores.max(1)
			num_correct += (predictions==y).sum()
			num_samples += predictions.size(0)

		print(f"got {num_correct}/ {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)