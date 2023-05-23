
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import numpy as np 
import spacy
import random
from torch.utils.tensorboard import SummaryWriter # to print to tensorboard
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


spacy_ger = spacy.load('de')
spacy_eng = spacy.load('en')

def tokenize_ger(text):
	return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenize_eng(text):
	return [tok.text for tok in spacy_eng.tokenizer(text)]

german = Field(tokenize = tokenize_ger, lower = True, init_token = '<sos>', eos_token = '<eos>')
english = Field(tokenize = tokenize_eng, lower = True, init_token = '<sos>', eos_token = '<eos>')

train_data, validation_data, test_data = Multi30k.splits(exts = ('.de', '.en'),
														fields = (german, english))


german.build_vocab(train_data, max_size = 10000, min_freq = 2)
english.build_vocab(train_data, max_size = 10000, min_freq = 2)


# Implimenting model

class Encoder(nn.Module):
	def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
		super(Encoder, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers

		self.dropout = nn.Dropout(p)
		self.embedding = nn.Embedding(input_size, embedding_size)
		self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout = p)

	def forward(self, x):
		# x shape : (seq_length, N) - N: batch_size

		embedding = self.dropout(self.embedding(x))
		# embedding shape : (seq_length, N, embedding_size)
		outputs, (hidden, cell) = self.rnn(embedding)

		return hidden, cell


class Decoder(nn.Module):
	def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
		super(Decoder, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers

		self.dropout = nn.Dropout(p)
		self.embedding = nn.Embedding(input_size, embedding_size)

		self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout = p)
		self.fc = nn.Linear(hidden_size, output_size)

	def forward(self, x, hidden, cell):
		# shape of x: (N) but we want (1, N)
		x = x.unsqueeze(0)

		embedding = self.dropout(self.embedding(x))
		# embedding shape: (1, N, embedding_size)

		outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
		# shape of output: (1, N, hidden_size)

		predictions = self.fc(outputs)
		# shape of predictions: (1, N, length_of_vocab)

		predictions = predictions.squeeze(0)

		return predictions, hidden, cell 


class Seq2Seq(nn.Module):
	def __init__(self, encoder, decoder):
		super(Seq2Seq, self).__init__()
		self.encoder = encoder
		self.decoder = decoder

	def forward(self, source, target, teacher_force_ratio = 0.5):
		batch_size = source.shape[1]
		# (target_len, N) 
		target_len = target.shape[0]
		target_vocab_size = len(english.vocab)

		outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
		hidden, cell = self.encoder(source)

		# grab start token
		x = target[0]

		for t in range(1, target_len):
			output, hidden, cell = self.decoder(x, hidden, cell)	
			# output shape: (N, english_vocab_size)

			outputs[t] = output

			best_guess = output.argmax(1)
			# With probability of teacher_force_ratio we take the actual next word
			# otherwise we take the word that the Decoder predicted it to be.
			# Teacher Forcing is used so that the model gets used to seeing
			# similar inputs at training and testing time, if teacher forcing is 1
			# then inputs at test time might be completely different than what the
			# network is used to. This was a long comment.
			x = target[t] if random.random() < teacher_force_ratio else best_guess

		return outputs

### Training Phase

# Training hyperparameters
num_epochs = 20
learning_rate = 0.001
batch_size = 64

# Model hyperparameters
load_model = False
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)

output_size = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300

hidden_size = 1024
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5

# Tensorboard
writer = SummaryWriter(f'runs/loss_plot')
step = 0

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
	(train_data, validation_data, test_data),
	batch_size = batch_size,
	sort_within_batch = True,
	sort_key = lambda x: len(x.src),
	device = device)

encoder_net = Encoder(input_size = input_size_encoder,
	embedding_size = encoder_embedding_size, 
	hidden_size = hidden_size,
	num_layers = num_layers, 
	p = enc_dropout).to(device)

decoder_net = Decoder(input_size = input_size_decoder,
	embedding_size = decoder_embedding_size, 
	hidden_size = hidden_size,
	output_size = output_size,
	num_layers = num_layers, 
	p = dec_dropout).to(device)


model = Seq2Seq(encoder_net, decoder_net).to(device)

pad_idx = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
if load_model:
	load_checkpoint(torch.load('my_checkpoint.pth.ptar'), model.optimizer)


sentence = "ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen."

for epoch in range(num_epochs):
	print(f"Epoch [{epoch} / {num_epochs}]")

	checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
	save_checkpoint(checkpoint)
	model.eval()

	translated_sentence = translate_sentence(
		model, sentence, german, english, device, max_length=50
	)

	print(f"Translated example sentence: \n {translated_sentence}")

	model.train()

	for batch_idx, batch in enumerate(train_iterator):
		inp_data = batch.src.to(device)
		target = batch.trg.to(device)

		output = model(inp_data, target)
		# output shape : (trg_len, batch_size, output_dim)

		# (N, 10) and targets would be (N)
		# 0 idx is the start token '<sos>'

		# Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
		# doesn't take input in that form. For example if we have MNIST we want to have
		# output to be: (N, 10) and targets just (N). Here we can view it in a similar
		# way that we have output_words * batch_size that we want to send in into
		# our cost function, so we need to do some reshapin. While we're at it
		# Let's also remove the start token while we're at it
		output = output[1:].reshape(-1, output.shape[2])
		target = target[1:].reshape(-1)

		optimizer.zero_grad()
		loss = criterion(output, target)

		loss.backward()

		# avoid gradient explosion
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)

		optimizer.step()
		writer.add_scalar('Training loss', loss, global_step = step)
		step += 1

# running on entire test data takes a while
score = bleu(test_data[1:100], model, german, english, device)
print(f"Bleu score {score * 100:.2f}")