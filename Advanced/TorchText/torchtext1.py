# STEPS

# 1. Specify how preprocessing should be done -> Fields
# 2. Use Dataset to load the data -> TabularDataset (JSON/CSV/TSV Files)
# 3. Construct an iterator to do batching & padding -> BucketIterator

from torchtext.data import Field, TabularDataset, BucketIterator
import spacy 


"""
to insatll space lenguages use:
python -m spacy download en
python -m spacy download de
"""

# tokenize = lambda x: x.split()
spacy_en = spacy.load('en')

def tokenize(text):
	return [tok.text for tok in spacy_en.tokenizer(text)]

quote = Field(sequential = True, use_vocab = True, tokenize = tokenize,  lower = True)
score = Field(sequential = False, use_vocab = False)

fields = {'quote': ('q', quote), 'score': ('s', score)}

# get the quote: batch.q -- get the score: batch.s

train_data, test_data = TabularDataset.splits(
							path = 'mydata',
							train = 'train.json',
							test = 'test.json',
							# validation = 'validation.json',
							format = 'json',
							fields = fields
							)

## for csv file
# train_data, test_data = TabularDataset.splits(
# 							path = 'mydata',
# 							train = 'train.csv',
# 							test = 'test.csv',
# 							# validation = 'validation.csv',
# 							format = 'csv',
# 							fields = fields
# 							)

## for tsv file
# train_data, test_data = TabularDataset.splits(
# 							path = 'mydata',
# 							train = 'train.tsv',
# 							test = 'test.tsv',
# 							# validation = 'validation.tsv',
# 							format = 'tsv',
# 							fields = fields
# 							)


# print(train_data[0].__dict__.keys())
# print(test_data[0].__dict__.keys())

quote.build_vocab(train_data, 
				max_size = 10000,
				min_freq = 1,
				# vectors = 'glove.6B.100d'  # 1 GB
				)

train_iterator, test_iterator = BucketIterator.splits(
	(train_data, test_data),
	batch_size = 2,
	device = 'cuda')

for batch in train_iterator:
	print(batch.q)
	print(batch.s)