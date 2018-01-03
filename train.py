import torch
import torch.autograd as autograd
import torch.utils as utils
import torch.optim as optim
import torch.nn as nn
import argparse
import data_helpers
import numpy as np
from tensorflow.contrib import learn
import text_cnn

parser = argparse.ArgumentParser()


parser.add_argument('--dev_sample_percentage', '-dsp', default=0.1, type=float, help='Percentage of the training data to use for validation')
parser.add_argument('--positive_data_file', '-pdf', default='data/rt-polarity.pos', help='Data source for the positive data.')
parser.add_argument('--negative_data_file', '-ndf', default='data/rt-polarity.neg', help='Data source for the negative data.')
parser.add_argument('--embedding_dim', '-ed', default=128, type=int, help='Dimensionality of character embedding (default: 128)')
parser.add_argument('--filter_sizes', '-fs', default='3,4,5', help="Comma-separated filter sizes (default: '3,4,5')")
parser.add_argument('--num_filters', '-nf', default=128, type=int, help='Number of filters per filter size (default: 128)')
parser.add_argument('--dropout_keep_prob', '-dkp', default=0.5, type=float, help='Dropout keep probability (default: 0.5)')
parser.add_argument('--l2_reg_lambda', '-lrl', default=0.0, type=float, help='L2 regularization lambda (default: 0.0)')
parser.add_argument('--batch_size', '-bs', default=64, type=int, help='Batch Size (default: 64)')
parser.add_argument('--num_epochs', '-ne', default=200, type=int, help='Number of training epochs (default: 200)')
parser.add_argument('--evaluate_every', '-ee', default=100, type=int, help='Evaluate model on dev set after this many steps (default: 100)')
parser.add_argument('--checkpoint_every', '-ce', default=100, type=int, help='Save model after this many steps (default: 100)')
parser.add_argument('--num_checkpoints', '-nc', default=5, type=int, help='Number of checkpoints to store (default: 5)')
parser.add_argument('--allow_soft_placement', '-asp', action='store_true', help='Allow device soft device placement') 
parser.add_argument('--log_device_placement', '-ldp', action='store_true', help='Log placement of ops on devices') 



print("List all parameters...")
args = parser.parse_args()
args_dict = vars(args)
for key,value in args_dict.items():
    print(key+": "+str(value))
    
print("")

# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(args.positive_data_file, args.negative_data_file)

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

dev_sample_index = -1 * int(args.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

del x, y, x_shuffled, y_shuffled

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

model = text_cnn.TextCNN(sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=args.embedding_dim,
            filter_sizes=list(map(int, args.filter_sizes.split(","))),
            num_filters=args.num_filters,
            dropout_keep_prob=args.dropout_keep_prob)

# out_variable = autograd.Variable(torch.from_numpy(x_train[:2]))
# output = model(out_variable)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Generate batches
batches = data_helpers.batch_iter(
    list(zip(x_train, y_train)), args.batch_size, args.num_epochs)
# Training loop. For each batch...
for i, batch in enumerate(batches):
    x_batch, y_batch = zip(*batch)
    
    x_batch = np.stack(x_batch, axis=0)
    x_batch = autograd.Variable(torch.from_numpy(x_batch))
    
    y_batch = [np.argmax(y) for y in y_batch]
    y_batch = autograd.Variable(torch.LongTensor(y_batch))
    
    
    optimizer.zero_grad()
     
    pred = model.forward(x_batch)
     
    cost = criterion(pred, y_batch)
     
    cost.backward()
    optimizer.step()
    
    print "Step "+str(i+1)+", Loss: " + "{:.6f}".format(cost.data[0])
    
    if i % args.evaluate_every == 0:
        
        x = autograd.Variable(torch.from_numpy(x_dev))
        pred = model.forward(x)
        
        _, y_pred = torch.max(pred, dim=1)
        y_pred = y_pred.data.numpy()
        
        y_gold = np.argmax(y_dev, axis=1)
        
        accuracy = np.sum(np.equal(y_pred, y_gold)) / float(len(x_dev))
    
        print "Step "+str(i+1)+", Accuracy: " + "{:.2f}".format(accuracy)
        
        


print("end")
