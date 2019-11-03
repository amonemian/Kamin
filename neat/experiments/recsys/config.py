import torch
import torch.nn as nn
from torch import autograd
from neat.phenotype.feed_forward import FeedForwardNet
import numpy
# from neat.visualize import draw_net
from time import time
from funk_svd import SVD


def train_one_epoch(model, inputs, targets, loss_fn, optimizer, epoch_no, device, verbose=1):
    'trains the model for one epoch and returns the loss'
    if verbose:
        print("Epoch = {}".format(epoch_no))
    # Training
    # get user, item and rating data
    t1 = time()
    epoch_loss = []
    # put the model in train mode before training
    model.train()
    # transfer the data to GPU
    for k, feed_dict in enumerate(inputs):
        """FIXME: for key in feed_dict:
			if type(feed_dict[key]) != type(None):
				feed_dict[key] = feed_dict[key].to(dtype = torch.long, device = device)"""
        # if(k % 100 == 0):
        #    print('Training with inputs. {0} out of {1} inputs'.format(k , len(inputs)))
        # get the predictions
        prediction = model(feed_dict)
        # print(prediction.shape)
        # get the actual targets
        rating = targets[k]

        # convert to float and change dim from [batch_size] to [batch_size,1]
        rating = rating.float().view(prediction.size())
        loss = loss_fn(prediction, rating)
        # clear the gradients
        optimizer.zero_grad()
        # backpropagate **** QUESTION
        loss.backward()
        # update weights
        optimizer.step()
        # accumulate the loss for monitoring
        epoch_loss.append(loss.item())
    epoch_loss = numpy.mean(epoch_loss)
    if verbose:
        print("Epoch completed {:.1f} s".format(time() - t1))
        print("Train Loss: {}".format(epoch_loss))
    return epoch_loss


class RecsysConfig:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VERBOSE = True

    NUM_INPUTS = 2
    NUM_OUTPUTS = 1
    USE_BIAS = True

    # ACTIVATION = 'sigmoid'
    ACTIVATION = 'ReLU'
    # ACTIVATION = 'tanh'

    SCALE_ACTIVATION = 4.9

    POPULATION_SIZE = 150
    NUMBER_OF_GENERATIONS = 150
    # NUMBER_OF_GENERATIONS = 1500

    SPECIATION_THRESHOLD = 3.0

    CONNECTION_MUTATION_RATE = 0.80
    CONNECTION_PERTURBATION_RATE = 0.90
    ADD_NODE_MUTATION_RATE = .1  # 0.03
    ADD_CONNECTION_MUTATION_RATE = .8  # 0.5

    CROSSOVER_REENABLE_CONNECTION_GENE_RATE = 0.25

    # Top percentage of species to be saved before mating
    PERCENTAGE_TO_SAVE = 0.30

    FITNESS_THRESHOLD = 13.0

    def fitness_fn(self, genome, shared_data):
        phenotype = FeedForwardNet(genome, self)
        phenotype.to(self.DEVICE)
        criterion = nn.MSELoss()
        totloss = 0.0

        epochs = 3
        # loss_fn = torch.nn.BCELoss()
        weight_decay = 0.00001
        optimizer = torch.optim.Adam(phenotype.parameters(), weight_decay=weight_decay)

        #for epoch in (range(epochs)):
            # print("Epoch {0} out of {1} epochs starts".format(epoch + 1, epochs))
            #epoch_loss = train_one_epoch(phenotype, shared_data.inputs_tr, shared_data.targets_tr, criterion, optimizer, epoch, self.DEVICE, False)

        for input, target in zip(shared_data.inputs_tst, shared_data.targets_tst):  # 4 training examples
            input, target = input.to(self.DEVICE), target.to(self.DEVICE)

            pred = phenotype(
                input)  # calling the FeedForwardNet (which is torch Module and calling it eventually causes forward to be called)
            # loss = torch.norm(pred - target)
            # loss = float(loss)
            # totloss += loss

            totloss += float(criterion(pred, target))  ####FIXME??

        totloss /= len(shared_data.targets_tst)
        genome.avgloss = totloss

        return shared_data.max_fitness - totloss

    def get_preds_and_labels(self, genome):
        phenotype = FeedForwardNet(genome, self)
        phenotype.to(self.DEVICE)

        predictions = []
        labels = []
        for input, target in zip(self.inputs, self.targets_tst):  # 4 training examples
            input, target = input.to(self.DEVICE), target.to(self.DEVICE)

            predictions.append(float(phenotype(input)))
            labels.append(float(target))

        return predictions, labels
