import os
import time
import datetime
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from fairseq.models.roberta import RobertaModel
from p2p.transformer import RobertaConstrastiveHead
from p2p.dataset import parse
from tensorboardX import SummaryWriter


dictionary = {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
    "H": 8,
    "I": 9,
    "J": 10,
    "K": 11,
    "L": 12,
    "M": 13,
    "N": 14,
    "O": 15,
    "P": 16,
    "Q": 17,
    "R": 18,
    "S": 19,
    "T": 20,
    "U": 21,
    "V": 22,
    "W": 23,
    "X": 24,
    "Y": 25,
    "Z": 26,
    ".": 27
}


def encode(x):
    """ Convert string to tokens. """
    tokens = list(map(lambda i: dictionary[i], list(x)))
    tokens = torch.Tensor(tokens)
    tokens = tokens.long()
    return tokens

def tokenize(gene, pos, neg, model, device):
    g = torch.cat(list(map(encode, gene)), 0)
    p = torch.cat(list(map(encode, pos)), 0)
    n = torch.cat(list(map(encode, neg)), 0)
    g = model.extract_features(g)
    p = model.extract_features(p)
    n = model.extract_features(n)
    g.to(device)
    p.to(device)
    n.to(device)
    return g, p, n

def train(pretrained_model,
          train_dataloader, test_dataloader,
          logging_path=None,
          emb_dimension=100, epochs=10, betas=(0.9, 0.95),
          summary_interval=100, device=None):
    """ Train the roberta model

    Parameters
    ----------
    pretrained_model : fairseq.models.roberta.RobertaModel
        Pretrained Roberta model.
    train_dataloader : torch.dataset.DataLoader
        DataLoader for training interactions.
    test_dataloader : torch.dataset.DataLoader
        DataLoader for testing interactions.
    emb_dimension : int
        Number of dimensions to train the model.
    logging_path : path
        Path of logging file.
    epochs : int
        Number of epochs for training.
    betas : tuple of float
        Adam beta parameters.
    summary_interval : int
        Number of steps before saving summary.
    device : str
        Name of device to run (specifies gpu or not)

    Returns
    -------
    finetuned_model : p2p.transformer.RobertaConstrastiveHead
    """
    last_summary_time = 0
    # the dimensionality of the roberta model
    roberta_dim = list(list(pretrained_model.parameters())[-1].shape)[0]
    finetuned_model = RobertaConstrastiveHead(roberta_dim, emb_dimension)
    optimizer = optim.Adamax(finetuned_model.parameters(), betas=betas)

    # Initialize logging path
    if logging_path is None:
        basename = "logdir"
        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        logging_path = "_".join([basename, suffix])

    writer = SummaryWriter(logging_path)
    for i in tqdm(range(epochs)):
        now = time.time()
        finetuned_model.train()
        for gene, pos, neg in train_dataloader:
            optimizer.zero_grad()
            g, p, n = tokenize(gene, pos, neg, pretrained_model, device)
            loss = finetuned_model.forward(g, p, n)
            loss.backward()
            optimizer.step()

        # write down summary stats
        now = time.time()
        if now - last_summary_time > summary_interval:
            err = []
            for gene, pos, neg in test_dataloader:
                g, p, n = tokenize(gene, pos, neg, pretrained_model, device)
                cv = finetuned_model.forward(g, p, n)
                err.append(cv)
            err = torch.mean(err)
            writer.add_scalar('test_error', err, i)
            writer.add_scalar('train_error', loss, i)
            last_summary_time = now
    return finetuned_model


def run(fasta_file, links_file,
        checkpoint_path, data_dir, model_path, logging_path,
        training_column='Training',
        emb_dimension=100, num_neg=10,
        epochs=10, betas=(0.9, 0.95),
        batch_size=10, num_workers=10,
        summary_interval=1,
        device='cpu'):
    """ Train interaction model

    Parameters
    ----------
    fasta_file : filepath
        Fasta file of sequences of interest.
    link_file : filepath
        Table of tab delimited interactions
    emb_dimensions : int
        Number of embedding dimensions.
    num_neg : int
        Number of negative samples.
    checkpoint_path : path
        Path for roberta model.
    data_dir : path
        Path to data used for pretraining.
    model_path : path
        Path for finetuned model.
    logging_path : path
        Path for logging information.
    batch_size : int
        Number of protein triples to analyze in a given batch.
    summary_interval : int
        Number of seconds for a summary update.
    device : str
        Name of device to run on.
    """
    # An example of how to load your own roberta model
    # roberta_checkpoint_path = 'checkpoints/uniref50'
    # data_dir = 'data/uniref50'
    # pytorch_dump_folder_path = 'checkpoints/roberta_TF_dump'
    # classification_head = False
    # roberta = FairseqRobertaModel.from_pretrained(
    #     roberta_checkpoint_path, 'checkpoint_best.pt', data_dir)

    pretrained_model = RobertaModel.from_pretrained(
        checkpoint_path, 'checkpoint_best.pt', data_dir)

    train_data, test_data, valid_data = parse(
        fasta_file, links_file, training_column,
        batch_size, num_workers, device)
    # train the fine_tuned model parameters
    finetuned_model = train(
        pretrained_model, train_data, test_data,
        logging_path, 
        emb_dimension, epochs, betas,
        summary_interval, device)

    total, correct = 0, 0
    # evaluate accuracy on validation dataset
    for gene, pos, neg in valid_data:
        g, p, n = tokenize(gene, pos, neg, pretrained_model, device)
        pred = finetuned_model.predict(g, p)
        predicted = torch.round(pred)
        correct += (predicted == 1.0).sum().item()

    # TODO: There needs to be a list of proteins where
    # there is for sure no interaction.

    total = len(valid_data)
    acc = 100 * correct / total
    print(f'Accuracy of model on the {total} validation interactions: {acc}')

    # save the model checkpoint
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    torch.save(finetuned_model.state_dict(),
               os.path.join(model_path, 'checkpoint_' + suffix))
    return acc
