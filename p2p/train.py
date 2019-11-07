import os
import time
import datetime
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from fairseq.models.roberta import RobertaModel
from p2p.transformer import RobertaConstrastiveHead
from p2p.dataset import InteractionDataDirectory
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

def tokenize(gene, pos, neg, model, device, pad=1024):

    with torch.no_grad():
        # extract features, and take <CLS> token
        g = list(map(lambda x: model.extract_features(encode(x))[:, 0, :], gene))
        p = list(map(lambda x: model.extract_features(encode(x))[:, 0, :], pos))
        n = list(map(lambda x: model.extract_features(encode(x))[:, 0, :], neg))
        
        g = torch.cat(g, 0)
        p = torch.cat(p, 0)
        n = torch.cat(n, 0)

    g.to(device)
    p.to(device)
    n.to(device)
    return g, p, n

# @profile
def train(pretrained_model, directory_dataloader,
          logging_path=None,
          emb_dimension=100, epochs=10, betas=(0.9, 0.95),
          summary_interval=100, checkpoint_interval=100, 
          model_path='model', device=None):
    """ Train the roberta model

    Parameters
    ----------
    pretrained_model : fairseq.models.roberta.RobertaModel
        Pretrained Roberta model.
    directory_dataloader : InteractionDataDirectory
        Creates dataloaders
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
    checkpoint_interval : int
        Number of steps before saving checkpoint.
    device : str
        Name of device to run (specifies gpu or not)

    Returns
    -------
    finetuned_model : p2p.transformer.RobertaConstrastiveHead
    """
    last_summary_time = time.time()
    last_checkpoint_time = time.time()
    # the dimensionality of the roberta model
    roberta_dim = list(list(pretrained_model.parameters())[-1].shape)[0]
    finetuned_model = RobertaConstrastiveHead(roberta_dim, emb_dimension)
    optimizer = optim.Adamax(finetuned_model.parameters(), betas=betas)

    # Initialize logging path
    if logging_path is None:
        basename = "logdir"
        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        logging_path = "_".join([basename, suffix])
    it = 0
    writer = SummaryWriter(logging_path)
    now = time.time()

    for k, dataloader in enumerate(directory_dataloader):
        train_dataloader, test_dataloader = dataloader
        num_batches = len(train_dataloader)
        num_cv_batches = len(test_dataloader)
    
        print(f'dataset {k}, time {now}')
        finetuned_model.train()
        for j, (gene, pos, neg) in enumerate(train_dataloader):
            now = time.time()
            optimizer.zero_grad()            
            g, p, n = tokenize(gene, pos, neg, pretrained_model, device)
            loss = finetuned_model.forward(g, p, n)
            loss.backward()
            optimizer.step()
            # clean up
            it += len(gene)
            err = loss.item()
            print(f'dataset {k}, batch {j}, err {err}, total batches {num_batches}, time {now}')
        
            # write down summary stats (uncomment after debugging)
            # now = time.time()
            # if (now - last_summary_time) > summary_interval:
            #     cv_err = 0
            #     for j, (cv_gene, cv_pos, cv_neg) in enumerate(test_dataloader):
            #         gv, pv, nv = tokenize(cv_gene, cv_pos, cv_neg, 
            #                               pretrained_model, device)
            #         cv = finetuned_model.forward(gv, pv, nv)
            #         cv_err += cv.item()
            #         print(f'epoch {i}, batch {j}, cv_err {cv_err}, total batches {num_cv_batches}, time {now}')
            # 
            #         #clean up
            #         del cv
            #         if 'cuda' in device:
            #             torch.cuda.empty_cache()
            # 
            #         # cap the number of cross validations
            #         if j > 100: continue
            #             
            #     writer.add_scalar('test_error', cv_err, it)
            #     writer.add_scalar('train_error', err, it)
            #     last_summary_time = now
            del loss
            if 'cuda' in device:
                torch.cuda.empty_cache()
        
            if (now - last_checkpoint_time) > checkpoint_interval:
                suffix = str(i)
                model_path_ = model_path + suffix
                torch.save(finetuned_model.state_dict(), model_path_)
    return finetuned_model


def run(fasta_file, links_directory,
        checkpoint_path, data_dir, model_path, logging_path,
        training_column='Training',
        emb_dimension=100, num_neg=10,
        epochs=10, betas=(0.9, 0.95),
        batch_size=10, num_workers=10,
        summary_interval=1, checkpoint_interval=1000,
        device='cpu'):
    """ Train interaction model

    Parameters
    ----------
    fasta_file : filepath
        Fasta file of sequences of interest.
    links_directory : filepath
        Directory of links files. Each link file contains
        a table of tab delimited interactions
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
    
    interaction_directory = InteractionDataDirectory(
        fasta_file, links_directory, training_column,
        batch_size, num_workers, device
    )

    # train the fine_tuned model parameters
    finetuned_model = train(
        pretrained_model, interaction_directory,
        logging_path, 
        emb_dimension, epochs, betas,
        summary_interval, checkpoint_interval, 
        model_path, device)

    # save the last model checkpoint
    suffix = 'last'
    model_path_ = model_path + suffix
    torch.save(finetuned_model.state_dict(), model_path_)
    
