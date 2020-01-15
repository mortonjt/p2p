import os
import time
import datetime
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from fairseq.models.roberta import RobertaModel
from poplar.model.ppibinder import PPIBinder
from poplar.dataset.interactions import InteractionDataDirectory
from poplar.dataset.interactions import ValidationDataset
from poplar.dataset.interactions import NegativeSampler
from poplar.evaluate import pairwise_auc
from poplar.summary import (
    summarize_gradients, checkpoint, initialize_logging)
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from transformers import AdamW, WarmupLinearSchedule


def train(model, dataloader, optimizer, scheduler, writer,
          logging_path, summary_interval, checkpoint_interval):
    """ Trains a single epoch.

    Parameters
    ----------
    model : fairseq.models.roberta.RobertaModel
        Operon prediction model
    dataloader : torch.DataLoader
        Genomic dataloader
    optimizer : torch.nn.optimizer
        Optimizer for gradient descent calculations
    scheduler : torch.scheduler
        Schedules the learning rate
    writer : SummaryWriter
        Writes intermediate results to tensorboard
    logging_path : str
        Path of logging file
    summary_interval : int
        Number of seconds until a summary is written
    checkpoint_interval : int
        Number of seconds until a checkpoint is written

    Returns
    -------
    finetuned_model : poplar.model.operoner
    """
    last_summary_time, last_checkpoint_time = time.time(), time.time()

    # Estimate running time
    num_data = len(dataloader)
    t_total = num_data // gradient_accumulation_steps
    max_steps = max(1, max_steps)
    epochs = max_steps // num_data
    optimizer = AdamW(ppi_model.parameters(), lr=learning_rate)
    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=warmup_steps, t_total=t_total)

    # Initialize logging path
    writer = initialize_logging(logging_path=None)
    it = 0  # number of steps (iterations)

    # converts sequences to peptide encodings
    train_dataloader, test_dataloader, valid_dataloader = dataloader
    num_batches = len(train_dataloader)
    batch_size = train_dataloader.batch_size

    for j, (gene, pos, neg) in enumerate(train_dataloader):
        model.train()

        g = model.encode(gene)
        p = model.encode(pos)
        n = model.encode(neg)
        loss = model.forward(g, p, n)

        if torch.cuda.device_count() > 1:
            loss = loss.mean()
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        loss.backward()
        clip_grad_norm_(ppi_model.parameters(), clip_norm)

        it += len(gene)
        err = loss.item()

        # write down summary stats
        last_summary_time = summarize_gradients(
            model, summary_interval,
            last_summary_time, it, writer)

        # clean up
        del loss, g, p, n
        if 'cuda' in device:
            torch.cuda.empty_cache()

        # checkpoint
        last_checkpoint_time = checkpoint(
            ppi_model, logging_path, checkpoint_interval,
            last_checkpoint_time, writer)

        # accumulate gradients - so that we do backprop after loss
        # has been calculated on entire batch
        if j % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()

    return model


def fit(model, train_dataloader, test_dataloader,
        optimizer, scheduler, epochs, writer,
        logging_path, summary_interval, checkpoint_interval):
    """ Performs a model fit over multiple epochs """

    for _ in range(epochs):
        model = train(model, train_dataloader, optimizer, scheduler, writer,
                      logging_path, summary_interval, checkpoint_interval)

        # cross validation after each epoch
        tpr = pairwise_auc(model, test_dataloader,
                           'Main/test', it, writer, device)
        print(f'tpr: {tpr}')

    # we may want to have additional validation on an
    # additional holdout dataset.
    return model


def operon(training_directory, genome_metadata,
           checkpoint_path, data_dir, model_path, logging_path,
           gbk_ext=['*.gb', '*.genbank'],
           emb_dimension=100, num_neg=10,
           max_steps=10, learning_rate=5e-5,
           warmup_steps=1000, gradient_accumulation_steps=16,
           clip_norm=10, batch_size=4, num_workers=10,
           summary_interval=1, checkpoint_interval=1000,
           freeze_lm=False, device='cpu'):
    """ Train protein-protein interaction model

    Parameters
    ----------
    training_directory : filepath
        Directory of genome genbank files. Each link file contains
        a table of tab delimited interactions
    genome_metadata: filepath
        Two column tab-delimited file specifying which genomes
        are heldout out for training, testing and validation.
    emb_dimensions : int
        Number of embedding dimensions.
    num_neg : int
        Number of negative samples.
    max_steps : int
        Maximum number of steps to run for. Each step corresponds to
        the evaluation of a protein pair.
    learning_rate : float
        Learning rate of ADAM
    warmup_steps : int
        Number of warmup steps for scheduler
    gradient_accumulation_steps : int
        Number of steps before gradients are computed.
    checkpoint_path : path
        Path for roberta model.
    data_dir : path
        Path to data used for pretraining.
    model_path : path
        Path for finetuned model.
    logging_path : path
        Path for logging information.
    gbk_ext : list of str
        List of file extensions to recognize genbank files.
    clip_norm : float
        Clipping norm of gradients
    batch_size : int
        Number of protein triples to analyze in a given batch.
    summary_interval : int
        Number of seconds for a summary update.
    freeze_lm : bool
        Specifies if the weights of the language model should be
        frozen. Default=False
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
    pretrained_model.to(device)
    # the dimensionality of the roberta model
    roberta_dim = int(list(list(pretrained_model.parameters())[-1].shape)[0])

    # freeze the weights of the pre-trained model
    if freeze_lm:
        for param in pretrained_model.parameters():
            param.requires_grad = False

    operon_model = Operoner(roberta_dim, emb_dimension,
                            pretrained_model, device)
    operon_model.to(device)

    n_gpu = torch.cuda.device_count()
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print(os.environ["CUDA_VISIBLE_DEVICES"], 'devices available')
        print("Utilizing ", torch.cuda.device_count(), device)
        if n_gpu > 1:
            operon_model = torch.nn.DataParallel(operon_model)

    batch_size = max(batch_size, batch_size * n_gpu)

    # train the fine_tuned model parameters
    finetuned_model = fit(
        model=operon_model, directory_dataloader=interaction_directory,
        logging_path=logging_path, emb_dimension=emb_dimension,
        max_steps=max_steps, learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        clip_norm=clip_norm, summary_interval=summary_interval,
        checkpoint_interval=checkpoint_interval,
        model_path=model_path, device=device)

    # save the last model checkpoint
    suffix = 'last'
    model_path_ = model_path + suffix
    torch.save(finetuned_model.state_dict(), model_path_)
