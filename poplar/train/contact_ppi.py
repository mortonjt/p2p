import os
import time
import datetime
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from fairseq.models.roberta import RobertaModel
from poplar.transformer import RobertaConstrastiveHead
from poplar.dataset import InteractionDataDirectory
from poplar.util import encode, tokenize
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from transformers import AdamW, WarmupLinearSchedule


def contactmap_train(
        pretrained_model, dataloader,
        logging_path=None,
        emb_dimension=100, max_steps=0,
        learning_rate=5e-5, warmup_steps=1000,
        gradient_accumulation_steps=1,
        clip_norm=10., summary_interval=100, checkpoint_interval=100,
        model_path='model', device=None):
    """ Train the contact map prediction model.

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
    max_steps : int
        Maximum number of steps to run for. Each step corresponds to
        the evaluation of a protein pair. If this is zero, then it'll
        default to one epochs worth of protein pairs (ie one pass through
        all of the protein pairs in the training dataset).
    learning_rate : float
        Learning rate of ADAM
    warmup_steps : int
        Number of warmup steps for scheduler
    clip_norm : float
        Clipping norm of gradients
    summary_interval : int
        Number of steps before saving summary.
    checkpoint_interval : int
        Number of steps before saving checkpoint.
    device : str
        Name of device to run (specifies gpu or not)

    Returns
    -------
    finetuned_model : poplar.contactmap.ContactMapLinear

    """
    roberta_dim = int(list(list(pretrained_model.parameters())[-1].shape)[0])
    finetuned_model = ContactMapLinear(roberta_dim, emb_dimension)

    optimizer = AdamW(finetuned_model.parameters(), lr=learning_rate)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)
    loss_f = torch.nn.MSELoss()

    finetuned_model.to(device)
    n_gpu = torch.cuda.device_count()
    print(os.environ["CUDA_VISIBLE_DEVICES"], 'devices available')
    print("Utilizing ", torch.cuda.device_count(), device)
    if n_gpu > 1:
        finetuned_model = torch.nn.DataParallel(finetuned_model)

    # Initialize logging path
    if logging_path is None:
        basename = "logdir"
        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        logging_path = "_".join([basename, suffix])
    it = 0
    writer = SummaryWriter(logging_path)

    # metrics to report
    err, cv_err, batch_size = 0, 0, 0

    now = time.time()
    last_now = time.time()

    print('Number of epochs', epochs)
    for e in range(epochs):
        for k, dataloader in enumerate(directory_dataloader):
            finetuned_model.train()
            train_dataloader, test_dataloader = dataloader
            num_batches = len(train_dataloader)
            num_cv_batches = len(test_dataloader)
            batch_size = train_dataloader.batch_size

            print(f'dataset {k}, num_batches {num_batches}, num_cvs {num_cv_batches}, '
                  f'seconds / batch {now - last_now}')
            for j, (gene, contacts) in enumerate(train_dataloader):
                last_now = now
                now = time.time()
                g = pretrained_model.extract_features(encode(x))
                pred = finetuned_model.forward(g)
                loss = loss_f(pred, contacts)

                if n_gpu > 1:
                    loss = loss.mean()
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                loss.backward()
                clip_grad_norm_(finetuned_model.parameters(), clip_norm)

                it += len(gene)
                err = loss.item()
                print(f'epoch {e}, dataset {k}, batch {j}, err {err}, total batches {num_batches}, '
                      f'seconds / batch {now - last_now}')

                # write down summary stats
                if (now - last_summary_time) > summary_interval:
                    writer.add_scalar('train_error', err, it)
                    # add gradients to histogram
                    for name, param in finetuned_model.named_parameters():
                        writer.add_histogram('grad/%s' %  name, param.grad, it)
                    last_summary_time = now

                # clean up
                del loss, g
                if 'cuda' in device:
                    torch.cuda.empty_cache()

                if (now - last_checkpoint_time) > checkpoint_interval:
                    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
                    model_path_ = model_path + suffix
                    # for parallel training
                    try:
                        state_dict = finetuned_model.module.state_dict()
                    except AttributeError:
                        state_dict = finetuned_model.state_dict()
                    torch.save(state_dict, model_path_)
                    last_checkpoint_time = now

                # accumulate gradients - so that we do backprop after loss
                # has been calculated on entire batch
                if j % gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    finetuned_model.zero_grad()

            # cross validation after each dataset is processed
            with torch.no_grad():
                cv_loss = 0
                for j, (cv_gene, cv_contacts) in enumerate(test_dataloader):
                    cv_pred = pretrained_model.extract_features(encode(x))
                    cv_loss += loss_f(cv_pred, cv_contacts).item()

                    #clean up
                    del cv_pred, cv_contacts, cv_loss
                    if 'cuda' in device:
                        torch.cuda.empty_cache()

                if len(test_dataloader) > 0:
                    cv_err = cv_err / len(test_dataloader)
                    tpr = tpr / len(test_dataloader)
                    print(f'epoch {e}, dataset {k}, cv_err {cv_err}, '
                          f'total batches {num_cv_batches}, '
                          f'seconds / batch {now - last_now}')
                    writer.add_scalar('test_error', cv_loss, it)

    # save hparams
    writer.add_hparams(
        hparam_dict={
            'emb_dimension': emb_dimension,
            'learning_rate': learning_rate,
            'warmup_steps': warmup_steps,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'batch_size': batch_size
            },
        metric_dict={
            'train_error': err,
            'test_error': cv_err
        }
    )

    writer.close()
    return finetuned_model
