import torch
import time
import datetime


def summarize_gradients(finetuned_model, summary_interval,
                        last_summary_time, writer):
    """ Summarize the model gradients.

    finetuned_model : torch.nn.Module
        Protein protein interaction model predictor
    summary_interval : int
        The frequency that summaries get recorded in seconds.
    last_summary_time : int
        The time of the last summary.
    writer : SummaryWrite
        Tensorboard summary writer.

    Returns
    -------
    now : time
        The current time

    TODO: add unittest
    """
    now = time.time()
    if (now - last_summary_time) > summary_interval:
        # add gradients to histogram
        writer.add_scalar('train_error', err, it)
        for name, param in finetuned_model.named_parameters():
            writer.add_histogram('grad/%s' %  name, param.grad, it)
    return now


def checkpoint(model, path, checkpoint_interval, last_checkpoint_time, writer):
    """ Save model at checkpoint.

    model : torch.nn.Module
        Protein protein interaction model predictor
    path : str
        Path to save the checkpoint.
    summary_interval : int
        The frequency that summaries get recorded in seconds.
    last_summary_time : int
        The time of the last summary.
    writer : SummaryWrite
        Tensorboard summary writer.

    TODO: add unittest
    """
    now = time.time()
    if (now - last_checkpoint_time) > checkpoint_interval:
        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        model_path_ = path + suffix
        # for parallel training
        try:
            state_dict = model.module.state_dict()
        except AttributeError:
            state_dict = model.state_dict()
        torch.save(state_dict, model_path_)
    return now


def initialize_logging(logging_path=None):
    """ Initializes tensorboard summary

    Parameters
    ----------
    logging_path : str
        Path of tensorboard logging path (optional)

    Returns
    -------
    writer : SummaryWriter
        Tensorboard summary writer.

    TODO: add unittest
    """
    if logging_path is None:
        basename = "logdir"
        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        logging_path = "_".join([basename, suffix])
    writer = SummaryWriter(logging_path)
