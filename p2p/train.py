from transformers.modeling_roberta import RobertaModel
from fairseq.modules import TransformerSentenceEncoderLayer
from fairseq.models.roberta import RobertaModel
from p2p.transformer import RobertaConstrastiveHead
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time
import datetime
import torch


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
    tokens = list(map(lambda i: dictionary[i], x))
    tokens = torch.Tensor(tokens)
    tokens = tokens.long()
    return tokens


def train(pretrained_model,
          train_dataloader, test_dataloader,
          emb_dimension=100, epochs=10, betas=(0.9, 0.95), clip_norm=10.,
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
    epochs : int
        Number of epochs for training.
    betas : tuple of float
        Adam beta parameters.
    clip_norm : float
        Gradient clipping for numerical stability.
    summary_interval : int
        Number of steps before saving summary.
    device : str
        Name of device to run (specifies gpu or not)

    Returns
    -------
    finetuned_model : p2p.transformer.RobertaConstrastiveHead
    """
    last_summary_time = 0
    finetuned_model = RobertaConstrastiveHead(roberta_dim, emb_dimension)
    optimizer = optim.Adamax(finetuned_model.parameters(),
                             betas=betas)

    # Initialize logging path
    if logging_path is None:
        basename = "logdir"
        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        self.logging_path = "_".join([basename, suffix])
    else:
        self.logging_path = logging_path
    writer = SummaryWriter(self.logging_path)
    ce_loss = torch.nn.CrossEntropyLoss()
    for i in tqdm(range(epochs)):
        now = time.time()
        finetuned_model.train()
        for gene, pos, neg in train_dataloader:

            gene = pretrained_model.extract_features(gene)
            pos = pretrained_model.extract_features(pos)
            neg = pretrained_model.extract_features(neg)

            optimizer.zero_grad()
            inp = inp.to(device, non_blocking=True)
            out = out.to(device, non_blocking=True)
            loss = finetuned_model.forward(gene, pos, neg)
            loss.backward()
            optimizer.step()

        # write down summary stats
        now = time.time()
        if now - last_summary_time > summary_interval:
            err = []
            for gene, pos, neg in test_dataloader:
                gene = pretrained_model.extract_features(gene)
                pos = pretrained_model.extract_features(pos)
                neg = pretrained_model.extract_features(neg)
                loss = finetuned_model.forward(gene, pos, neg)
                cv = ce_loss(pred, out)
                err.append(cv)
            err = torch.mean(err)
            writer.add_scalar('test_error', err, i)
            writer.add_scalar('train_error', loss, i)
            last_summary_time = now
    return finetuned_model


def run(fasta_file, links_file,
        checkpoint_path, data_dir, model_path, logging_path,
        training_column='Training',
        num_neg=10, batch_size=10, num_workers=10, arm_the_gpu=True):
    """ Train interaction model.

    Parameters
    ----------
    fasta_file : filepath
        Fasta file of sequences of interest.
    link_file : filepath
        Table of tab delimited interactions
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
    arm_the_gpu : bool
        Use a gpu or not.
    """
    # An example of how to load your own roberta model
    # roberta_checkpoint_path = 'checkpoints/uniref50'
    # data_dir = 'data/uniref50'
    # pytorch_dump_folder_path = 'checkpoints/roberta_TF_dump'
    # classification_head = False
    # roberta = FairseqRobertaModel.from_pretrained(
    #     roberta_checkpoint_path, 'checkpoint_best.pt', data_dir)

    # the dimensionality of the roberta model
    roberta_dim = list(list(roberta.parameters())[-1].shape)[0]
    classification_head = False
    pretrained_model = RobertaModel.from_pretrained(
        checkpoint_path, 'checkpoint_best.pt', data_dir)

    train_data, test_data, valid_data = parse(
        fasta_file, links_file, training_column,
        batch_size, num_workers, arm_the_gpu)

    # train the fine_tuned model parameters
    finetuned_model = train(
        pretrained_model, train_data, test_data,
        emb_dimension, epochs, betas,
        summary_interval, device)

    # evaluate accuracy on validation dataset
    for gene, pos, neg in valid_data:
        gene = pretrained_model.extract_features(gene)
        pos = pretrained_model.extract_features(pos)
        neg = pretrained_model.extract_features(neg)
        pred = finetuned_model.predict(gene, pos)
        predicted = torch.round(pred)
        total += labels.size(0)
        correct += (predicted == 1.0).sum().item()

    # TODO: There needs to be a list of proteins where
    # there is for sure no interaction.

    l = len(valid_data)
    acc = 100 * correct / total
    print(f'Accuracy of model on the {l} test interactions: {acc}')

    # save the model checkpoint
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    torch.save(finedtuned_model.state_dict(),
               os.path.join(model_path, 'checkpoint_' + suffix))
