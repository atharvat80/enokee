import argparse
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataloader import DataLoader
from src.model import EnokeeConfig, EnokeeEncoder
from src.tokenizer import LUKETokenizer
from src.utils import get_num_param_and_model_size, load_checkpoint, save_checkpoint

torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    output_dir,
    dataloader,
    model,
    optimizer,
    epochs,
    scheduler=None,
    logger=None,
    save_every=100,
    previous_state=(0, 0),
    clip_val=5,
):
    print("INFO: Using device {}".format(str(device)))
    print("INFO: Starting training, press CTRL+C to stop")
    # print model details
    get_num_param_and_model_size(model)

    # setup
    # model.to(device)
    model.train()
    step, epoch = previous_state
    criterion = torch.nn.NLLLoss(ignore_index=-1)
    softmax = torch.nn.functional.log_softmax
    tokenizer = LUKETokenizer()
    while epoch < epochs:
        total = 42299053 - dataloader.iterator._currow
        pbar = tqdm(dataloader, desc=f"[EPOCH {epoch}|{epochs}]", total=total)
        for sentences, spans, targets in pbar:
            # zero grad
            optimizer.zero_grad()
            # forward pass
            inputs = tokenizer(sentences, spans).to(device)
            outputs = model(**inputs)
            outputs = outputs.view(-1, 51000)
            # loss and backward pass
            loss = criterion(softmax(outputs, dim=1), targets.flatten().to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            pbar.set_postfix({"loss": loss.item()})
            # save checkpoints
            if step % save_every == 0:
                save_checkpoint(
                    output_dir, step, epoch, dataloader, model, optimizer, scheduler
                )
                # log loss
                if logger is not None:
                    logger.add_scalar("Loss/Train", loss.item(), step)
            # global step
            step += 1
            pbar.update(dataloader.batch_size)
        # global epoch
        epoch += 1


def main(
    output_dir, 
    dataset_path, 
    default_output_dir, 
    batch_size, 
    epochs, 
    compile_model=False
):
    # initialise dataloader, model, optimizer and (optionally schedular)
    step = 0
    epoch = 0
    dataloader = None
    config = EnokeeConfig()
    model = EnokeeEncoder(config).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = None

    # load checkpoints if exist
    if output_dir is not None:
        print("INFO: Loading checkpoints")
        output_dir = Path(output_dir)
        (
            step,
            epoch,
            dataloader_state_dict,
            model_state_dict,
            optimizer_state_dict,
            scheduler_state_dict,
        ) = load_checkpoint(output_dir, device)
        # load dataloader_state_dict
        dataloader = DataLoader.from_state_dict(dataloader_state_dict)
        dataloader.batch_size = batch_size
        # load model_state_dict
        model.load_state_dict(model_state_dict, strict=False)
        # load optimizer_state_dict
        optimizer.load_state_dict(optimizer_state_dict)
        # load scheduler_state_dict
        if scheduler is not None and scheduler_state_dict is not None:
            scheduler.load_state_dict(scheduler_state_dict)

    elif dataset_path is not None:
        print("INFO: Loading dataset")
        dataset_path = Path(dataset_path)
        output_dir = Path(default_output_dir)
        output_dir.mkdir(exist_ok=True)
        if dataset_path.exists():
            dataloader = DataLoader(dataset_path, batch_size=batch_size)
        else:
            raise FileNotFoundError("Dataset does not exist at the provided path")

    else:
        raise ValueError(
            "No arguments provided, run `python train.py --help to list arguments"
        )

    # initialise summary writer
    logger = SummaryWriter(output_dir)
    # train
    if compile_model and torch.__version__.startswith("2"):
        try:
            model = torch.compile(model)
        except RuntimeError:
            print("WARN: Could not compile model, unsupported platform")

    try:
        train(
            output_dir,
            dataloader,
            model,
            optimizer,
            epochs,
            scheduler,
            logger,
            previous_state=(step, epoch),
        )
    except KeyboardInterrupt:
        if logger is not None:
            logger.close()
        print("Stopped training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enokee training script")
    parser.add_argument("--dataset_path", type=str, help="Dataset Path")
    parser.add_argument("--output_dir", type=str, help="Checkpoints & logs directory")
    parser.add_argument("--default_output_dir", type=str, default="./output",
                        help="Default checkpoints & logs directory",)
    parser.add_argument("--batch_size", type=int, default=16, required=False,
                        help="Batch size (default 16)",)
    parser.add_argument("--epochs", type=int, default=5, required=False, 
                        help="Train epochs")

    args = parser.parse_args()

    main(
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
        default_output_dir=args.default_output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
