"""Implement a simple bag-of-words model.

   $ python bow.py -e 100 -d 0.5 ../../data/classes 2>&1 | tee bow_log.txt

 - The test & dev sets must be very different from train, looking at the losses
 - Adding dropout (p = 0.5) helps to control the overfitting (test acc = 40%)
 - Adding better init gained another percent
"""

import argparse as ap
import time
from collections import OrderedDict
from typing import Dict

import torch
from sklearn import metrics
from torch import nn, optim
from torch.utils.data import DataLoader

import nlp


def cli() -> ap.ArgumentParser:
    p = ap.ArgumentParser()
    p.add_argument("-e", "--epochs", type=int, default=10, dest="EPOCHS")
    p.add_argument("-d", "--dropout", type=float, default=0.0, dest="DROPOUT")
    p.add_argument("DATADIR", type=str)
    return p


if __name__ == "__main__":
    args = cli().parse_args()
    print(args)

    DATA = nlp.ClassesDataset(args.DATADIR)

    # PyTorch model objects ----------------------------------------------------

    model = nlp.BOWClassifier(len(DATA.train.vocab), class_size=5, dropout=args.DROPOUT)
    print()
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, threshold=1e-2, patience=2, min_lr=1e-6, verbose=True
    )

    train_l = DataLoader(DATA.train, batch_size=16, num_workers=4)
    dev_l = DataLoader(DATA.dev, batch_size=16, num_workers=4)
    test_l = DataLoader(DATA.test, batch_size=1)

    # Training -----------------------------------------------------------------

    checkpoints: Dict[str, float] = OrderedDict()
    checkpoints["train_start"] = time.perf_counter()

    N_TRAIN_BATCHES = len(train_l)
    N_DEV_BATCHES = len(dev_l)

    print()
    print(f"Training for {args.EPOCHS} epochs")
    print(f"Dropout p = {args.DROPOUT:5.3f}")
    print(f"Total parameters = {sum(p.numel() for p in model.parameters()):,d}")

    last_train_loss = 1e20
    last_dev_loss = 1e20
    running_worse = 0
    histories = []

    for EPOCH in range(args.EPOCHS):
        print(f"\nEPOCH {1 + EPOCH}:")

        checkpoints["epoch_train_start"] = time.perf_counter()

        # Train loss, etc. -----------------------------------------------------

        train_loss = 0.0
        truths, preds = [], []

        model.train()
        for xbatch, ybatch in train_l:
            optimizer.zero_grad()
            outputs = model(xbatch)
            loss = criterion(outputs, ybatch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() / N_TRAIN_BATCHES
            preds.extend(outputs.detach().argmax(dim=-1).tolist())
            truths.extend(ybatch.detach().tolist())

        delta = time.perf_counter() - checkpoints["epoch_train_start"]
        print(f"Train: {train_loss:7.5f} in {delta:.03f} s", end=" ")
        print("↑" if train_loss > last_train_loss else "↓", end="")
        print(f"{100 * abs(train_loss/last_train_loss - 1):7.3f}%", end="")

        _acc = metrics.accuracy_score(truths, preds)
        _f1s = metrics.f1_score(truths, preds, average=None)
        print(f"{100 * _acc:6.1f}%", end=" ")
        print(("(" + 5 * "{:6.1f}" + ")").format(*[100 * f for f in _f1s]))

        histories.append({"train": {"loss": train_loss, "acc": _acc, "fscores": _f1s}})
        last_train_loss = train_loss
        checkpoints["epoch_dev_start"] = time.perf_counter()

        # Dev loss, etc. -------------------------------------------------------

        model.eval()
        with torch.no_grad():
            dev_loss = 0.0
            truths, preds = [], []
            for xbatch, ybatch in dev_l:
                outputs = model(xbatch)
                dev_loss += criterion(outputs, ybatch).item() / N_DEV_BATCHES
                preds.extend(outputs.detach().argmax(dim=-1).tolist())
                truths.extend(ybatch.detach().tolist())

        delta = time.perf_counter() - checkpoints["epoch_dev_start"]
        print(f"Dev:   {dev_loss:7.5f} in {delta:.03f} s", end=" ")
        print("↑" if dev_loss > last_dev_loss else "↓", end="")
        print(f"{100 * abs(dev_loss/last_dev_loss - 1):7.3f}%", end="")

        _acc = metrics.accuracy_score(truths, preds)
        _f1s = metrics.f1_score(truths, preds, average=None)
        print(f"{100 * _acc:6.1f}%", end=" ")
        print(("(" + 5 * "{:6.1f}" + ")").format(*[100 * f for f in _f1s]))

        histories[-1]["dev"] = {"loss": dev_loss, "acc": _acc, "fscores": _f1s}

        scheduler.step(dev_loss)

        # Early stopping
        if dev_loss >= last_dev_loss * 0.999:
            running_worse += 1
        else:
            running_worse = 0
        if running_worse >= 5:
            break

        last_dev_loss = dev_loss
        checkpoints["epoch_dev_end"] = time.perf_counter()

    checkpoints["train_end"] = time.perf_counter()
    checkpoints["test_start"] = time.perf_counter()

    # Test loss, etc. ----------------------------------------------------------

    N_TEST_BATCHES = len(test_l)
    model.eval()
    test_loss = 0.0
    truths, preds = [], []
    for xbatch, ybatch in test_l:
        outputs = model(xbatch)
        test_loss += criterion(outputs, ybatch).item() / N_TEST_BATCHES
        preds.extend(outputs.detach().argmax(dim=-1).tolist())
        truths.extend(ybatch.detach().tolist())

    delta = time.perf_counter() - checkpoints["test_start"]
    print(f"Test:  {test_loss:7.5f} in {delta:.03f} s", end=" ")
    print("         ", end="")
    _acc = metrics.accuracy_score(truths, preds)
    _f1s = metrics.f1_score(truths, preds, average=None)
    print(f"{100 * _acc:6.1f}%", end=" ")
    print(("(" + 5 * "{:6.1f}" + ")").format(*[100 * f for f in _f1s]))

    histories[-1]["test"] = {"loss": test_loss, "acc": _acc, "fscores": _f1s}

    dropout_str = f"{args.DROPOUT:5.3f}".replace(".", "")
    torch.save(model, f"bow_model_e{args.EPOCHS:04d}_d{dropout_str}.pt")
    torch.save(histories, f"bow_history_e{args.EPOCHS:04d}_d{dropout_str}.pt")
