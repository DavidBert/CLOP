from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch

criterion_classifier = nn.NLLLoss(reduction="mean")


def train(
    model,
    optimizer,
    trainloader,
    epochs=30,
    test_loader=None,
    m_eval=False,
    scheduler=None,
):
    t = tqdm(range(epochs))
    for epoch in t:
        corrects = 0
        total = 0
        for x, y in trainloader:
            loss = 0
            x = x.cuda()
            y = y.cuda()
            y_hat = model(x)

            loss += criterion_classifier(y_hat, y)
            _, predicted = y_hat.max(1)
            corrects += predicted.eq(y).sum().item()
            total += y.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(
                f"epoch:{epoch} current accuracy:{round(corrects / total * 100, 2)}%"
            )
        if scheduler is not None:
            scheduler.step()
        if test_loader is not None:
            test(model, test_loader, m_eval)
    return corrects / total


def test(model, testloader, m_eval=True):
    if m_eval:
        model.eval()
    with torch.no_grad():
        corrects = 0
        total = 0
        for x, y in testloader:
            x = x.cuda()
            y = y.cuda()
            y_hat = model(x)

            _, predicted = y_hat.max(1)
            corrects += predicted.eq(y).sum().item()
            total += y.size(0)
        print(f"test accuracy: {round(corrects / total * 100, 2)}%")
        model.train()
        return corrects / total


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_mixup(
    model,
    optimizer,
    trainloader,
    epochs=30,
    test_loader=None,
    m_eval=False,
    scheduler=None,
    alpha=1,
):
    t = tqdm(range(epochs))
    for epoch in t:
        corrects = 0
        total = 0
        for x, y in trainloader:
            loss = 0
            x = x.cuda()
            y = y.cuda()
            mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=alpha)
            y_hat = model(x)

            loss += mixup_criterion(criterion_classifier, y_hat, y_a, y_b, lam)
            _, predicted = y_hat.max(1)
            corrects += predicted.eq(y).sum().item()
            total += y.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(
                f"epoch:{epoch} current accuracy:{round(corrects / total * 100, 2)}%"
            )
        if scheduler is not None:
            scheduler.step()
        if test_loader is not None:
            test(model, test_loader, m_eval)
    return corrects / total
