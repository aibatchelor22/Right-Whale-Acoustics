import torch


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):

    model.train()

    running_loss = 0.0

    for x, y in loader:

        x = x.to(device)

        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)

        loss = criterion(
            logits,
            y
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)

            y = y.to(device)

            logits = model(x)

            loss = criterion(
                logits,
                y
            )

            running_loss += loss.item()

    return running_loss / len(loader)
