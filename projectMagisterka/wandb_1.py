import wandb
import random

# there are some interesting examples what can be done at
# https://wandb.ai/bartekkawa2021-agh-university-of-science-and-technology/first-project/overview

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="first-project",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
    }
)

# simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()