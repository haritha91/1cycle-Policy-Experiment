import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os

import classifier_1cycle
import classifier_fixedLR

#Plot the comparison
plt.figure(6)
plt.title("Train Accuracy Comparison")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(classifier_fixedLR.epoch_counter_train_base,classifier_fixedLR.train_acc_base,color = 'r', label="Fixed")
plt.plot(classifier_1cycle.epoch_counter_train,classifier_1cycle.train_acc,color = 'g', label="1cycle")
plt.legend()
plt.show()

plt.figure(7)
plt.title("Validation Accuracy Comparison")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(classifier_fixedLR.epoch_counter_val_base,classifier_fixedLR.val_acc_base,color = 'r', label="Fixed")
plt.plot(classifier_1cycle.epoch_counter_val,classifier_1cycle.val_acc,color = 'g', label="1cycle")
plt.legend()
plt.show()

plt.figure(8)
plt.title("Validation Loss Comparison")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(classifier_fixedLR.epoch_counter_val_base,classifier_fixedLR.val_loss_base,color = 'r', label="Fixed")
plt.plot(classifier_1cycle.epoch_counter_val,classifier_1cycle.val_loss,color = 'g', label="1cycle")
plt.legend()
plt.show()

plt.figure(9)
plt.title("Train Loss Comparison")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(classifier_fixedLR.epoch_counter_train_base,classifier_fixedLR.train_loss_base,color = 'r', label="Fixed")
plt.plot(classifier_1cycle.epoch_counter_train,classifier_1cycle.train_loss,color = 'g', label="1cycle")
plt.legend()
plt.show()

