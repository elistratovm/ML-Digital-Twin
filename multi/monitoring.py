import random
from IPython import display
import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm


class Monitor:
    def __init__(self):
        if hasattr(tqdm.tqdm, '_instances'):
            [*map(tqdm.tqdm._decr_instances, list(tqdm.tqdm._instances))]

        self.train_loss_curve = []
        self.test_loss_curve = []

    def add_train_loss(self, value):
        self.train_loss_curve.append(value)

    def add_test_loss(self, value):
        self.test_loss_curve.append(value)

    def show(self):
        display.clear_output(wait=True)
        plt.figure()
        
        plt.plot(self.train_loss_curve, label="Train Loss", marker="o")
        plt.plot(self.test_loss_curve, label="Test Loss", marker="s")
        
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Test Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def show2(self):
        display.clear_output(wait=True)
        
        fig, axes = plt.subplots(1, 1, figsize=(12, 5))
        
        axes[0].plot(self.train_loss_curve, label="Train Loss", marker="o")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss Curve")
        axes[0].legend()
        axes[0].set_ylim((-0.01, 1.1*max(self.train_loss_curve)))
        axes[0].grid(True)

        axes[1].plot(self.test_loss_curve, label="Test Loss", marker="s")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Test Loss Curve")
        axes[1].legend()
        axes[1].set_ylim((-0.01, 1.1*max(self.test_loss_curve)))
        axes[1].grid(True)

        plt.tight_layout()
        display.display(plt.gcf())
        plt.close()