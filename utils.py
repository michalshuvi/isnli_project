import os

import matplotlib.pyplot as plt


def create_metrics_graph(logfile_path, model_name, output_dir="loss_plots"):

    with open(logfile_path) as f:
        lines = f.readlines()

    train_metric = []
    i = 0
    for line in lines:
        if f"loss:" in line:
            if i < 5:
                i += 1
                continue
            loss = line.split()[1]
            train_metric.append(float(loss[:-1]))

    plt.plot(train_metric, label=f"train loss")
    plt.xlabel("batch")
    plt.ylabel('loss')
    plt.title(f"{model_name} loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{model_name}--loss"))
    plt.show()


def create_metrics_graph_by_epoch(logfile_path, model_name, output_dir="loss_plots"):

    with open(logfile_path) as f:
        lines = f.readlines()

    train_metric = []
    i = 0
    for line in lines:
        if f"allennlp.training.tensorboard_writer - loss" in line:
            loss = line.split("|")[1]
            train_metric.append(float(loss[:-1]))

    plt.plot(train_metric, label=f"train loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(f"{model_name} loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{model_name}--loss"))
    plt.show()
