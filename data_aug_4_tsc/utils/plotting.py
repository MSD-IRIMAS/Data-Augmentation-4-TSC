"""Plotting functionalities."""

import os

import matplotlib.pyplot as plt
import numpy as np
from aeon.distances import pairwise_distance
from matplotlib.lines import Line2D


def plot_generated_only(
    output_directory: str,
    file_name: str,
    xgenerated: np.ndarray,
    lw=3,
):
    """Plot the generations only.

    Parameters
    ----------
    output_directory: str,
        The output directory where the figure is saved.
    file_name: str,
        The name of the file to save, without the pdf extension.
    xgenerated: np.ndarray, shape (n_samples, length_TS, n_channels),
        The generated samples.
    lw: int, default = 3,
        The line width.
    """
    xgenerated = np.array(xgenerated)
    n_samples = int(len(xgenerated))
    n_channels = int(len(xgenerated[0, 0]))

    fig, ax = plt.subplots(nrows=1, ncols=n_channels, figsize=(15 * n_channels, 10 * 1))

    for c in range(n_channels):
        try:
            ax[c].plot(xgenerated[0, :, c], lw=lw)
            ax[c].set_title("Channel #" + str(c), fontsize=15)
        except TypeError:
            ax.plot(xgenerated[0, :, c], lw=lw)
            ax.set_title("Channel #" + str(c), fontsize=15)

    if n_samples > 1:
        for c in range(n_channels):
            try:
                ax[c].plot(xgenerated[:, :, c].T, lw=lw)
            except TypeError:
                ax.plot(xgenerated[:, :, c].T, lw=lw)

    fig.suptitle("Generated Samples", fontsize=15)

    fig.savefig(os.path.join(output_directory, file_name + ".pdf"), bbox_inches="tight")

    fig.clear()
    plt.clf()


def plot_same_axes(
    output_directory: str,
    file_name: str,
    xgenerated: np.ndarray,
    xreal: np.ndarray,
    gen_color: str = "red",
    real_color: str = "blue",
    lw=3,
):
    """Plot the generations same axes as real.

    Parameters
    ----------
    output_directory: str,
        The output directory where the figure is saved.
    file_name: str,
        The name of the file to save, without the pdf extension.
    xgenerated: np.ndarray, shape (n_samples, length_TS, n_channels),
        The generated samples.
    xreal: np.ndarray, shape (m_samples, length_TS, n_channels),
        The real samples.
    gen_color: str, default = "red",
        A matlplotlib color for the generated samples.
    real_color: str, default = "blue",
        A matlplotlib color for the real samples.
    lw: int, default = 3,
        The line width.
    """
    xgenerated = np.array(xgenerated)
    xreal = np.array(xreal)
    n_channels = int(len(xgenerated[0, 0]))

    fig, ax = plt.subplots(nrows=1, ncols=n_channels, figsize=(15 * n_channels, 10 * 1))

    for c in range(n_channels):
        try:
            ax[c].plot(xreal[:, :, c].T, color=real_color, lw=lw)
            ax[c].plot(xgenerated[:, :, c].T, color=gen_color, lw=lw, alpha=0.4)

            ax[c].set_title(
                "Channel #" + str(c) + " - Real & Generated Samples", fontsize=15
            )

            ax[c].legend(
                handles=[
                    Line2D([], [], color=real_color, lw=lw, label="real"),
                    Line2D(
                        [], [], color=gen_color, lw=lw, label="generation", alpha=0.4
                    ),
                ],
                prop={"size": 15},
            )

        except (IndexError, TypeError):
            ax.plot(xreal[:, :, c].T, color=real_color, lw=lw)
            ax.plot(xgenerated[:, :, c].T, color=gen_color, lw=lw, alpha=0.4)

            ax.set_title(
                "Channel #" + str(c) + " - Real & Generated Samples", fontsize=15
            )

            ax.legend(
                handles=[
                    Line2D([], [], color=real_color, lw=lw, label="real"),
                    Line2D(
                        [], [], color=gen_color, lw=lw, label="generation", alpha=0.4
                    ),
                ],
                prop={"size": 15},
            )

    fig.savefig(os.path.join(output_directory, file_name + ".pdf"), bbox_inches="tight")

    fig.clear()
    plt.clf()


def plot_parallel_axes(
    output_directory: str,
    file_name: str,
    xgenerated: np.ndarray,
    xreal: np.ndarray,
    gen_color: str = "red",
    real_color: str = "blue",
    lw=3,
):
    """Plot the generations parallel to real.

    Parameters
    ----------
    output_directory: str,
        The output directory where the figure is saved.
    file_name: str,
        The name of the file to save, without the pdf extension.
    xgenerated: np.ndarray, shape (n_samples, length_TS, n_channels),
        The generated samples.
    xreal: np.ndarray, shape (m_samples, length_TS, n_channels),
        The real samples.
    gen_color: str, default = "red",
        A matlplotlib color for the generated samples.
    real_color: str, default = "blue",
        A matlplotlib color for the real samples.
    lw: int, default = 3,
        The line width.
    """
    xgenerated = np.array(xgenerated)
    xreal = np.array(xreal)
    n_channels = int(len(xgenerated[0, 0]))

    fig, ax = plt.subplots(nrows=2, ncols=n_channels, figsize=(15 * n_channels, 10 * 1))

    for c in range(n_channels):
        try:
            ax[0, c].plot(xreal[:, :, c].T, color=real_color, lw=lw)
            ax[1, c].plot(xgenerated[:, :, c].T, color=gen_color, lw=lw)

            ax[0, c].set_title("Channel #" + str(c) + " - Real Samples", fontsize=15)
            ax[1, c].set_title(
                "Channel #" + str(c) + " - Generated Samples", fontsize=15
            )
        except IndexError:
            ax[0].plot(xreal[:, :, c].T, color=real_color, lw=lw)
            ax[1].plot(xgenerated[:, :, c].T, color=gen_color, lw=lw)

            ax[0].set_title("Channel #" + str(c) + " - Real Samples", fontsize=15)
            ax[1].set_title("Channel #" + str(c) + " - Generated Samples", fontsize=15)

    fig.savefig(os.path.join(output_directory, file_name + ".pdf"), bbox_inches="tight")

    fig.clear()
    plt.clf()


def plot_generated_with_nn(
    output_directory: str,
    file_name: str,
    xgenerated: np.ndarray,
    xreal: np.ndarray,
    gen_color: str = "red",
    real_color: str = "blue",
    lw=3,
    n_neighbors: int = 1,
    distance_used: str = "euclidean",
):
    """Plot the generations with their nearest neighbor to real.

    Parameters
    ----------
    output_directory: str,
        The output directory where the figure is saved.
    file_name: str,
        The name of the file to save, without the pdf extension.
    xgenerated: np.ndarray, shape (n_samples, length_TS, n_channels),
        The generated samples.
    xreal: np.ndarray, shape (m_samples, length_TS, n_channels),
        The real samples.
    gen_color: str, default = "red",
        A matlplotlib color for the generated samples.
    real_color: str, default = "blue",
        A matlplotlib color for the real samples.
    lw: int, default = 3,
        The line width.
    n_neighbors: int, default = 1,
        The number of nearest neighbors from xreal.
    distance_used: str, default = "euclidean",
        An aeon distance to find nearest neighbor.
    """
    xgenerated = np.array(xgenerated)
    xreal = np.array(xreal)
    n_samples = min(5, int(len(xgenerated)))
    n_channels = int(len(xgenerated[0, 0]))

    chosen_indices_to_plot = np.random.choice(
        np.arange(len(xgenerated)), size=n_samples, replace=False
    )

    fig, ax = plt.subplots(
        nrows=n_samples, ncols=n_channels, figsize=(15 * n_channels, 10 * n_samples)
    )

    distance_matrix = pairwise_distance(
        x=np.transpose(xgenerated[chosen_indices_to_plot], [0, 2, 1]),
        y=np.transpose(xreal, [0, 2, 1]),
        method=distance_used,
    )

    neighbors = np.argsort(distance_matrix, axis=1)[:, :n_neighbors]

    for n in range(len(chosen_indices_to_plot)):
        neighbors_ = xreal[neighbors[n]]

        for c in range(n_channels):
            try:
                ax[n, c].plot(neighbors_[:, :, c].T, color=real_color, lw=lw)
                ax[n, c].plot(
                    xgenerated[chosen_indices_to_plot[n], :, c], color=gen_color, lw=lw
                )

                ax[n, c].set_title("Channel #" + str(c), fontsize=15)

                ax[n, c].legend(
                    [
                        Line2D([], [], lw=lw, color=real_color),
                        Line2D([], [], lw=lw, color=gen_color),
                    ],
                    ["Neighbors' Channel #" + str(c), "Generated's Channel #" + str(c)],
                    prop={"size": 15},
                )
            except IndexError:
                ax[n].plot(neighbors_[:, :, c].T, color=real_color, lw=lw)
                ax[n].plot(
                    xgenerated[chosen_indices_to_plot[n], :, c], color=gen_color, lw=lw
                )

                ax[n].set_title(
                    "Sample #" + str(n) + " - Channel #" + str(c), fontsize=15
                )

                ax[n].legend(
                    [
                        Line2D([], [], lw=lw, color=real_color),
                        Line2D([], [], lw=lw, color=gen_color),
                    ],
                    ["Neighbors' Channel #" + str(c), "Generated's Channel #" + str(c)],
                    prop={"size": 15},
                )

    fig.savefig(os.path.join(output_directory, file_name + ".pdf"), bbox_inches="tight")
