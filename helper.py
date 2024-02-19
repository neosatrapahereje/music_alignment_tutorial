#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Helper methods for plotting and visualizing alignments
"""
import os
from shutil import make_archive
from typing import List, Optional, Tuple, Union

import numpy as np
import partitura as pt
from matplotlib import lines
from matplotlib import pyplot as plt
from partitura.io.exportparangonada import alignment_dicts_to_array
from partitura.performance import PerformanceLike
from partitura.score import ScoreLike
from partitura.utils.misc import PathLike
from scipy.sparse import csc_matrix
from sklearn.datasets import make_blobs

# Define random state for reproducibility
RNG = np.random.RandomState(1984)


def generate_example_sequences(
    lenX: int = 100,
    centers: int = 3,
    n_features: int = 5,
    maxreps: int = 4,
    minreps: int = 1,
    noise_scale: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates example pairs of related sequences. Sequence X are samples of
    an K-dimensional space around a specified number of centroids.
    Sequence Y is a non-constant "time-streched" version of X with some
    noise added.

    Parameters
    ----------
    lenX : int
        Number of elements in the X sequence
    centers: int
        Number of different centers ("classes") that the elements
        of the sequences represent
    n_features: int
        Dimensionality of the features ($K$) in the notation of the
        Notebook
    noise_scale: float
        Scale of the noise

    Returns
    -------
    X : np.ndarray
        Sequence X (a matrix where each row represents
        an element of the sequence)
    Y: np.ndarray
        Sequence Y
    ground_truth_path: np.ndarray
        Alignment between X and Y where the first column represents the indices
        in X and the second column represents the corresponding index in Y.
    """

    X, _ = make_blobs(n_samples=lenX, centers=centers, n_features=n_features)
    # Time stretching X! each element in sequence X is
    # repeated a random number of times
    # and then we add some noise to spice things up :)

    if minreps == maxreps:
        n_reps = np.ones(len(X), dtype=int) * minreps
    else:
        n_reps = RNG.randint(minreps, maxreps, len(X))
    y_idxs = [rp * [i] for i, rp in enumerate(n_reps)]
    y_idxs = np.array([el for reps in y_idxs for el in reps], dtype=int)
    # Add a bias, so that Y has a different "scaling" than X
    Y = X[y_idxs]
    # add some noise
    Y += noise_scale * RNG.randn(*Y.shape)
    ground_truth_path = np.column_stack((y_idxs, np.arange(len(Y))))
    return X, Y, ground_truth_path


def plot_alignment(
    X: np.ndarray,
    Y: np.ndarray,
    alignment_path: np.ndarray,
) -> None:
    """
    Visualize alignment between two sequences.

    Parameters
    ----------
    X : np.ndarray
        Reference sequence (a matrix where each row represents an element of
        the sequence)
    Y : np.ndarray
        The sequence we want to align to X.
    alignment_path : np.ndarray
        A 2D array where each row corresponds to the indices in array X and its
        corresponding element in X.
    """
    vmax = max(max(abs(X.max()), abs(X.min())), max(abs(Y.max()), abs(Y.min())))
    fig, axes = plt.subplots(2, sharex=True)
    axes[0].imshow(
        X.T,
        cmap="gray",
        origin="lower",
        aspect="equal",
        interpolation=None,
        vmax=vmax,
        vmin=-vmax,
    )
    axes[0].set_ylabel(r"$\mathbf{X}$")
    axes[1].imshow(
        Y.T,
        cmap="gray",
        origin="lower",
        aspect="equal",
        interpolation=None,
        vmax=vmax,
        vmin=-vmax,
    )
    axes[1].set_ylabel(r"$\mathbf{Y}$")
    axes[0].set_xlim((-1, max(len(X), len(Y)) + 1))
    axes[1].set_xlim((-1, max(len(X), len(Y)) + 1))
    axes[0].set_ylim((-1, X.shape[1] + 1))
    axes[1].set_ylim((-1, Y.shape[1] + 1))

    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].spines["bottom"].set_visible(False)
    axes[0].spines["left"].set_visible(False)
    axes[0].get_xaxis().set_ticks([])
    axes[0].get_yaxis().set_ticks([])
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].spines["bottom"].set_visible(False)
    axes[1].spines["left"].set_visible(False)
    axes[1].get_xaxis().set_ticks([])
    axes[1].get_yaxis().set_ticks([])

    for ref_idx, perf_idx in alignment_path:
        # Add line from one subplot to the other
        xyA = [ref_idx, 0]
        axes[0].plot(*xyA)
        xyB = [perf_idx, Y.shape[1] - 0.75]
        axes[1].plot(*xyB)
        transFigure = fig.transFigure.inverted()
        coord1 = transFigure.transform(axes[0].transData.transform(xyA))
        coord2 = transFigure.transform(axes[1].transData.transform(xyB))
        line = lines.Line2D(
            (coord1[0], coord2[0]),  # xdata
            (coord1[1], coord2[1]),  # ydata
            transform=fig.transFigure,
            color="red",
            linewidth=0.5,
        )
        fig.lines.append(line)

    plt.show()


def compute_pitch_class_pianoroll(
    note_info: Union[
        pt.score.ScoreLike,
        pt.performance.PerformanceLike,
        np.ndarray,
        csc_matrix,
    ],
    normalize: bool = True,
    time_unit: str = "auto",
    time_div: int = "auto",
    onset_only: bool = False,
    note_separation: bool = False,
    time_margin: int = 0,
    remove_silence: bool = True,
    end_time: Optional[float] = None,
    binary: bool = False,
) -> np.ndarray:
    """
    Compute a pitch class piano roll.

    Parameters
    ----------

    """
    pianoroll = None
    if isinstance(note_info, csc_matrix):
        pianoroll = note_info

    if pianoroll is None:

        pianoroll = pt.utils.compute_pianoroll(
            note_info=note_info,
            time_unit=time_unit,
            time_div=time_div,
            onset_only=onset_only,
            note_separation=note_separation,
            pitch_margin=-1,
            time_margin=time_margin,
            return_idxs=False,
            piano_range=False,
            remove_drums=True,
            remove_silence=remove_silence,
            end_time=end_time,
        )

    pc_pianoroll = np.zeros((12, pianoroll.shape[1]), dtype=float)
    for i in range(int(np.ceil(128 / 12))):
        pr_slice = pianoroll[i * 12 : (i + 1) * 12, :].toarray().astype(float)
        pc_pianoroll[: pr_slice.shape[0], :] += pr_slice

    if binary:
        # only show active pitch classes
        pc_pianoroll[pc_pianoroll > 0] = 1

    if normalize:
        norm_term = pc_pianoroll.sum(0)
        # avoid dividing by 0 if a slice is empty
        norm_term[np.isclose(norm_term, 0)] = 1
        pc_pianoroll /= norm_term

    return pc_pianoroll


def compute_pitch_class_distribution_windowed(piano_roll, time_div, win_size):
    n_windows = int(np.ceil(piano_roll.shape[1] / (time_div * win_size)))

    window_size = win_size * time_div

    observations = np.zeros((n_windows, 12))
    for win in range(n_windows):
        idx = slice(win * window_size, (win + 1) * window_size)
        segment = piano_roll[:, idx].sum(1)
        dist = np.zeros(12)
        pitch_idxs = np.where(segment != 0)[0]
        for pix in pitch_idxs:
            dist[pix % 12] += segment[pix]
        dist /= dist.sum()
        observations[win] = dist

    return observations


def evaluate_alignment_notewise(
    prediction: List[dict],
    ground_truth: List[dict],
    types: List[str] = ["match", "deletion", "insertion"],
) -> Tuple[float, float, float]:
    """
    Evaluate Alignments.

    This methods evaluates note-level alignments by computing the
    precision, recall and F-score.

    Parameters
    ----------
    prediction: List of dicts
        List of dictionaries containing the predicted alignments
    ground_truth:
        List of dictionaries containing the ground truth alignments
    types: List of strings
        List of alignment types to consider for evaluation
        (e.g ['match', 'deletion', 'insertion']

    Returns
    -------
    precision: float
       The precision
    recall: float
        The recall
    f_score: float
       The F score
    """

    sanitize_alignment(prediction)
    sanitize_alignment(ground_truth)
    pred_filtered = list(filter(lambda x: x["label"] in types, prediction))
    gt_filtered = list(filter(lambda x: x["label"] in types, ground_truth))

    filtered_correct = [pred for pred in pred_filtered if pred in gt_filtered]

    n_pred_filtered = len(pred_filtered)
    n_gt_filtered = len(gt_filtered)
    n_correct = len(filtered_correct)

    if n_pred_filtered > 0 or n_gt_filtered > 0:
        precision = n_correct / n_pred_filtered if n_pred_filtered > 0 else 0.0
        recall = n_correct / n_gt_filtered if n_gt_filtered > 0 else 0
        f_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
    else:
        # no prediction and no ground truth for a
        # given type -> correct alignment
        precision, recall, f_score = 1.0, 1.0, 1.0

    return precision, recall, f_score


def sanitize_alignment(alignment: List[dict]) -> None:
    """
    Ensure that note ids are strings in alignments.
    These method changes alignments in-place.

    Parameters
    ----------
    alignment : List[dict]
        List of dictionaries containing an alignment.
    """
    for note in alignment:

        score_id = note.get("score_id", None)

        if score_id is not None:
            note["score_id"] = str(score_id)
        perf_id = note.get("performance_id", None)

        if perf_id is not None:
            note["performance_id"] = str(perf_id)


def dummy_linear_alignment(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    A Dummy linear alignment
    """

    alignment_times_score = np.arange(X.shape[0])
    alignment_times_perf = np.floor(
        np.arange(X.shape[0]) * Y.shape[0] / X.shape[0],
    )

    linear_alignment = np.column_stack(
        [
            alignment_times_score,
            alignment_times_perf,
        ],
    ).astype(int)

    return linear_alignment


def compute_tempo_curve(
    perf: PerformanceLike,
    score: ScoreLike,
    alignment: List[dict],
) -> np.ndarray:
    """
    A naïve calculation of the tempo curve of a performance
    in beats per minute

    Parameters
    ----------
    perf : PerformanceLike
        The performance
    score: ScoreLike
        The score of the performance
    alignment : List[dict]
        A list of dictionaries containing alignment
        information at the note level.

    Returns
    -------
    tempo_info : np.ndarray
        A 2-column array. The first column represents the
        score time in beats, and the second column is the
        corresponding tempo in beats per minute

    """

    snote_array = pt.utils.music.ensure_notearray(score)
    # Score time from the first to the last onset
    score_time = np.linspace(
        snote_array["onset_beat"].min(), snote_array["onset_beat"].max(), 100
    )
    # Include the last offset
    score_time_ending = np.r_[
        score_time,
        (snote_array["onset_beat"] + snote_array["duration_beat"]).max(),  # last offset
    ]
    # Get score time to performance time map
    (
        _,
        stime_to_ptime_map,
    ) = pt.musicanalysis.performance_codec.get_time_maps_from_alignment(
        perf,
        score,
        alignment,
    )
    # Compute naïve tempo curve
    performance_time = stime_to_ptime_map(score_time_ending)
    tempo_curve = 60 * np.diff(score_time_ending) / np.diff(performance_time)

    tempo_info = np.column_stack((score_time, tempo_curve))

    return tempo_info
