"""Class with high-level methods for processing NAPS and NAPS BE datasets."""

from config import DATA_NAPS_BE_ALL
from lib import partition_naps
from lib import plot
from lib import plot_clusters
from lib import plot_clusters_with_probability
from lib import plot_setup
from lib import read_naps
from lib import read_naps_be
from lib import reindex_partitions
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import sklearn


class Runner:
  """Provides methods for processing NAPS with the clustering algorithm."""
  def __init__(self, input_data, config):
    self.input_data = input_data
    self.config = config

  def compute_raw_partitions(self):
    """Compute the k-means and returns the cluster index for each sample."""
    kmeans = partition_naps(
        samples=self.input_data.samples,
        n_clusters=self.config.n_clusters)
    return kmeans.labels_

  def compute_stable_partitions(self):
    """Same as compute_raw_partition, but with stable index coloring."""
    return reindex_partitions(
        samples=self.input_data.samples,
        indices=self.compute_raw_partitions())

  def compute_average_partitions(self):
    """
    Repeats the stable colored k-means and computes the average membership
    of each input sample. For each sample, return the percentage of membership
    to a cluster, as an array of size n_clusters (Monte-Carlo simulation).
    """
    cluster_hist = np.zeros(
        (self.input_data.size, self.config.n_clusters))
    for k in range(self.config.n_iterations):
      indices = self.compute_stable_partitions()
      for i, cluster in enumerate(indices):
        cluster_hist[i][cluster] += 1
    return np.divide(cluster_hist, self.config.n_iterations)

  def compute_stable_argmax_partitions(self):
    """Computes the stable partitions using the Monte-Carlo simulation, and
    selects the most frequent cluster based on the probability (argmax)."""
    indices_prob = self.compute_average_partitions()
    self._display_undecided_index_count(indices_prob)
    return np.argmax(indices_prob, axis=1)

  def compute_naps_results(self, num_samples=5, prefix_dir='naps-clustering'):
    """Saves the clustering results and plots the NAPS clusters."""
    with self.config.fork() as config:
      p = config.n_iterations
      for k in range(*config.n_clusters_range):
        config.n_clusters = k

        # Partition with caching.
        indices = np.array(self.cached(
            func=self.compute_stable_argmax_partitions,
            prefix_dir=prefix_dir,
            name='naps-clustering-k=%d-p=%d' % (k, p)))

        # Split the input data.
        partitioned_data = self.partition_input_data(indices)

        # Save the separated datasets.
        partitions_filename = self.join_path(
            prefix_dir,
            'naps-clustering-partitioned-full-k=%d-p=%d.csv' % (k, p))
        with open(partitions_filename, "w") as f:
          for cluster, data in partitioned_data.items():
            f.write(data.serialize() + "\n")

        # Save the chosen samples.
        samples_filename = self.join_path(
            prefix_dir,
            'naps-clustering-partitioned-samples-k=%d-p=%d.csv' % (k, p))
        with open(samples_filename, "w") as f:
          for cluster, data in partitioned_data.items():
            chunk = data.reduce_to_samples(num_samples)
            f.write(chunk.serialize(";", use_quotes=False) + "\n")

        self.plot(
          indices=indices,
          filename=self.join_path(
              prefix_dir,
              'naps-clustering-k=%d-p=%d.png' % (k, p)),
          output_action='save')

  def compute_naps_be_results(
      self,
      x_axis,
      y_axis,
      num_samples=5,
      prefix_dir='naps-be-clustering'):
    """Saves the clustering results and plots the NAPS BE clusters."""
    p = self.config.n_iterations
    k = self.config.n_clusters

    # Partition with caching.
    indices = np.array(self.cached(
        func=self.compute_stable_argmax_partitions,
        prefix_dir=prefix_dir,
        name='naps-be-clustering-%s-%s-k=%d-p=%d' % (x_axis, y_axis, k, p)))

    # Split the input data.
    partitioned_data = self.partition_input_data(indices)

    # Save the separated datasets.
    partitions_filename = self.join_path(
        prefix_dir,
        'naps-be-clustering-partitioned-full-%s-%s-k=%d-p=%d.csv' % (
            x_axis, y_axis, k, p))
    with open(partitions_filename, "w") as f:
      for cluster, data in partitioned_data.items():
        f.write(data.serialize() + "\n")

    # Save the chosen samples.
    samples_filename = self.join_path(
        prefix_dir,
        'naps-be-clustering-partitioned-samples-%s-%s-k=%d-p=%d.csv' % (
            x_axis, y_axis, k, p))
    with open(samples_filename, "w") as f:
      for cluster, data in partitioned_data.items():
        chunk = data.reduce_to_samples(num_samples)
        f.write(chunk.serialize(";", use_quotes=False) + "\n")

    self.plot(
      indices=indices,
      filename=self.join_path(
          prefix_dir,
          'naps-be-clustering-%s-%s-k=%d-p=%d.png' % (x_axis, y_axis, k, p)),
      output_action='save')

  def compute_stability_error_of_iterations(self):
    """Computes the stability error curve as a function of number of
    iterations."""
    with self.config.fork() as config:
      return [
        self._compute_stability_error_point(config.n_iterations)
        for config.n_iterations in
        range(*config.n_iterations_range)
      ]

  def compute_stability_error_of_partition_count(self):
    """Computes the stability error curve as a function of number of
    clusters."""
    with self.config.fork() as config:
      return [
        self._compute_stability_error_point(config.n_clusters)
        for config.n_clusters in
        range(*config.n_clusters_range)
      ]

  def partition_input_data(self, indices):
    """Splits the input data to partitions as defined by the indices."""
    return self.input_data.split_on_key(lambda i, row: indices[i])

  def plot(self, indices, output_action='save', filename=None):
    """Plots the clusters."""
    if filename is None:
      # TODO: Add date?
      filename = self.join_path('out-single-run.png')
    plot_clusters(
        indices=indices,
        input_data=self.input_data,
        n_clusters=self.config.n_clusters,
        output_action=output_action,
        filename=filename)

  def plot_repeated(
      self,
      partition_factory,
      n_plots=10,
      name='out',
      prefix_dir='.'):
    """
    Runs the partition_factory requested number of times, plots and saves the
    images.
    """
    for i in range(n_plots):
      self.plot(
          indices=partition_factory(),
          output_action='save',
          filename=self.join_path(prefix_dir, '%s-%02d.png' % (name, i)))

  def plot_fuzzy(self, prefix_dir='.', name='out-fuzzy-simple'):
    """Plots the undecidable points."""
    indices_prob = np.array(self.cached(
        func=self.compute_average_partitions,
        name=name,
        prefix_dir=prefix_dir))
    plot_clusters_with_probability(
        indices_prob=indices_prob,
        input_data=self.input_data,
        plot_fuzzy_simple=True,
        output_action='save',
        filename=self.join_path(prefix_dir, '%s.png' % name))

  def plot_cluster_number_evaluation_curve(
      self,
      evaluate,
      title,
      name,
      score_label,
      prefix_dir='.'):
    """Plots the evaluation curve as a function of number of clusters K."""
    samples = self.input_data.samples
    k_range = range(*self.config.n_clusters_range)
    score = [evaluate(samples, k) for k in k_range]
    self.save_csv(
        data=zip(k_range, score),
        columns=['partition count', score_label],
        prefix_dir=prefix_dir,
        name=name)
    plt.figure(num=None, figsize=(16, 9), dpi=300)
    plt.title(title)
    plt.xlabel('partition count')
    plt.ylabel(score_label)
    plt.xticks(np.arange(*self.config.n_clusters_range, 2.0))
    plt.plot(k_range, score)
    plt.grid()
    plt.savefig(self.join_path(prefix_dir, '%s.png' % name))
    return plt

  def plot_stability_error_curve(
      self,
      results,
      title,
      name,
      xlabel,
      ylabel,
      xticks=200,
      yticks=5,
      figsize=(16, 6),
      dpi=300,
      prefix_dir='.'):
    plt.figure(num=None, figsize=figsize, dpi=dpi)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(np.arange(0, 1 + max([x for x, y in results]), xticks))
    plt.yticks(np.arange(0, 1 + max([y for x, y in results]), yticks))
    plt.plot(*zip(*results))
    plt.grid()
    plt.savefig(
        self.join_path(prefix_dir, '%s.png' % name),
        bbox_inches='tight')
    return plt

  def plot_multiple_cluster_number_evaluation_curves(
      self,
      input_data_list,
      evaluate,
      n_clusters_range,
      title,
      name,
      score_label,
      prefix_dir='.'):
    """Plots the evaluation curve for a given range of K."""
    fig, ax = plot_setup()
    plt.title(title)
    plt.xlabel('partition count')
    plt.ylabel(score_label)
    plt.xticks(np.arange(*n_clusters_range, 2.0))
    color = plt.cm.rainbow(np.linspace(0, 1, len(input_data_list)))
    k_range = range(*n_clusters_range)
    score_vectors = []
    for i, input_data in enumerate(input_data_list):
      score = [evaluate(input_data.samples, k) for k in k_range]
      ax.plot(k_range, score, color=color[i], label=input_data.label_name)
      score_vectors.append(score)
    score_average = np.average(score_vectors, axis=0)
    ax.plot(k_range, score_average, color=(0, 0, 0, 1), label="Average")
    plt.grid()
    plt.legend()
    plt.savefig(self.join_path(prefix_dir, '%s.png' % name))

  def _compute_stability_error_point(self, variable):
    """Computes one error point though the given number of evaluation
    simulations."""
    cluster_hist = np.zeros(
        (self.input_data.size, self.config.n_clusters))
    for i in range(self.config.n_evaluations):
      indices = self.compute_stable_argmax_partitions()
      for j, cluster in enumerate(indices):
        cluster_hist[j][cluster] += 1
    total_error = self._compute_total_histogram_error(
        cluster_hist, self.config.n_evaluations)
    error_point = (variable, total_error)
    print(error_point)
    return error_point

  def cached(self, func, name, prefix_dir='.'):
    """Runs the provided method using a caching mechanism."""
    filename = self.join_path(prefix_dir, '%s.cached-result.json' % name)
    if os.path.exists(filename):
      with open(filename, 'r') as f:
        results = json.load(f)
    else:
      results = func()
      with open(filename, 'w') as f:
        try:
          results = results.tolist()
        except:
          pass
        json.dump(results, f)
    return results

  def save_csv(
      self,
      data,
      columns,
      name,
      delimiter=';',
      prefix_dir='.',
      extension='.csv'):
    """Saves data into a CSV file."""
    filename = self.join_path(prefix_dir, name + extension);
    def encode(item):
      return str(item)
    with open(filename, 'w') as f:
      f.write(delimiter.join(['"%s"' % column for column in columns]) + '\n')
      for row in data:
        f.write(delimiter.join([encode(item) for item in row]) + '\n')

  def join_path(self, *args):
    """Joins a path for an output file and creates directories if they don't
    exist."""
    filename = os.path.join(self.config.out_dir, *args)
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
      os.makedirs(dirname, 0o755)
    print("I/O path:", os.path.abspath(filename))
    return filename

  def _compute_total_histogram_error(self, hist, n_evaluations):
    """Computes the total error from the histogram of point cluster
    membership."""
    hist[hist == n_evaluations] = 0
    sums_per_row = (hist != 0).sum(1)
    return sums_per_row.sum() - np.count_nonzero(sums_per_row)

  def _display_undecided_index_count(self, indices_prob):
    """Counts and prints out how many points have appeared at the edges of
    clusters (the undecidability region)."""
    print("Undecided count:", len(list(filter(
        lambda row: np.max(row) == 0.5, indices_prob))))

  @staticmethod
  def compute_silhouette_score(samples, n_clusters):
    """Computes the silhouette score for a provided clustering result."""
    kmeans = partition_naps(samples, n_clusters)
    return sklearn.metrics.silhouette_score(
        samples,
        kmeans.labels_,
        metric='euclidean')

  @staticmethod
  def stream_naps_be(
      config,
      x_dimensions, y_dimensions,
      x_dimension_names, y_dimension_names):
    """Generates datasets for chosen pairs of dimensions."""
    for i in range(len(x_dimensions)):
      for j in range(len(y_dimensions)):
        if len(x_dimensions) == len(y_dimensions) and j <= i:
          continue
        x_axis, y_axis = x_dimensions[i], y_dimensions[j]
        x_name, y_name = x_dimension_names[i], y_dimension_names[j]
        input_data = read_naps_be(
            DATA_NAPS_BE_ALL,
            label_field="label",
            x_axis=x_axis,
            y_axis=y_axis,
            label_name="Label",
            x_name=x_name,
            y_name=y_name)
        yield Runner(input_data=input_data, config=config), x_axis, y_axis
