"""Library with basic functions for data input and plotting."""

from plotting import PlotAnnotator
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import csv
from contextlib import contextmanager


BASIC_EMOTIONS = [
    'happiness', 'sadness', 'fear', 'surprise', 'anger', 'disgust']
BASIC_EMOTION_NAMES = BASIC_EMOTIONS  # can be replaced for translations.
"""Six basic emotions."""

BASIC_STIMULI = ['valance', 'arousal']
BASIC_STIMULUS_NAMES = BASIC_STIMULI  # can be replaced for translations.
"""List of stimuli dimensions"""


class Config:
  """Encapsulates the configuration for running the clustering algorithm."""
  def __init__(
      self,
      n_clusters=4,
      n_clusters_range=(2, 11),
      n_iterations=100,
      n_iterations_range=(1, 3000, 100),
      n_evaluations=100,
      out_dir='.'):
    self.n_clusters = n_clusters
    self.n_clusters_range = n_clusters_range
    self.n_iterations = n_iterations
    self.n_iterations_range = n_iterations_range
    self.n_evaluations = n_evaluations
    self.out_dir = out_dir

  @contextmanager
  def fork(self, **kwargs):
    """Makes a copy with applied changes ('with' context)."""
    original = self.__dict__.copy()
    self.__dict__.update(kwargs)
    try:
      yield self
    finally:
      self.__dict__.update(original)


class InputData:
  """Input data structure (dataset)."""
  def __init__(self, label, x, y, label_name=None, x_name=None, y_name=None):
    self.label = np.array(label)
    self.x = np.array(x)
    self.y = np.array(y)
    self.label_name = label_name
    self.x_name = x_name
    self.y_name = y_name
    self._samples = np.column_stack((self.x, self.y))

  @property
  def size(self):
    return self._samples.shape[0]

  @property
  def samples(self):
    """Returns the combined X, Y values in the shape of [N, 2]."""
    return self._samples

  def split_on_filter(self, filt):
    """Splits the data into two new instances as defined by the lambda filter."""
    result = (self._init_interim(), self._init_interim())
    indices = [[], []]
    for i, row in enumerate(self._samples):
      partition = int(not filt(i, row))
      result[partition]["label"].append(self.label[i])
      result[partition]["x"].append(self.x[i])
      result[partition]["y"].append(self.y[i])
      indices[partition].append(i)
    return (
        InputData(**result[0]),
        InputData(**result[1]),
        np.array(indices[0]),
        np.array(indices[1]))

  def split_on_key(self, key_func):
    """Splits the data into two new instances as defined by the key_func."""
    result = {}
    for i, row in enumerate(self._samples):
      key = key_func(i, row)
      if not key in result:
        result[key] = self._init_interim()
      result[key]["label"].append(self.label[i])
      result[key]["x"].append(self.x[i])
      result[key]["y"].append(self.y[i])
    for key, interim in result.items():
      result[key] = InputData(**interim)
    return result

  def serialize(self, delimiter=';', use_quotes=True, round_decimals=True):
    """Returns the data set serialized as CSV or TSV."""
    def quote(str):
      if use_quotes:
        return '"%s"' % str
      else:
        return str

    def serialize_float(value):
      if round_decimals:
        return "%.2f" % value
      else:
        return str(value)

    out = delimiter.join([
      quote(self.label_name),
      quote(self.x_name),
      quote(self.y_name)
    ]) + "\n"
    for i in range(self.size):
      out += delimiter.join([
        quote(self.label[i]),
        serialize_float(self.x[i]),
        serialize_float(self.y[i])
      ]) + "\n"
    return out

  def reduce_to_samples(self, num_samples):
    """Returns samples from the data that are closest to their centroid."""
    centroid = (np.sum(self.x) / self.size, np.sum(self.y) / self.size)
    centroid_vector = np.column_stack(
        (np.repeat(centroid[0], self.size),
         np.repeat(centroid[1], self.size)))
    distance = np.linalg.norm(
        self.samples - centroid_vector, keepdims=True, axis=1)
    data = np.column_stack((
        self.label.astype(np.object),
        self.x,
        self.y,
        distance))
    data = data[data[:,3].argsort()]
    data = data[0:num_samples]
    result = self._init_interim()
    for i, var in enumerate(["label", "x", "y"]):
      result[var] = list(data[:, i])
    return InputData(**result)

  def _init_interim(self):
    """
    Returns the interim structure to fill in when creating a new data set.
    """
    return {
      "label": [],
      "x": [],
      "y": [],
      "label_name": self.label_name,
      "x_name": self.x_name,
      "y_name": self.y_name
    }


def read_input_data(
    filename,
    label_field="label",
    x_axis="valance",
    y_axis="arousal",
    label_name="label",
    x_name="valance",
    y_name="arousal",
    fields=["label", "valance", "arousal"],
    delimiter=",",
    quotechar='"'):
  """Loads and deserializes the data set into the provided columns."""
  label, x, y = [], [], []
  with open(filename, "r") as f:
    reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
    for row in reader:
      row_data = {}
      for i, field in enumerate(fields):
        row_data[field] = row[i]
      label.append(row_data[label_field])
      x.append(float(row_data[x_axis]))
      y.append(float(row_data[y_axis]))
  return InputData(
      label=label,
      x=x,
      y=y,
      label_name=label_name,
      x_name=x_name,
      y_name=y_name)


def read_naps(filename, **kwargs):
  """Returns a loaded NAPS dataset as an InputData."""
  return read_input_data(
      filename,
      delimiter=",",
      quotechar='"',
      label_field="label",
      x_axis="valance",
      y_axis="arousal",
      fields=["label", "valance", "arousal"],
      **kwargs)


def read_naps_be(filename, **kwargs):
  """Returns a loaded NAPS BE dataset as an InputData."""
  return read_input_data(
      filename,
      delimiter=';',
      fields=['image_name', 'label'] + BASIC_EMOTIONS + ['arousal', 'valance'],
      **kwargs)


def partition_naps(samples, n_clusters):
  """Computes the K-means partitions on samples of shape [N,2]."""
  return KMeans(n_clusters=n_clusters, init='random').fit(samples)


def find_dominant(sample):
  """Finds the first max value in the sample (i.e. argmax).
     Returns the position and the value."""
  dominant_index = np.argmax(sample)
  return dominant_index, sample[dominant_index]


def plot_setup(x_label="X", y_label="Y"):
  """Creates a new subplot with the provided setup."""
  fig, ax = plt.subplots()
  fig.set_size_inches(w=16, h=9)
  fig.set_dpi(80)
  ax.set_xlabel(x_label)
  ax.set_ylabel(y_label)
  return fig, ax


def reindex_partitions(samples, indices):
  """Reindexes partitions based on the centroid positions.
     (lexicographical sorting of points for stable coloring)."""
  count = len(indices)
  partitions = {}
  for i in range(count):
    cluster = indices[i]
    if not cluster in partitions:
      partitions[cluster] = {
        'x': 0.0,
        'y': 0.0,
        'count': 0.0,
        'center': None,
        'cluster': cluster
      }
    partitions[cluster]['x'] += samples[i, 0]
    partitions[cluster]['y'] += samples[i, 1]
    partitions[cluster]['count'] += 1.0

  ordering = [None] * len(partitions.keys())
  for cluster, partition in partitions.items():
    partition['center'] = (
        partition['x'] / partition['count'],
        partition['y'] / partition['count'])
    ordering[cluster] = partition

  ordering = list(sorted(ordering, key=lambda p: p['center']))
  new_ordering = [None] * len(partitions.keys())
  for i, partition in enumerate(ordering):
    new_ordering[partition['cluster']] = i
  return map(lambda c: new_ordering[c], indices)


def plot(input_data, title='', output_action='show', filename=None):
  """Plots the data set with its labels."""
  fig, ax = plot_setup(x_label=input_data.x_name, y_label=input_data.y_name)
  ax.scatter(input_data.x, input_data.y)
  PlotAnnotator(input_data).output(plt, ax, title, output_action, filename)


def partition_for_plotting(indices, input_data, n_clusters):
  """Partitions the data set for plotting."""
  partitions = {i: {'x': [], 'y':[], 'label': None} \
                for i in range(n_clusters)}
  for i, partition_index in enumerate(indices):
    partitions[partition_index]['x'].append(input_data.x[i])
    partitions[partition_index]['y'].append(input_data.y[i])
    partitions[partition_index]['label'] = input_data.label[i]
  return partitions


def plot_clusters(indices, input_data, n_clusters, cluster_names=None,
                  title=None, output_action='show', filename=None,
                  block=True):
  """PLots the clusters with different colors and labels them."""
  if cluster_names is None:
    cluster_names = ["P" + str(i) for i in range(n_clusters)]

  fig, ax = plot_setup(x_label=input_data.x_name, y_label=input_data.y_name)
  color = plt.cm.rainbow(np.linspace(0, 1, n_clusters))

  partitions = partition_for_plotting(indices, input_data, n_clusters)

  for partition_index, partition in partitions.items():
    ax.scatter(
        partition['x'], partition['y'],
        c=color[partition_index],
        label=cluster_names[partition_index])
  if not block:
    plt.ion()
  plt.legend()
  PlotAnnotator(input_data).output(plt, ax, title, output_action, filename)


def mix_colors(colors, indices_prob, post_alpha=0.5):
  """Weighted color mixing: the color ratio depending on indices_prob."""
  # C - n_clusters
  # N - n_samples
  # indices_prob: (N, C) x colors: (C, 4) --> (N, 4) . (1, 4)
  return np.multiply(
      np.matmul(indices_prob, colors),
      np.array([1.0, 1.0, 1.0, post_alpha]))


def plot_clusters_with_probability(
    indices_prob, input_data, cluster_names=None,
    title=None, output_action='show', filename=None,
    block=True, plot_fuzzy_simple=False):
  """
  Plots the clusters with different colors and assigns the labels, for each
  point the degree of membership is defined in an array of size n_clusters.
  """
  n_clusters = indices_prob.shape[1]
  if cluster_names is None:
    cluster_names = ["P" + str(i) for i in range(n_clusters)]

  fig, ax = plot_setup(x_label=input_data.x_name, y_label=input_data.y_name)
  color = plt.cm.rainbow(np.linspace(0, 1, n_clusters))

  # Split the data into to subsets: points that 100% of the time fall into their
  # own cluster, and the ones that don't.
  indices = np.argmax(indices_prob, axis=1)
  maxes = np.max(indices_prob, axis=1)
  exact_data, fuzzy_data, match_idx, non_match_idx = (
      input_data.split_on_filter(lambda i, row: maxes[i] == 1.0))
  exact_indices = np.take(indices, match_idx)
  fuzzy_indices_prob = np.take(indices_prob, non_match_idx, axis=0)
  fuzzy_color = mix_colors(color, fuzzy_indices_prob)

  # Plot points that are 100% of the time in their own cluster.
  partitions = partition_for_plotting(exact_indices, exact_data, n_clusters)
  for partition_index, partition in partitions.items():
    ax.scatter(
        partition['x'], partition['y'],
        c=color[partition_index],
        label=cluster_names[partition_index])

  # Plot the fuzzy points - the edge points between the clusters.
  for i in range(fuzzy_data.size):
    ax.scatter(
        [fuzzy_data.x[i]],
        [fuzzy_data.y[i]],
        color=fuzzy_color[i] if not plot_fuzzy_simple else [0,0,0,1])

  if not block:
    plt.ion()
  plt.legend()
  PlotAnnotator(input_data).output(plt, ax, title, output_action, filename)
