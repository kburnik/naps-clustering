"""Main program for analysis of NAPS ands NAPS BE with K-means clustering."""

from config import DATA_NAPS_ALL
from config import DATA_NAPS_BE_ALL
from config import OUT_DIR
from lib import BASIC_EMOTION_NAMES
from lib import BASIC_EMOTIONS
from lib import BASIC_STIMULI
from lib import BASIC_STIMULUS_NAMES
from lib import Config
from lib import InputData
from lib import partition_naps
from lib import plot
from lib import read_naps
from lib import read_naps_be
from runner import Runner
import inspect
import numpy as np
import scipy
import sys


config = Config(
    n_clusters=4,
    n_clusters_range=(2, 9),
    n_iterations=2000,
    n_iterations_range=(200, 2401, 200),
    n_evaluations=10,
    out_dir=OUT_DIR)

input_data = read_naps(
    DATA_NAPS_ALL,
    label_name="Label",
    x_name="Valance",
    y_name="Arousal")

runner = Runner(input_data=input_data, config=config)


def snippet_01_interactive_plot():
  """Displays the interactive data plot."""
  runner.plot(runner.compute_raw_partitions(), output_action='show')


def snippet_02_unstable_partition_indices():
  """Shows that index coloring is unstable."""
  runner.plot_repeated(
      partition_factory=runner.compute_raw_partitions,
      n_plots=10,
      prefix_dir='02-unstable-partition-indices',
      name='02-unstable-partition-indices')


def snippet_03_undecidable_partitioning():
  """Displays the undecidability of partitioning."""
  with config.fork(n_iterations=1000):
    runner.plot_fuzzy(
        prefix_dir='03-undecidable-partitioning',
        name='03-undecidable-partitioning')


def snippet_04_naps_elbow_curve():
  """Displays the elbow method curve for NAPS."""
  with config.fork(n_clusters_range=(2, 51)):
    runner.plot_cluster_number_evaluation_curve(
        evaluate=lambda samples, k: partition_naps(samples, k).score(samples),
        title='NAPS - variation score vs number of partitions (elbow method)',
        score_label='variation score',
        prefix_dir='04-naps-elbow-curve',
        name='04-naps-elbow-curve')


def snippet_05_naps_silhouette_curve():
  """Displays the silhouette curve for NAPS."""
  with config.fork(n_clusters_range=(2, 51)):
    runner.plot_cluster_number_evaluation_curve(
        evaluate=Runner.compute_silhouette_score,
        title='NAPS - silhouette coefficient vs number of partitions',
        score_label='silhouette coefficient',
        prefix_dir='05-naps-silhouette-curve',
        name='05-naps-silhouette-curve')


def snippet_06_naps_results_for_cluster_range():
  """Saves the results for NAPS (k = 2 do 8)."""
  with config.fork(n_clusters_range=(2, 9)):
    runner.compute_naps_results(prefix_dir='06-naps-stable-clustering')


def snippet_07_naps_be_emotion_corellations():
  """Shows the correlations between the 6 basic emotions."""
  prefix_dir = '07-naps-be-emotion-correlations'
  correlations = []
  for runner, x_axis, y_axis in Runner.stream_naps_be(
      config,
      BASIC_EMOTIONS,
      BASIC_EMOTIONS,
      BASIC_EMOTION_NAMES,
      BASIC_EMOTION_NAMES):
    pearson_correlation = np.corrcoef(
        runner.input_data.x, runner.input_data.y)[1, 0]
    spearmanr = scipy.stats.spearmanr(
        runner.input_data.x, runner.input_data.y)
    correlations.append(
        (BASIC_EMOTION_NAMES[BASIC_EMOTIONS.index(x_axis)],
         BASIC_EMOTION_NAMES[BASIC_EMOTIONS.index(y_axis)],
         pearson_correlation,
         spearmanr.correlation,
         ))
    result_name = '07-naps-be-emotion-correlations-%s-%s' % (x_axis, y_axis)
    plot(
        runner.input_data,
        title='NAPS BE - correlation between 2 emotions',
        output_action='save',
        filename=runner.join_path(
            prefix_dir,
            '%s.png' % result_name))
  runner.save_csv(
    data=correlations,
    columns=['X-axis', 'Y-axis', 'pearson_coef', 'spearman_coef'],
    prefix_dir=prefix_dir,
    name='07-naps-be-emotion-correlations')


def snippet_08_naps_be_emotion_stimuli_corellations():
  """Displays the correlations between 6 basic emotions and stimuli."""
  prefix_dir = '08-naps-be-emotion-stimuli-correlations'
  correlations = []
  for runner, x_axis, y_axis in Runner.stream_naps_be(
      config,
      BASIC_EMOTIONS,
      BASIC_STIMULI,
      BASIC_EMOTION_NAMES,
      BASIC_STIMULUS_NAMES):
    pearson_correlation = np.corrcoef(
      runner.input_data.x, runner.input_data.y)[1, 0]
    spearmanr = scipy.stats.spearmanr(
        runner.input_data.x, runner.input_data.y)
    result_name = (
        '08-naps-be-emotion-stimuli-correlations-%s-%s' % (x_axis, y_axis))
    correlations.append(
        (BASIC_EMOTION_NAMES[BASIC_EMOTIONS.index(x_axis)],
         BASIC_STIMULUS_NAMES[BASIC_STIMULI.index(y_axis)],
         pearson_correlation,
         spearmanr.correlation,
         ))
    plot(
        runner.input_data,
        title='NAPS BE - correlation between emotion and stimuli',
        output_action='save',
        filename=runner.join_path(
            prefix_dir,
            '%s.png' % result_name))
  runner.save_csv(
    data=correlations,
    columns=['X-axis', 'Y-axis', 'pearson_coef', 'spearman_coef'],
    prefix_dir=prefix_dir,
    name='08-naps-be-emotion-stimuli-correlations')


def snippet_09_naps_be_silhouette_curve_for_emotion_pairs():
  """Plots the chosen emo-emo correlations - silhouette method."""
  pairs = [
    ('happiness', 'sadness'),
    ('happiness', 'anger'),
    ('sadness', 'anger'),
    ('fear', 'surprise')
  ]
  input_data_list = []
  for emotion_pair in pairs:
    x_axis, y_axis = emotion_pair
    x_name = BASIC_EMOTION_NAMES[BASIC_EMOTIONS.index(x_axis)]
    y_name = BASIC_EMOTION_NAMES[BASIC_EMOTIONS.index(y_axis)]
    input_data_list.append(read_naps_be(
        DATA_NAPS_BE_ALL,
        label_field="label",
        x_axis=x_axis,
        y_axis=y_axis,
        label_name="%s - %s" % (x_name, y_name)))
  runner.plot_multiple_cluster_number_evaluation_curves(
      input_data_list,
      n_clusters_range=(2, 20),
      evaluate=Runner.compute_silhouette_score,
      title='NAPS - silhouette coefficient vs number of partitions',
      prefix_dir='09-naps-be-silhouette-curves-for-emotion-pairs',
      name='09-naps-be-silhouette-curves-for-emotion-pairs',
      score_label='silhouette coefficient')


def snippet_10_naps_be_silhouette_curves_for_emotion_stimuli_pairs():
  """Plots the chosen emo-stimuli correlations - silhouette method."""
  pairs = [
    ('happiness', 'valance'),
    ('sadness', 'valance'),
    ('fear', 'arousal'),
    ('surprise', 'arousal'),
    ('anger', 'valance')
  ]
  input_data_list = []
  for emotion_stimulus in pairs:
    x_axis, y_axis = emotion_stimulus
    x_name = BASIC_EMOTION_NAMES[BASIC_EMOTIONS.index(x_axis)]
    y_name = BASIC_STIMULUS_NAMES[BASIC_STIMULI.index(y_axis)]
    input_data_list.append(read_naps_be(
        DATA_NAPS_BE_ALL,
        label_field="label",
        x_axis=x_axis,
        y_axis=y_axis,
        label_name="%s - %s" % (x_name, y_name)))
  runner.plot_multiple_cluster_number_evaluation_curves(
      input_data_list,
      n_clusters_range=(2, 20),
      evaluate=Runner.compute_silhouette_score,
      title='NAPS - silhouette coefficient vs number of partitions',
      prefix_dir='10-naps-be-silhouette-curves-for-emotion-stimuli-pairs',
      name='10-naps-be-silhouette-curves-for-emotion-stimuli-pairs',
      score_label='silhouette coefficient')


def snippet_11_naps_be_stable_clustering_for_emotion_pairs():
  """Saves the results for NAPS BE emo-emo (k = 4)."""
  config = Config(
    n_clusters=4,
    n_clusters_range=None,
    n_iterations=2000,
    out_dir=OUT_DIR)
  pairs = [
    ('happiness', 'sadness'),
    ('fear', 'surprise')
  ]
  for emotion_pair in pairs:
    x_axis, y_axis = emotion_pair
    x_name = BASIC_EMOTION_NAMES[BASIC_EMOTIONS.index(x_axis)]
    y_name = BASIC_EMOTION_NAMES[BASIC_EMOTIONS.index(y_axis)]
    input_data = read_naps_be(
        DATA_NAPS_BE_ALL,
        label_field="label",
        x_axis=x_axis,
        y_axis=y_axis,
        x_name=x_name,
        y_name=y_name,
        label_name="Description")
    runner = Runner(input_data=input_data, config=config)
    runner.compute_naps_be_results(
        x_axis=x_axis,
        y_axis=y_axis,
        prefix_dir='11-naps-be-stable-clustering-for-emotion-pairs')


def snippet_12_naps_be_stable_clustering_for_emotion_stimuli_pairs():
  """Saves the results for NAPS BE emo-stim (k = 4)."""
  config = Config(
    n_clusters=4,
    n_clusters_range=None,
    n_iterations=2000,
    out_dir=OUT_DIR)
  pairs = [
    ('happiness', 'valance'),
    ('surprise', 'arousal')
  ]
  for emotion_pair in pairs:
    x_axis, y_axis = emotion_pair
    x_name = BASIC_EMOTION_NAMES[BASIC_EMOTIONS.index(x_axis)]
    y_name = BASIC_STIMULUS_NAMES[BASIC_STIMULI.index(y_axis)]
    input_data = read_naps_be(
        DATA_NAPS_BE_ALL,
        label_field="label",
        x_axis=x_axis,
        y_axis=y_axis,
        x_name=x_name,
        y_name=y_name,
        label_name="Description")
    runner = Runner(input_data=input_data, config=config)
    runner.compute_naps_be_results(
        x_axis=x_axis,
        y_axis=y_axis,
        prefix_dir='12-naps-be-stable-clustering-for-emotion-stimuli-pairs')


def snippet_13_stability_error_as_function_of_iteration_count():
  """Computes the stability error as a function of number of iterations."""
  prefix_dir = '13-stability-error-for-iteration-count'
  result_name = '13-stability-error-as-function-of-iteration-count'
  with config.fork(n_iterations_range=(10, 200, 20)):
    low_range_error_points = runner.cached(
        func=runner.compute_stability_error_of_iterations,
        prefix_dir=prefix_dir,
        name='%s-low-range' % result_name)

  with config.fork(n_iterations_range=(200, 2400, 200)):
    high_range_error_points = runner.cached(
        func=runner.compute_stability_error_of_iterations,
        prefix_dir=prefix_dir,
        name='%s-high-range' % result_name)

  error_points = low_range_error_points + high_range_error_points
  runner.save_csv(
        data=error_points,
        columns=['Iterations (p)', 'Error (e)'],
        prefix_dir=prefix_dir,
        name=result_name)

  interpolated_high_range_points = []
  for x, y in high_range_error_points:
    start = x + 10
    for i in range(start, start + 170, 20):
      interpolated_high_range_points.append((i, y))

  plot_error_points = low_range_error_points + interpolated_high_range_points
  runner.plot_stability_error_curve(
      plot_error_points,
      xlabel='iterations',
      ylabel='error',
      title='Total stability error as a function of number of iterations',
      prefix_dir=prefix_dir,
      name=result_name,
      xticks=200,
      yticks=5,
      figsize=(16, 6),
      dpi=300)


def snippet_14_stability_error_as_function_of_cluster_count():
  """Computes the stability error as a function of the number of clusters."""
  prefix_dir = '14-stability-error-for-cluster-count'
  result_name = '14-stability-error-as-function-of-cluster-count'
  with config.fork(
      n_iterations=100,
      n_evaluations=10,
      n_clusters_range=(2, 31)):
    error_points = runner.cached(
        func=runner.compute_stability_error_of_partition_count,
        prefix_dir=prefix_dir,
        name=result_name)
    runner.save_csv(
        data=error_points,
        columns=['Cluster count (k)', 'Error (e)'],
        prefix_dir=prefix_dir,
        name=result_name)
    runner.plot_stability_error_curve(
        error_points,
        xlabel='cluster count',
        ylabel='error',
        title='Stability error as a function of the number of clusters',
        prefix_dir=prefix_dir,
        name=result_name,
        xticks=5,
        yticks=100,
        figsize=(16, 6),
        dpi=300)

def display_snippet_info(index, member):
  """Displays the basic info for the snippet."""
  name, func = member
  print("%02d" % index, name)
  print("  ", func.__doc__)
  print("")

if __name__ == '__main__':
  sys.stdout.flush()
  snippets = list(filter(
        lambda p: inspect.isfunction(p[1]) and p[0].startswith('snippet_'),
        inspect.getmembers(sys.modules[__name__])))

  if len(sys.argv) < 2:
    for i, member in enumerate(snippets):
      display_snippet_info(i + 1, member)
    print("Run as: python -u", sys.argv[0], '<snippet_number>')
  else:
    arg = int(sys.argv[1])
    snippet = snippets[arg - 1]
    display_snippet_info(arg, snippet)
    _, func = snippet
    func()
