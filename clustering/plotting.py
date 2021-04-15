"""Utility library for interactive pyplot plots."""

import kdtree


class PlotAnnotator(object):
  """Helper class for interactive plotting for data with labels."""

  def __init__(self, input_data, max_point_distance=0.1):
    """Initializes the plot with points and their labels.

    Example usage:
      input_data = InputData(
          x=[2015, 2016, 2017]
          y=[10, 50, 100]
          label=["a", "b", "c"])
      fig, ax = plt.subplots()
      ax.scatter(x, y)
      ax.set_xlabel("Year")
      ax.set_ylabel("Value")
      PlotAnnotator(input_data).show(plt, ax)

    Args:
      x: List of x coordinates.
      y: List of y coordinates.
      label: List of labels.
      max_point_distance: max distance of the mouse pointer to the point
    """
    self.x = input_data.x
    self.y = input_data.y
    self.label = input_data.label
    self.max_point_distance = max_point_distance

    self.plot = None
    """matplotlib.pyplot object."""

    self.ax = None
    """matplotlib.AxesSubplot object (subplot)."""

    self._annotation = None
    """Currently displayed annotation."""

    self._tree = None
    """KD-tree: a lookup structure to query a point from the data set
       that is closest to the given coordinates (e.g. mouse coordinates).
       This is useful for displaying a mark when the mouse pointer is close to
       some points on the graph."""

    self._label_map = {}
    """Maps the data point to the corresponding label."""

    self.annotation_style = dict(
        backgroundcolor="black",
        color="white",
        arrowprops=dict(facecolor='black', shrink=0.05))
    """Style for the labels, params are for matplotlib.AxesSubplot.annotate."""

  def output(self, plot, ax, title=None, output_action='show', filename=None):
    """Associates a subplot with an interactive label display and displays a
       plot in new window and / or saves to a file.

    Args:
      plot: matplotlib.pyplot object.
      ax: a subplot object.
    """
    self.plot = plot
    self.ax = ax
    tree, label_map = self._build_tree_and_label_map(self.x, self.y, self.label)
    self._tree = tree
    self._label_map = label_map
    self.plot.connect('motion_notify_event', self._display_annotation)
    if title is not None:
      self.plot.title(title)
    if "show" in output_action:
      self.plot.show()
    if "save" in output_action and filename is not None:
      self.plot.savefig(filename, bbox_inches='tight')

  def _build_tree_and_label_map(self, x, y, label):
    """Builds a KD-tree and a label map."""
    points = []
    label_map = {}
    for i, txt in enumerate(label):
      point = (x[i], y[i])
      points.append(point)
      label_map[point] = "%s (%.2f, %.2f)" % \
                         (label[i], point[0], point[1])
    return kdtree.create(points), label_map

  def _display_annotation(self, event):
    """Shows the label nearest to the mouse pointer."""
    # Remove the previous label.
    if self._annotation is not None:
      self._annotation.remove()
      self._annotation = None
      self.plot.draw()

    if event.xdata is None or event.ydata is None:
      return

    # Find the point closest to the mouse pointer.
    point = (event.xdata, event.ydata)
    node, dist = self._tree.search_nn(point)
    if node is None or dist > self.max_point_distance:
      return

    # Show the label.
    self._annotation = self.ax.annotate(
        self._label_map[node.data],
        node.data,
        **self.annotation_style)
    self.plot.draw()
