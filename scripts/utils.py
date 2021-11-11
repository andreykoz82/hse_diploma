from matplotlib import patches
import matplotlib.pyplot as plt


def get_rectangle_edges_from_pascal_bbox(bbox):
    xmin_top_left, ymin_top_left, xmax_bottom_right, ymax_bottom_right = bbox

    bottom_left = (xmin_top_left, ymax_bottom_right)
    width = xmax_bottom_right - xmin_top_left
    height = ymin_top_left - ymax_bottom_right

    return bottom_left, width, height


def draw_pascal_voc_bboxes(
        plot_ax,
        bboxes,
        get_rectangle_corners_fn=get_rectangle_edges_from_pascal_bbox,
):
    for bbox in bboxes:
        bottom_left, width, height = get_rectangle_corners_fn(bbox)

        rect_1 = patches.Rectangle(
            bottom_left,
            width,
            height,
            linewidth=1,
            edgecolor="green",
            fill=False,
        )

        # Add the patch to the Axes
        plot_ax.add_patch(rect_1)


def draw_labels(plot_ax, bboxes, class_labels):
    for box, label in zip(bboxes, class_labels):
        xmin_top_left, ymin_top_left, xmax_bottom_right, ymax_bottom_right = box
        plot_ax.text(xmin_top_left + 6, ymin_top_left - 5, label, color='red',
                     fontsize=8, weight='bold')


def show_image(
        image, bboxes=None, draw_bboxes_fn=draw_pascal_voc_bboxes,
        draw_labels_fn=draw_labels, class_labels=None, figsize=(4, 3)
):
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)

    if bboxes is not None:
        draw_bboxes_fn(ax, bboxes)
        draw_labels_fn(ax, bboxes, class_labels)
    plt.show()


def visualize(
        image,
        predicted_bboxes,
        predicted_class_labels,
        draw_bboxes_fn=draw_pascal_voc_bboxes,
        draw_labels_fn=draw_labels,
        figsize=(4, 3)
):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(image)
    ax.set_title("Prediction")

    draw_bboxes_fn(ax, predicted_bboxes)
    draw_labels_fn(ax, predicted_bboxes, predicted_class_labels)

    plt.show()
