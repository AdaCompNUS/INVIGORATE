import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import datetime
import numpy as np

from config.config import *

class paperFig(object):
    def __init__(self, data, size):
        """
        :param data: a list of dict. each element includes "data" and "type"
        :param layout:
        :param size:
        """

        self.data = data
        self.size = size

        self.fig = plt.figure(figsize=self.size, facecolor='white')

    def color_transfer(self, color):
        assert len(color) == 3
        return [float(color[i]) / 256. for i in range(3)]

    def draw_sub_fig(self, ax, data, axis_off = False):
        if data["type"] == "image":
            imgplot = plt.imshow(data["data"])
            ax.set_title(data["title"])
            if axis_off:
                ax.set_xticks([])
                ax.set_yticks([])

    def draw_figure_simple(self, layout, axis_off=False):
        for i, d in enumerate(self.data):
            ax = self.fig.add_subplot(layout[0], layout[1], i + 1)
            self.draw_sub_fig(ax, d, axis_off)

    def draw_figure_with_locs_and_size(self, locs, axis_off=False):
        assert len(locs) == len(self.data)
        for i, d in enumerate(self.data):
            ax = self.fig.add_axes([locs[i][0], locs[i][1], locs[i][2], locs[i][3]])
            self.draw_sub_fig(ax, d, axis_off)

    def draw_arrows(self, arrow_locs, arrow_size = 20):
        for i, loc in enumerate(arrow_locs):
            self.draw_single_arrow(loc[:2], loc[2:], arrow_size)

    def draw_single_arrow(self, tail_coord, head_coord, arrow_size):
        x_head, y_head = head_coord
        x_tail, y_tail = tail_coord
        ax = self.fig.add_axes([0, 0, 1, 1])
        ax.patch.set_facecolor('None')
        arrow = mpatches.FancyArrowPatch((x_tail, y_tail), (x_head, y_head),
                                         mutation_scale=arrow_size, arrowstyle="simple",
                                         edgecolor = None, color = self.color_transfer((66, 66, 66)))
        ax.add_patch(arrow)
        self.hide_axis(ax)

    def draw_single_text(self, loc, content, fontsize = 16, color = (0,0,0),
                         v="center", h="center"):
        ax = self.fig.add_axes([0, 0, 1, 1])
        x, y = loc
        ax.text(x, y, content,
                horizontalalignment=h,
                verticalalignment=v,
                transform=ax.transAxes,
                fontsize = fontsize,
                color = self.color_transfer(color))
        self.hide_axis(ax)

    def draw_rect(self, loc, size, linestyle = '-.'):
        ax = self.fig.add_axes([0, 0, 1, 1])
        ax.patch.set_facecolor('None')
        x_min, y_min = loc
        width, height = size
        p = mpatches.Rectangle(
            (x_min, y_min), width, height,
            fill=False, transform=ax.transAxes, clip_on=False,
            linestyle=linestyle
        )
        ax.add_patch(p)
        self.hide_axis(ax)

    def hide_axis(self, ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

def gen_paper_fig(expr, results):
    # od_data = [
    #     {"data": r["od_img"][:,:,::-1].astype(np.float32) / 256.,
    #      "type": "image",
    #      "title": ""}
    #     for r in results]
    od_data = [
        {"data": r["ground_img"][:,:,::-1].astype(np.float32) / 256.,
         "type": "image",
         "title": ""}
        for r in results]
    action_data = [r["action_str"] for r in results]
    answer_data = [r["answer"] for r in results]
    mrt_data = [
        {"data": r["mrt_img"][:,:,::-1].astype(np.float32) / 256.,
         "type": "image",
         "title": ""}
        for r in results]
    ori_data = [
        {"data": r["origin_img"][:,:,::-1].astype(np.float32) / 256.,
         "type": "image",
         "title": ""}
        for r in results]
    fig_size = (10 * len(results), 20)

    # data = ori_data + od_data + mrt_data
    data = mrt_data + od_data + ori_data
    paper_fig = paperFig(data, size=fig_size)

    sub_fig_width = 9
    interval = 1
    left = 0.5

    # (x, y, w, h)
    interval = float(interval) / fig_size[0]
    left = float(left) / fig_size[0]
    width = float(sub_fig_width) / fig_size[0]
    # ori_locs = [[left + i * (width + interval), 0.02, width, 0.24] for i in range(len(results))]
    # od_locs = [[left + i * (width + interval), 0.3, width, 0.24] for i in range(len(results))]
    # mrt_locs = [[left + i * (width + interval), 0.58, width, 0.24] for i in range(len(results))]
    mrt_locs = [[left + i * (width + interval), 0.1, width, 0.24] for i in range(len(results))]
    od_locs = [[left + i * (width + interval), 0.38, width, 0.24] for i in range(len(results))]
    ori_locs = [[left + i * (width + interval), 0.66, width, 0.24] for i in range(len(results))]
    locs = mrt_locs + od_locs + ori_locs
    paper_fig.draw_figure_with_locs_and_size(locs, axis_off=False)

    # arrow_locs = [
    #     [(2 * loc[0] + loc[2]) / 2.,
    #      loc[1] + loc[3],
    #      (2 * loc[0] + loc[2]) / 2.,
    #      loc[1] + loc[3] + 0.03]
    #     for loc in locs
    # ]

    # draw vertical arrows
    arrow_locs = [
        [
            (2 * loc[0] + loc[2]) / 2., loc[1],        # head position
            (2 * loc[0] + loc[2]) / 2., loc[1] - 0.03  # tail position
        ]
        for loc in locs
    ]

    # draw strings
    for i, action_str in enumerate(action_data):
        text_loc = ((2 * od_locs[i][0] + od_locs[i][2]) / 2., 0.05)
        paper_fig.draw_single_text(text_loc, action_str)
    for i, answer_str in enumerate(answer_data):
        text_loc = ((2 * od_locs[i][0] + od_locs[i][2]) / 2., 0.92)
        paper_fig.draw_single_text(text_loc, "User's answer:" + str(answer_str))
    text_loc = ((0.5, 0.95))
    paper_fig.draw_single_text(text_loc, "User's Command: " + expr)

    for i, pic_loc in enumerate(od_locs):
        rec_loc = (pic_loc[0] - 0.2 * interval, 0.01)
        rec_size = (width + 0.4 * interval, 0.92)
        paper_fig.draw_rect(rec_loc, rec_size)

    # draw horizontal arrows
    arrow_locs += [
        [
            od_locs[i][0] + width + 0.2 * interval, 0.5,
            od_locs[i+1][0] - 0.2 * interval, 0.5,
        ]
        for i in range(len(od_locs[:-1]))
    ]
    paper_fig.draw_arrows(arrow_locs)

    current_date = datetime.datetime.now()
    image_id = "{}-{}-{}-{}".format(current_date.year, current_date.month, current_date.day,
                                    time.strftime("%H:%M:%S"))
    plt.savefig(ROOT_DIR +  "images/output/paper_fig/" + image_id + ".png")

if __name__=="__main__":
    img = cv2.imread("../../images/1.png")
    img = img[:,:,::-1].astype(np.float32) / 256.
    data = [
        {"data": img, "type": "image", "title": ""},
        {"data": img, "type": "image", "title": ""},
        {"data": img, "type": "image", "title": ""},
        {"data": img, "type": "image", "title": ""},
    ]

    fig_size = (4 * len(data), 8)
    sub_fig_width = 2
    interval = 2
    left = 0.5

    test = paperFig(data, fig_size)

    # (x, y, w, h)
    interval = float(interval) / fig_size[0]
    left = float(left) / fig_size[0]
    width = float(sub_fig_width) / fig_size[0]
    locs = [
        [left + i * (width + interval), 0.05, width, 0.35] for i in range(len(data))
    ]
    test.draw_figure_with_locs_and_size(locs, axis_off=True)

    # arrow locs
    arrow_locs = [
        [(2 * loc[0] + loc[2]) / 2., 0.35, (2 * loc[0] + loc[2]) / 2., 0.45] for loc in locs
    ]
    test.draw_arrows(arrow_locs)
    test.draw_single_text(loc = (0.5,0.5), content="This is a test string")

    plt.show()
    plt.close()