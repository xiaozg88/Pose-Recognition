
from matplotlib import pyplot as plt
import os
import io
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import requests

# 可视化


class PoseClassificationVisualizer(object):
    """Keeps track of classifcations for every frame and renders them.
    跟踪每一帧的分类并呈现它们"""

    def __init__(self,
                 class_name,   # 要可视化的姿态类别名称
                 plot_location_x=0.05,
                 plot_location_y=0.05,  # 运动变化图在整个屏幕中的起始点位置
                 plot_max_width=0.8,
                 plot_max_height=0.8,   # 修改运动变化图所在图表的宽和高
                 # plot_max_width=0.4,
                 # plot_max_height=0.4,
                 plot_figsize=(12, 8),    # 分类图例的尺寸
                 #plot_figsize=(9, 4),
                 plot_x_max=None,
                 plot_y_max=None,  # x,y轴的最大值
                 counter_location_x=0.85,
                 counter_location_y=0.05,  # 计数器的位置
                 # 计数器字体路径
                 counter_font_path='https://github.com/googlefonts/roboto/blob/main/src/hinted/Roboto-Regular.ttf?raw=true',
                 counter_font_color='red',
                 counter_font_size=0.15):
        self._class_name = class_name
        self._plot_location_x = plot_location_x
        self._plot_location_y = plot_location_y
        self._plot_max_width = plot_max_width
        self._plot_max_height = plot_max_height
        self._plot_figsize = plot_figsize
        self._plot_x_max = plot_x_max
        self._plot_y_max = plot_y_max
        self._counter_location_x = counter_location_x
        self._counter_location_y = counter_location_y
        self._counter_font_path = counter_font_path
        self._counter_font_color = counter_font_color
        self._counter_font_size = counter_font_size

        # self._counter_font = None
        self._counter_font = 'Roboto-Regular.ttf'

        self._pose_classification_history = []  # 保存分类的历史结果
        self._pose_classification_filtered_history = []  # 保存经过滤波之后的历史结果

    def __call__(self,
                 frame,
                 pose_classification,
                 pose_classification_filtered,
                 repetitions_count):
        """Renders pose classifcation and counter until given frame."""
        # Extend classification history.
        self._pose_classification_history.append(pose_classification) # 获取分类结果
        self._pose_classification_filtered_history.append(
            pose_classification_filtered)  # 获取的是平滑之后的分类结果

        # Output frame with classification plot and counter.
        # 将frame转化成一个image对象 ，将经过mediapipe处理生成关键点以及关节方向，但是没有添加左上角图例和右上角计数值的frame
        output_img = Image.fromarray(frame)


        output_width = output_img.size[0]
        output_height = output_img.size[1]  # 获取output_img的宽高

        # Draw the plot.
        # thumbnail()函数的参数: 1.size 指定缩略图的大小（width，height），resample：指定缩略图的重采样方法，Image.ANTIALIAS缩略图生成过程中使用的抗锯齿滤波器

        img = self._plot_classification_history(output_width, output_height)
        img.thumbnail((int(output_width * self._plot_max_width),
                       int(output_height * self._plot_max_height)),
                      Image.ANTIALIAS) # 将图片缩放到指定的宽度与高度
        output_img.paste(img, # 要黏贴的图像（源图像）
                         (int(output_width * self._plot_location_x),
                          int(output_height * self._plot_location_y))) # 将图表粘贴到实时获取的帧的左上方
        # paste()函数，将一张图片复制到另外一张图片上，在此代码中将img粘贴到output_img的指定位置。

        # Draw the count.
        output_img_draw = ImageDraw.Draw(output_img) # 创建一个可绘制的对象
        if self._counter_font is None: # 检查计数器使用的字体是否已经加载
            font_size = int(output_height * self._counter_font_size)  # 图像字体的大小
            font_request = requests.get( # 使用requests库发送HTTP请求，从指定的字体路径获取字体文件。
                self._counter_font_path, allow_redirects=True)
            self._counter_font = ImageFont.truetype(  # 函数加载字体文件
                io.BytesIO(font_request.content), size=font_size)  # 字体文件路径以及字体的大小
        else:
            font_size = int(output_height * self._counter_font_size)
            self._counter_font = ImageFont.truetype('Roboto-Regular.ttf', size=font_size)
        # 图像上绘制文本
        output_img_draw.text((output_width * self._counter_location_x,
                              output_height * self._counter_location_y),
                             str(repetitions_count),  # 表示要绘制的文本内容即当前运动计数的个数
                             font=self._counter_font,  # 使用的字体
                             fill=self._counter_font_color)   # 字体的颜色

        return output_img    # 返回的是带有姿态分类历史和计算器的最终output_img图像

    def _plot_classification_history(self, output_width, output_height):
        fig = plt.figure(figsize=self._plot_figsize)   # 创建一个指定大小的图表对象

        for classification_history,label in [(self._pose_classification_history,'Source data'),
                                             (self._pose_classification_filtered_history,'Filtered data')]:
            classification_history = classification_history[-100:]
            # 只显示分类历史列表的前100个值
            y = []
            for classification in classification_history:  # 遍历分类结果的列表
                if classification is None:
                    y.append(None)
                elif self._class_name in classification:
                    y.append(classification[self._class_name])
                else:
                    y.append(0)
            plt.plot(y, linewidth=7, label=label) # 根据y值列表绘制折线图，设置线宽为7

        plt.grid(axis='y', alpha=0.75)  # 在图表上添加网格线，即在纵轴方向的网格线
        plt.xlabel('Frame',fontsize=24)
        plt.ylabel('Confidence',fontsize=24)
        plt.xticks(fontsize=24)  # Set the font size for the x-axis ticks
        plt.yticks(fontsize=24)
        plt.title('Classification history for M_end', fontsize=26, pad=20)   # 标题
        # plt.title('Classification history for `{}`'.format(self._class_name))
        plt.legend(loc='upper right', fontsize=16, bbox_to_anchor=(1.1, 1.15))

        if self._plot_y_max is not None:
            plt.ylim(top=self._plot_y_max)
        if self._plot_x_max is not None:
            plt.xlim(right=self._plot_x_max)

        # Convert plot to image.
        buf = io.BytesIO()
        dpi = min(
            output_width * self._plot_max_width / float(self._plot_figsize[0]),
            output_height * self._plot_max_height / float(self._plot_figsize[1]))
        fig.savefig(buf, dpi=dpi)
        buf.seek(0)
        img = Image.open(buf)
        plt.close()

        return img


# 显示图像
def show_image(img, figsize=(10, 10)):
    """Shows output PIL image."""
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.show()
