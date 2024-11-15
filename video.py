import cv2
import copy
import numpy as np
import matplotlib as pltpip

def Transitions(video, canvas1, canvas2, method="fade", num_frames=60):
    """
    镜头转换，过渡效果

    args:
    video:  视频，VideoWriter对象
    canvas: 过渡的两个帧画面
    method: 过渡方法。fade:淡入淡出；slide:滑动；wipe:擦除
    num_frames: 过渡帧数，默认为60

    :return:
    video： VideoWriter对象
    """

    for i in range(num_frames):
        if method == "slide":
            x_offset = int((i / num_frames) * frame_width)
            slide_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            slide_frame[:, x_offset:] = canvas1[:, x_offset:]
            slide_frame[:, :x_offset] = canvas2[:, :x_offset]
            out.write(slide_frame)
        elif method == "wipe":
            mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            cv2.rectangle(mask, (0, 0), (int((i / num_frames) * frame_width), frame_height), 255, -1)
            wipe_frame = np.where(mask[:, :, np.newaxis] == 255, canvas2, canvas1)
            out.write(wipe_frame)
        else:
            alpha = i / num_frames  # 透明度
            beta = 1.0 - alpha
            blended_frame = cv2.addWeighted(canvas1, beta, canvas2, alpha, 0)
            out.write(blended_frame)
    return video


if __name__ == "__main__":
    image = cv2.imread(r'images\photo.png')
    resized_image = cv2.resize(image, (480, 480))
    frame_width = 960
    frame_height = 640
    fps = 60

    # 创建 FourCC 代码
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # 创建 VideoWriter 对象
    out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

    # 检查图像是否成功读取
    if image is None:
        print("Error: 无法读取图像")
    else:
        # 获取图像的宽度和高度
        image_height, image_width = image.shape[:2]

        # 创建背景
        black_canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        white_canvas = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

        ''' 片头 '''
        # 定义字体
        font = cv2.FONT_HERSHEY_TRIPLEX

        # 定义文字内容和位置
        begin_text_1 = "Welcome to this video!"
        begin_text_2 = "This is a simple beginning of the video, which is"
        begin_text_3 = "the first homework of the Computer Vision."
        org = (50, frame_height // 2 - 100)
        font_scale = 1
        color = (0, 0, 0)
        thickness = 2
        line_spacing = 50

        # 在白色画布上写开头文字
        begin_canvas = copy.deepcopy(white_canvas)
        cv2.putText(begin_canvas, begin_text_1, (org[0], org[1]), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(begin_canvas, begin_text_2, (org[0], org[1] + line_spacing), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(begin_canvas, begin_text_3, (org[0], org[1] + line_spacing * 2), font, font_scale, color, thickness, cv2.LINE_AA)

        for _ in range(fps * 2):
            out.write(begin_canvas)

        ''' 照片 '''
        photo_canvas = copy.deepcopy(black_canvas)

        # 计算图像在画布上的位置
        x_offset = (frame_width - image_width) // 2
        y_offset = (frame_height - image_height) // 2

        # 将图像放置在画布的中心
        photo_canvas[y_offset:y_offset + image_height, x_offset:x_offset + image_width] = image

        # 中文不能写
        # name_text = "姓名： 孙思研"
        # ID_text = "学号: 22421126"
        # org = (540, 540)
        # color = (255, 255, 255)
        # cv2.putText(photo_canvas, name_text, (org[0], org[1]), font, font_scale, color, thickness, cv2.LINE_AA)
        # cv2.putText(photo_canvas, ID_text, (org[0], org[1] + 50), font, font_scale, color, thickness, cv2.LINE_AA)

        num_frames_first_part = fps * 2

        # 镜头切换
        out = Transitions(out, begin_canvas, photo_canvas, method="fade")

        # 写入前三秒的帧
        for _ in range(num_frames_first_part):
            out.write(photo_canvas)

        ''' 画形状 '''
        # 形状绘制的镜头切换
        shape_canvas = copy.deepcopy(white_canvas)
        shape_text = "Shape Drawing"
        org = (frame_width // 2 - 130, 50)
        color = (0, 0, 0)
        cv2.putText(shape_canvas, shape_text, org, font, font_scale, color, thickness, cv2.LINE_AA)

        out = Transitions(out, photo_canvas, shape_canvas, method="fade", num_frames=30)

        pts = np.array([[100, 200], [300, 200], [300, 400], [100, 400], [100, 200]], np.int32)
        pts = pts.reshape((-1, 1, 2))

        # 矩形
        color = (0, 255, 0)  # 绿色
        for i in range(len(pts) - 1):
            for j in range(1, 51):  # 50步完成一条线段的绘制
                x1, y1 = pts[i][0]
                x2, y2 = pts[i + 1][0]
                x = int(x1 + (x2 - x1) * (j / 50))
                y = int(y1 + (y2 - y1) * (j / 50))
                cv2.line(shape_canvas, (x1, y1), (x, y), color, thickness)
                out.write(shape_canvas)  # 写入帧

        # 圆形
        center = (500, 300)
        radius = 100
        color = (0, 0, 255)  # 红色

        # 上一个点的坐标
        initial_angle = 3  # 从5度开始，避免从圆心开始
        initial_x = int(center[0] + radius * np.cos(np.deg2rad(initial_angle)))
        initial_y = int(center[1] + radius * np.sin(np.deg2rad(initial_angle)))
        prev_x, prev_y = initial_x, initial_y

        # 逐步绘制点绕着圆心旋转形成的线
        for angle in range(0, 360, 3):  # 每3度绘制一次
            x = int(center[0] + radius * np.cos(np.deg2rad(angle)))
            y = int(center[1] + radius * np.sin(np.deg2rad(angle)))
            cv2.line(shape_canvas, (prev_x, prev_y), (x, y), color, thickness)
            out.write(shape_canvas)
            prev_x, prev_y = x, y

        cv2.line(shape_canvas, (prev_x, prev_y), (initial_x, initial_y), color, thickness)
        out.write(shape_canvas)

        # 多边形
        pts = np.array([[700, 300], [800, 200], [900, 300], [850, 400], [750, 400], [700, 300]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        color = (255, 0, 0)  # 白色
        for i in range(len(pts) - 1):
            for j in range(1, 51):
                x1, y1 = pts[i][0]
                x2, y2 = pts[i + 1][0]
                x = int(x1 + (x2 - x1) * (j / 50))
                y = int(y1 + (y2 - y1) * (j / 50))
                cv2.line(shape_canvas, (x1, y1), (x, y), color, thickness)
                out.write(shape_canvas)

        for _ in range(fps):  # 剩余的帧数 (20 fps * 7 seconds)
            out.write(shape_canvas)

        # 片尾
        end_text = "That's all, thank you!"
        org = (frame_width // 2 - 200, frame_height // 2 - 50)
        color = (0, 0, 0)
        end_canvas = copy.deepcopy(white_canvas)
        cv2.putText(end_canvas, end_text, (org[0], org[1]), font, font_scale, color, thickness, cv2.LINE_AA)

        out = Transitions(out, shape_canvas, end_canvas, method='slide')

        for _ in range(fps * 3):
            out.write(end_canvas)

        print('vedio created')
        print('press to continue')
        # 等待按键按下
        input()

        ''' 播放 '''
        cap = cv2.VideoCapture('output.avi')
        paused = False
        while cap.isOpened():
            if paused:
                key = cv2.waitKey(0)
            else:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imshow('Video', frame)
                key = cv2.waitKey(10)

            if key == ord(' '):  # 空格键
                paused = not paused
            elif key == ord('q'):  # 'q' 键退出
                break

        cap.release()
        cv2.destroyAllWindows()

    # 释放 VideoWriter 对象
    out.release()


