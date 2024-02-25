# -*- coding:utf-8 -*-
# @Time: 2024-01-12 23:57
# @Author: zer01
# @File：main.py
# @Description:
'''
https://doc.qt.io/qt-5/qtwidgets-module.html

'''
import os.path
import shutil
import sys
import threading

# PyQt5
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

# yolov5-detect.py
import torch
import platform
from pathlib import Path
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode

# SIP与QT版本不匹配警告
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class MainWindow(QTabWidget):

    def __init__(self):
        super(MainWindow, self).__init__()
        # self.weights = "runs/train/exp4/weights/best.pt"         # "pt/yolov5x6.pt"
        self.weights = "pt/yolov5x6.pt"

        # cpu版
        # self.device = "cpu"
        # self.half = False

        # gpu版
        self.device = "cuda:0"
        self.half = False

        self.data = "data/coco128.yaml"
        # self.data = "None"

        self.dnn = True
        self.output_size = 480
        self.img2predict = ""
        self.webcam = True
        self.model = self.model_load(self.weights, self.device, self.data, self.dnn, self.half)
        self.stopEvent = threading.Event()  # https://blog.csdn.net/sky0Lan/article/details/107727757
        self.stopEvent.clear()
        self.init_ui()

    def init_ui(self):
        # self.resize(1000, 800)
        self.setFixedSize(1200, 800)    # 不可拉伸缩放，同时禁用最大化
        # 可使窗口打开时最大化按钮可用
        # QWidget::setMaximumSize: (/MainWindow) The largest allowed size is (16777215,16777215)
        MAIN_SIZE_MAX = QSize(16777215,16777215)
        self.setMaximumSize(MAIN_SIZE_MAX)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)

        self.setWindowTitle('目标检测系统v0.2 --by R4bbit')
        self.setWindowIcon(QIcon("./img/yolo_logo.png"))
        # 图片检测子界面
        font_title = QFont('楷体', 16)
        font_main = QFont('楷体', 14)
        # 图片识别界面, 两个按钮，上传图片和显示结果
        img_detection_widget = QWidget()
        img_detection_layout = QVBoxLayout()
        img_detection_title = QLabel("图片识别功能")
        img_detection_title.setFont(font_title)
        mid_img_widget = QWidget()
        mid_img_layout = QHBoxLayout()
        self.left_img = QLabel()
        self.right_img = QLabel()

        self.left_img.setPixmap(QPixmap("./img/left.png"))      # 200×200
        self.left_img.setFixedSize(600, 600)        # 修改图片大小为600×600
        self.right_img.setPixmap(QPixmap("./img/right.png"))      # 200×200
        self.right_img.setFixedSize(600, 600)

        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)
        mid_img_layout.addWidget(self.left_img)
        mid_img_layout.addStretch(0)
        mid_img_layout.addWidget(self.right_img)
        mid_img_widget.setLayout(mid_img_layout)
        up_img_button = QPushButton("上传图片")
        det_img_button = QPushButton("开始检测")
        up_img_button.clicked.connect(self.upload_img)
        det_img_button.clicked.connect(self.detect_img)
        up_img_button.setFont(font_main)
        det_img_button.setFont(font_main)
        up_img_button.setStyleSheet("QPushButton{color:white}"
                                    "QPushButton:hover{background-color: rgb(2,110,180);}"
                                    "QPushButton{background-color:rgb(48,124,208)}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:5px}"
                                    "QPushButton{padding:5px 5px}"
                                    "QPushButton{margin:5px 5px}")
        det_img_button.setStyleSheet("QPushButton{color:white}"
                                     "QPushButton:hover{background-color: rgb(2,110,180);}"
                                     "QPushButton{background-color:rgb(48,124,208)}"
                                     "QPushButton{border:2px}"
                                     "QPushButton{border-radius:5px}"
                                     "QPushButton{padding:5px 5px}"
                                     "QPushButton{margin:5px 5px}")
        img_detection_layout.addWidget(img_detection_title, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(up_img_button)
        img_detection_layout.addWidget(det_img_button)
        img_detection_widget.setLayout(img_detection_layout)

        # 视频识别界面
        vid_detection_widget = QWidget()
        vid_detection_layout = QVBoxLayout()
        vid_title = QLabel("视频检测功能")
        vid_title.setFont(font_title)
        self.vid_img = QLabel()

        self.vid_img.setPixmap(QPixmap("img/video_img.jpg"))
        # self.vid_img.setFixedSize(1200, 600)

        vid_title.setAlignment(Qt.AlignCenter)
        self.vid_img.setAlignment(Qt.AlignCenter)
        self.webcam_detection_btn = QPushButton("摄像头实时监测")
        self.mp4_detection_btn = QPushButton("视频文件检测")
        self.vid_stop_btn = QPushButton("停止检测")
        self.webcam_detection_btn.setFont(font_main)
        self.mp4_detection_btn.setFont(font_main)
        self.vid_stop_btn.setFont(font_main)
        self.webcam_detection_btn.setStyleSheet("QPushButton{color:white}"
                                                "QPushButton:hover{background-color: rgb(2,110,180);}"
                                                "QPushButton{background-color:rgb(48,124,208)}"
                                                "QPushButton{border:2px}"
                                                "QPushButton{border-radius:5px}"
                                                "QPushButton{padding:5px 5px}"
                                                "QPushButton{margin:5px 5px}")
        self.mp4_detection_btn.setStyleSheet("QPushButton{color:white}"
                                             "QPushButton:hover{background-color: rgb(2,110,180);}"
                                             "QPushButton{background-color:rgb(48,124,208)}"
                                             "QPushButton{border:2px}"
                                             "QPushButton{border-radius:5px}"
                                             "QPushButton{padding:5px 5px}"
                                             "QPushButton{margin:5px 5px}")
        self.vid_stop_btn.setStyleSheet("QPushButton{color:white}"
                                        "QPushButton:hover{background-color: rgb(2,110,180);}"
                                        "QPushButton{background-color:rgb(48,124,208)}"
                                        "QPushButton{border:2px}"
                                        "QPushButton{border-radius:5px}"
                                        "QPushButton{padding:5px 5px}"
                                        "QPushButton{margin:5px 5px}")
        self.webcam_detection_btn.clicked.connect(self.open_cam)
        self.mp4_detection_btn.clicked.connect(self.upload_vid)
        self.vid_stop_btn.clicked.connect(self.close_detect)
        # 添加组件到布局上
        vid_detection_layout.addWidget(vid_title)
        vid_detection_layout.addWidget(self.vid_img)
        vid_detection_layout.addWidget(self.webcam_detection_btn)
        vid_detection_layout.addWidget(self.mp4_detection_btn)
        vid_detection_layout.addWidget(self.vid_stop_btn)
        vid_detection_widget.setLayout(vid_detection_layout)

        self.left_img.setAlignment(Qt.AlignCenter)
        self.addTab(img_detection_widget, '图片检测')
        self.addTab(vid_detection_widget, '视频检测')
        self.setTabIcon(0, QIcon('img/cat.png'))
        self.setTabIcon(1, QIcon('img/dog.png'))

    """=======================模型加载======================="""
    @torch.no_grad()    # 不使用梯度
    def model_load(self, weights, device, data, dnn=False, half=False):
        try:
            # Load model
            device = select_device(device)
            model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
            stride, names, pt = model.stride, model.names, model.pt
            # imgsz = check_img_size(imgsz, s=stride)  # check image size
            print("YOLOv5模型加载完毕~ 🚀🚀🚀🚀🚀🚀")
            return model
        except Exception as e:
            print("YOLOv5模型加载失败~ 🚀🚀🚀🚀🚀🚀")
            print(f"An error occurred: {e}")

    """=======================图片上传======================="""
    def upload_img(self):
        IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"  # include image suffixes
        file_formats = ' '.join(f'*.{format}' for format in IMG_FORMATS)
        # file_filter = f'{file_formats};;All Files (*)'
        imgName, _ = QFileDialog.getOpenFileName(self, '打开图片', '', file_formats)
        # self.img2predict = imgName
        # try:
        #     img = QPixmap(imgName).scaled(self.left_img.width(), self.left_img.height())
        #     self.left_img.setPixmap(img)
        # except Exception as e:
        #     print(f"An error occurred: {e}")
        if imgName:
            suffix = imgName.split(".")[-1]
            # print(suffix)
            save_path = os.path.join("data/tmp", "tmp_upload." + suffix)
            # print(save_path)
            shutil.copy(imgName, save_path)    # shutil.copy(src, dst)：将文件src复制至dst。dst可以是个目录，会在该目录下创建与src同名的文件
            # 应该调整一下图片的大小，然后统一防在一起
            im0 = cv2.imread(save_path)
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("data/tmp/upload_show_result.png", im0)
            self.img2predict = imgName
            self.left_img.setPixmap(QPixmap("data/tmp/upload_show_result.png"))
            # 上传图片之后右侧的图片重置，
            self.right_img.setPixmap(QPixmap("img/right.png"))

    """=======================图片检测======================="""
    def detect_img(self):
        model = self.model
        device = select_device(self.device)
        source = self.img2predict
        imgsz = (640, 640)
        augment = False
        visualize = False
        conf_thres = 0.25  # confidence threshold
        iou_thres = 0.45  # NMS IOU threshold
        classes = None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms = False,  # class-agnostic NMS
        max_det = 1000  # maximum detections per image
        vid_stride = 1  # video frame-rate stride
        line_thickness = 3  # bounding box thickness (pixels)
        save_crop = False  # save cropped prediction boxes
        hide_conf = False # hide confidences
        output_size = self.output_size
        hide_labels = False
        # save_path = "runs/detect"

        if source == "":
            QMessageBox.warning(self, "请上传", "请先上传图片再进行检测")
        else:
            # Load model
            # model = DetectMultiBackend(self.weights, device=device, dnn=self.dnn, data=self.data, fp16=self.half)
            stride, names, pt = model.stride, model.names, model.pt
            imgsz = check_img_size(imgsz, s=stride)  # check image size

            # Dataloader
            bs = 1  # batch_size
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            vid_path, vid_writer = [None] * bs, [None] * bs
            """=======================算法核心：推理======================="""
            # Run inference
            model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
            seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
            for path, im, im0s, vid_cap, s in dataset:
                '''
                path: 文件路径（source）
                im: resize后的image
                im0s：原始image
                vid_cap：
                s：image基本信息（size、path）
                '''
                with dt[0]:
                    im = torch.from_numpy(im).to(model.device)
                    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim
                    if model.xml and im.shape[0] > 1:
                        ims = torch.chunk(im, im.shape[0], 0)

                # Inference
                with dt[1]:
                    # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                    if model.xml and im.shape[0] > 1:
                        pred = None
                        for image in ims:
                            if pred is None:
                                pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                            else:
                                pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)),
                                                 dim=0)
                        pred = [pred, None]
                    else:
                        pred = model(im, augment=augment, visualize=visualize)
                # NMS
                with dt[2]:
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

                # Process predictions
                for i, det in enumerate(pred):  # per image
                    seen += 1
                    webcam = False
                    if webcam:  # batch_size >= 1
                        p, im0, frame = path[i], im0s[i].copy(), dataset.count
                        s += f"{i}: "
                    else:
                        p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

                    p = Path(p)  # to Path
                    # save_path = str(save_dir / p.name)  # im.jpg
                    # txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
                    # s += "%gx%g " % im.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    imc = im0.copy() if save_crop else im0  # for save_crop
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()  # detections per class
                            # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)  # integer class
                            label = names[c] if hide_conf else f"{names[c]}"
                            confidence = float(conf)
                            confidence_str = f"{confidence:.2f}"

                            # if save_csv:
                            #     write_to_csv(p.name, label, confidence_str)

                            # if save_txt:  # Write to file
                            #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            #     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            #     with open(f"{txt_path}.txt", "a") as f:
                            #         f.write(("%g " * len(line)).rstrip() % line + "\n")
                            save_img = True
                            view_img = True
                            if save_img or save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                                annotator.box_label(xyxy, label, color=colors(c, True))
                            # if save_crop:
                            #     save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

                    # Stream results
                    """图片展示"""
                    im0 = annotator.result()
                    # try:
                    #     img = QPixmap(im0).scaled(self.right_img.width(), self.right_img.height())
                    #     self.right_img.setPixmap(img)
                    # except Exception as e:
                    #     print(f"An error occurred: {e}")

                    resize_scale = output_size / im0.shape[0]
                    im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
                    cv2.imwrite("data/tmp/result_img.png", im0)
                    # 目前的情况来看，应该只是ubuntu下会出问题，但是在windows下是完整的，所以继续
                    self.right_img.setPixmap(QPixmap("data/tmp/result_img.png"))

                    # if view_img:
                    #     if platform.system() == "Linux" and p not in windows:
                    #         windows.append(p)
                    #         cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    #         cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    #     cv2.imshow(str(p), im0)
                    #     cv2.waitKey(1)  # 1 millisecond
                    #
                    # # Save results (image with detections)
                    # if save_img:
                    #     if dataset.mode == "image":
                    #         cv2.imwrite(save_path, im0)
                    #     else:  # 'video' or 'stream'
                    #         if vid_path[i] != save_path:  # new video
                    #             vid_path[i] = save_path
                    #             if isinstance(vid_writer[i], cv2.VideoWriter):
                    #                 vid_writer[i].release()  # release previous video writer
                    #             if vid_cap:  # video
                    #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    #             else:  # stream
                    #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
                    #             save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                    #             vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h))
                    #         vid_writer[i].write(im0)

                # Print time (inference-only)
                # LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

            # Print results
            # t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
            # LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
            # if save_txt or save_img:
            #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
            #     LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
            # if update:
            #     strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

    """=======================摄像头打开======================="""
    def open_cam(self):
        self.webcam_detection_btn.setEnabled(False)     # 打开摄像头之后，禁用按钮-摄像头实时监测
        self.mp4_detection_btn.setEnabled(False)     # 打开摄像头之后，禁用按钮-视频检测
        self.vid_stop_btn.setEnabled(True)     # 打开摄像头之后，只能按-停止检测
        self.vid_source = '0'
        self.webcam = True
        th = threading.Thread(target=self.detect_vid)
        th.start()

    """=======================视频上传======================="""
    def upload_vid(self):
        VID_FORMATS = "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv"  # include video suffixes
        file_formats = ' '.join(f'*.{format}' for format in VID_FORMATS)
        vidName, _ = QFileDialog.getOpenFileName(self, '打开视频', '', file_formats)    # vidName, vidType
        if vidName:
            # 上传视频之后，只有按-停止检测，才能结束，才能按其他按钮
            self.webcam_detection_btn.setEnabled(False)
            self.mp4_detection_btn.setEnabled(False)
            self.vid_stop_btn.setEnabled(True)
            self.vid_source = vidName
            self.webcam = False
            th = threading.Thread(target=self.detect_vid)
            th.start()

    def detect_vid(self):
        model = self.model
        device = select_device(self.device)
        # source = self.img2predict
        source = str(self.vid_source)
        imgsz = (640, 640)
        augment = False
        visualize = False
        conf_thres = 0.25  # confidence threshold
        iou_thres = 0.45  # NMS IOU threshold
        classes = None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms = False,  # class-agnostic NMS
        max_det = 1000  # maximum detections per image
        # vid_stride = 1  # video frame-rate stride
        vid_stride = 2  # 每2帧进行视频目标检测
        line_thickness = 3  # bounding box thickness (pixels)
        save_crop = False  # save cropped prediction boxes
        hide_conf = False  # hide confidences
        output_size = 640
        hide_labels = False
        webcam = self.webcam
        # save_path = "runs/detect"

        # Load model
        # model = DetectMultiBackend(self.weights, device=device, dnn=self.dnn, data=self.data, fp16=self.half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        bs = 1  # batch_size
        if webcam:
            view_img = check_imshow(warn=True)
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            bs = len(dataset)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        vid_path, vid_writer = [None] * bs, [None] * bs

        """=======================算法核心：推理======================="""
        # Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
        for path, im, im0s, vid_cap, s in dataset:
            '''
            path: 文件路径（source）
            im: resize后的image
            im0s：原始image
            vid_cap：
            s：image基本信息（size、path）
            '''
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                if model.xml and im.shape[0] > 1:
                    ims = torch.chunk(im, im.shape[0], 0)

            # Inference
            with dt[1]:
                # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                if model.xml and im.shape[0] > 1:
                    pred = None
                    for image in ims:
                        if pred is None:
                            pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                        else:
                            pred = torch.cat(
                                (pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)),
                                dim=0)
                    pred = [pred, None]
                else:
                    pred = model(im, augment=augment, visualize=visualize)
            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f"{i}: "
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

                p = Path(p)  # to Path
                # save_path = str(save_dir / p.name)  # im.jpg
                # txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
                # s += "%gx%g " % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = names[c] if hide_conf else f"{names[c]}"
                        confidence = float(conf)
                        confidence_str = f"{confidence:.2f}"

                        save_img = True
                        view_img = True
                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                            annotator.box_label(xyxy, label, color=colors(c, True))

                # Stream results
                """图片展示"""
                im0 = annotator.result()
                # try:
                #     img = QPixmap(im0).scaled(self.right_img.width(), self.right_img.height())
                #     self.right_img.setPixmap(img)
                # except Exception as e:
                #     print(f"An error occurred: {e}")

                # frame = im0
                # resize_scale = output_size / frame.shape[0]
                # frame_resized = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
                # cv2.imwrite("data/tmp/single_result_vid.jpg", frame_resized)
                # self.vid_img.setPixmap(QPixmap("data/tmp/single_result_vid.jpg"))

                # 缩放图像
                resize_scale = output_size / im0.shape[0]
                frame_resized = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)

                # 将numpy数组转换为QPixmap并设置到QLabel中
                h, w, c = frame_resized.shape
                qImg = QImage(frame_resized.data, w, h, w * c, QImage.Format_RGB888).rgbSwapped()
                pixmap = QPixmap.fromImage(qImg)
                self.vid_img.setPixmap(pixmap)


            # waitKey() 函数的功能是不断刷新图像 , 频率时间为delay , 单位为ms
            # 返回值为当前键盘按键值，即ascii码
            if cv2.waitKey(2) & self.stopEvent.is_set() == True:
                self.stopEvent.clear()
                self.webcam_detection_btn.setEnabled(True)
                self.mp4_detection_btn.setEnabled(True)
                # self.reset_vid()
                break

                # if view_img:
                #     if platform.system() == "Linux" and p not in windows:
                #         windows.append(p)
                #         cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                #         cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                #     cv2.imshow(str(p), im0)
                #     cv2.waitKey(1)  # 1 millisecond
                #
                # # Save results (image with detections)
                # if save_img:
                #     if dataset.mode == "image":
                #         cv2.imwrite(save_path, im0)
                #     else:  # 'video' or 'stream'
                #         if vid_path[i] != save_path:  # new video
                #             vid_path[i] = save_path
                #             if isinstance(vid_writer[i], cv2.VideoWriter):
                #                 vid_writer[i].release()  # release previous video writer
                #             if vid_cap:  # video
                #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
                #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                #             else:  # stream
                #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
                #             save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                #             vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h))
                #         vid_writer[i].write(im0)

            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # Print results
        t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
        # if save_txt or save_img:
        #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        #     LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        # if update:
        #     strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

    """=======================界面重置======================="""
    def reset_vid(self):
        self.webcam_detection_btn.setEnabled(True)
        self.mp4_detection_btn.setEnabled(True)
        self.vid_img.setPixmap(QPixmap("img/video_img.jpg"))
        self.vid_source = '0'
        self.webcam = True

    """=======================视频重置======================="""
    # self.vid_stop_btn.clicked.connect(self.close_detect)
    def close_detect(self):
        self.stopEvent.set()    # 将event的标志设置为True，调用wait方法的所有线程将被唤醒
        self.reset_vid()

    # 界面关闭，函数名不可改，继承QWidget
    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     '关闭',
                                     "你确认退出？",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()


if __name__ == '__main__':
    # stylesheet = """
    #         MainWindow {
    #             background-image: url("./img/moon.jpg");
    #             background-repeat: no-repeat;
    #             background-position: center;
    #         }
    #     """
    app = QApplication(sys.argv)
    # app.setStyleSheet(stylesheet)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
