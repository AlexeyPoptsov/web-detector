import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import cv2

import torch
from mmdet.apis import inference_detector, init_detector
from collections import Counter

import time

FPS = np.zeros(5)
FPS_index = 0


class Detector(object):
    def __init__(self, model_IDs: list, path_images: str):
        MODEL_CONFIG_FILES: str = 'app/model_config_1/'
        # MODEL_CONFIG_FILES: str = 'model_config_1/'

        self.model_IDs: list = model_IDs
        if len(self.model_IDs) == 0:
            self.model_IDs = [0]

        self.path_images: str = path_images

        #self.score_threshold = 0.4

        if torch.cuda.is_available():
            self.DEVICE = torch.device('cuda')
        else:
            self.DEVICE = torch.device('cpu')
        self.device_properties = torch.cuda.get_device_properties(self.DEVICE)


        self.models: list = []
        if 2 in self.model_IDs:
            model_dict = dict(name='YOLOv3', config=MODEL_CONFIG_FILES + 'yolov3_mobilenetv2_320_300e_coco.py',
                              checkpoint=MODEL_CONFIG_FILES + 'yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth')
            model_dict['model'] = init_detector(model_dict['config'], model_dict['checkpoint'], device='cuda:0')
            model_dict['score_threshold'] =0.2
            self.models.append(model_dict)


        if 3 in self.model_IDs:
            model_dict = dict(name='faster_rcnn_r50',
                              config=MODEL_CONFIG_FILES + 'faster_rcnn_r50_fpn_tnr-pretrain_1x_coco.py',
                              checkpoint=MODEL_CONFIG_FILES + 'faster_rcnn_r50_fpn_tnr-pretrain_1x_coco_20220320_085147-efedfda4.pth')
            model_dict['model'] = init_detector(model_dict['config'], model_dict['checkpoint'], device='cuda:0')
            model_dict['score_threshold'] = 0.6
            self.models.append(model_dict)

        if 4 in self.model_IDs:
            model_dict = dict(name='mask_rcnn_r50',
                              config=MODEL_CONFIG_FILES + 'mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py',
                              checkpoint=MODEL_CONFIG_FILES + 'mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth ')
            model_dict['model'] = init_detector(model_dict['config'], model_dict['checkpoint'], device='cuda:0')
            model_dict['score_threshold'] = 0.5
            self.models.append(model_dict)

        if 5 in self.model_IDs:
            model_dict = dict(name='vfnet_r101', config=MODEL_CONFIG_FILES + 'vfnet_r101_fpn_mstrain_2x_coco.py',
                              checkpoint=MODEL_CONFIG_FILES + 'vfnet_r101_fpn_mstrain_2x_coco_20201027pth-4a5d53f1.pth')
            model_dict['model'] = init_detector(model_dict['config'], model_dict['checkpoint'], device='cuda:0')
            model_dict['score_threshold'] = 0.4
            self.models.append(model_dict)

        # if 10 in self.model_IDs:
        #     model_dict = dict(name='YOLOv5', config=MODEL_CONFIG_FILES + 'yolov5_m-p6-v62_syncbn_fast_8xb16-300e_coco.py ',
        #                       checkpoint=MODEL_CONFIG_FILES + 'yolov5_m-p6-v62_syncbn_fast_8xb16-300e_coco_20221027_230453-49564d58.pth ')
        #     model_dict['model'] = init_detector(model_dict['config'], model_dict['checkpoint'], device='cuda:0')
        #     self.models.append(model_dict)

        # if 6 in self.model_IDs:
        #     model_dict = dict(name='YOLO8', config='', checkpoint='')
        #     self.model_YOLO8 = model = YOLO('yolov8n.pt')
        #     model_dict['model'] = self.model_YOLO8
        #     self.models.append(model_dict)

    def transform(self, img):
        # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
        ratio = 1.0
        if (img.shape[1] > 1280) or (img.shape[0] > 1280):
            if (img.shape[1] > img.shape[0]):
                ratio = 1280 / img.shape[1]
            else:
                ratio = 1280 / img.shape[0]
        img =cv2.resize(img, (np.int32(ratio * img.shape[1]), np.int32(ratio * img.shape[0])), interpolation = cv2.INTER_AREA)

        return img

    def count_classes(self, model, score_threshold, result):
        if type(result) == list:
            bbox_result = result
        else:
            # if instance segmentation
            bbox_result, segm_result = result

        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
        labels = np.concatenate(labels)
        bboxes = np.vstack(bbox_result)

        # Номера боксов c уверенностью выше
        labels_impt = np.where(bboxes[:, -1] > score_threshold)[0]

        # Уверенность для этих номеров
        confidence_imp = bboxes[bboxes[:, -1] > score_threshold][:, 4]

        # Коды классов для этих номеров
        labels_impt_list = [labels[i] for i in labels_impt]

        classes_names = model.CLASSES

        # Названия классов для номеров боксов
        labels_class = [classes_names[i] for i in labels_impt_list]

        result_dict = Counter(labels_class)

        return result_dict

    def detect_from_frame(self, img, ID=0):
        model = self.models[ID]['model']
        score_threshold = self.models[ID]['score_threshold']

        result = inference_detector(model, img)
        img = model.show_result(img, result, score_thr=score_threshold, show=False)
        count_classes = self.count_classes(self.models[ID]['model'], score_threshold, result)

        return (img, count_classes)

    def show_frame(self, img, ID):
        start_time = time.time()

        # perform the resizing
        # h = 320.0
        # r = h / image.shape[0]
        # dim = (int(image.shape[1] * r), int(h))
        # image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        image, count_classes = self.detect_from_frame(img, ID)

        # draw classes block
        x = image.shape[1] - 200
        y = image.shape[0] - 150
        cv2.putText(image, 'Detected objects', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (192, 192, 192), 2)
        image = cv2.rectangle(image, (x, y), (x + 200 - 2, y + 200 - 2), (192, 192, 192), 1)
        step = 20
        for _class in count_classes:
            cv2.putText(image,
                        f'{_class}: {count_classes[_class]}',
                        (x + 10, y + step),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (192, 192, 192), 1)
            step += 20

        work_time = time.time() - start_time
        global FPS, FPS_index
        FPS_index = (FPS_index + 1) % 5
        FPS[FPS_index] = (1 / work_time)
        cv2.putText(image, f'({image.shape[1]}x{image.shape[0]}), FPS: {FPS.mean():.2f}',
                    (x + 10, y + 140), cv2.FONT_HERSHEY_SIMPLEX, .5, (192, 192, 192), 1)

        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()

        return frame

    def detect_from_file(self, image_name):
        fig = plt.figure()

        img = cv2.imread(image_name)
        self.origin_image_shape = (img.shape[1], img.shape[0], img.shape[2])

        img = self.transform(img)
        self.image_shape = (img.shape[1], img.shape[0], img.shape[2])

        for i, model_dict in enumerate(self.models):
            model = model_dict['model']

            ## Первое обращение очень медленное, для корректного измерения времени одно пустое предсказание
            if i == 0:
                result = inference_detector(model, img)

            start_time = time.time()
            result = inference_detector(model, img)
            work_time = time.time() - start_time

            score_threshold = model_dict['score_threshold']

            self.models[i]['count_classes'] = self.count_classes(model, score_threshold, result)

            result_name = f'result{i}.jpg'
            self.models[i]['result_name'] = result_name
            self.models[i]['time'] = f'{work_time:.4f}'

            model.show_result(img, result, score_thr=score_threshold, show=False,
                              out_file=self.path_images + result_name)
            self.models[i]['score_threshold'] = score_threshold

            plt.close(fig)

    if __name__ == '__main__':
        detector = Detector(model_IDs=[0, 1, 2, 3, 4, 5], path_images='/app/static/uploads/')
        detector.get_result(r'F:/python/web-detector/app/static/uploads/demo.jpg')
