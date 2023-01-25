import numpy as np

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

import torch
import mmcv
import time
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from collections import Counter

# print(torch.cuda.get_device_properties(DEVICE))

# img_name1 = 'static/uploads/1559AE33-326E-4C9F-845E-299F62F2676F.jpeg'

# plt.imshow(img[:, :, ::-1]);
# plt.show()
#
# imgs = []
# for i, model_dict in enumerate(models):
#     # plt.sca(ax[i])
#     # initialize the detector
#     model = init_detector(model_dict['config'], model_dict['checkpoint'], device='cuda:0')
#
#     start_time = time.time()
#     # Use the detector to do inference
#     result = inference_detector(model, img[:, :, ::-1])
#
#     # Let's plot the result
#     # show_result_pyplot(model, img, result, score_thr=0.3, title=model_dict['name'])
#     # imgs.append(model.show_result(img[:,:,::-1], result, score_thr=0.3, show=False))
#
#     models[i]['result'] = model.show_result(img[:, :, ::-1], result, score_thr=0.3, show=False)
#     models[i]['time'] = time.time() - start_time
#
# plt.figure(figsize=(20, 5))
# for i, model_dict in enumerate(models):
#     plt.subplot(1, 3, i + 1)
#     plt.imshow(model_dict['result'])
# plt.show()
#

class Detector(object):
    def __init__(self, model_IDs: list, path_images: str):
        MODEL_CONFIG_FILES: str = 'app/model_config/'
        # MODEL_CONFIG_FILES: str = 'model_config/'

        self.model_IDs: list = model_IDs
        if len(self.model_IDs) == 0:
            self.model_IDs = [0]

        self.path_images: str = path_images

        if torch.cuda.is_available():
            self.DEVICE = torch.device('cuda')
        else:
            self.DEVICE = torch.device('cpu')
        self.device_properties = torch.cuda.get_device_properties(self.DEVICE)

        self.models: list = []

        # if 0 in self.model_IDs:
        #     model_dict = dict(name='mask_rcnn', config=MODEL_CONFIG_FILES + 'ms_rcnn_x101_64x4d_fpn_1x_coco.py',
        #                       checkpoint=MODEL_CONFIG_FILES + 'ms_rcnn_x101_64x4d_fpn_1x_coco_20200206-86ba88d2.pth')
        #     self.model_SEG = init_detector(model_dict['config'], model_dict['checkpoint'], device='cuda:0')
        #     model_dict['model'] = self.model_SEG
        #     self.models.append(model_dict)

        # if 1 in self.model_IDs:
        #     model_dict = dict(name='faster_rcnn_x101', config=MODEL_CONFIG_FILES + 'faster_rcnn_x101_64x4d_fpn_1x_coco.py',
        #                       checkpoint=MODEL_CONFIG_FILES + 'faster_rcnn_x101_64x4d_fpn_1x_coco_20200204-833ee192.pth')
        #     model_dict['model'] = init_detector(model_dict['config'], model_dict['checkpoint'], device='cuda:0')
        #     self.models.append(model_dict)

        if 2 in self.model_IDs:
            model_dict = dict(name='yolo3', config=MODEL_CONFIG_FILES + 'yolov3_mobilenetv2_320_300e_coco.py',
                              checkpoint=MODEL_CONFIG_FILES + 'yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth')
            self.model_YOLO3 = init_detector(model_dict['config'], model_dict['checkpoint'], device='cuda:0')
            model_dict['model'] = self.model_YOLO3
            self.models.append(model_dict)
        self.score_threshold = 0.3

        if 3 in self.model_IDs:
            model_dict = dict(name='faster_rcnn_r50', config=MODEL_CONFIG_FILES + 'faster_rcnn_r50_fpn_tnr-pretrain_1x_coco.py',
                              checkpoint=MODEL_CONFIG_FILES + 'faster_rcnn_r50_fpn_tnr-pretrain_1x_coco_20220320_085147-efedfda4.pth')
            model_dict['model'] = init_detector(model_dict['config'], model_dict['checkpoint'], device='cuda:0')
            self.models.append(model_dict)

        if 4 in self.model_IDs:
            model_dict = dict(name='mask_rcnn_r50', config=MODEL_CONFIG_FILES + 'mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py',
                              checkpoint=MODEL_CONFIG_FILES + 'mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth ')
            self.model_SEG = init_detector(model_dict['config'], model_dict['checkpoint'], device='cuda:0')
            model_dict['model'] = self.model_SEG
            self.models.append(model_dict)

        # if 5 in self.model_IDs:
        #     model_dict = dict(name='YOLO8', config='', checkpoint='')
        #     self.model_YOLO8 = model = YOLO('yolov8n.pt')
        #     model_dict['model'] = self.model_YOLO8
        #     self.models.append(model_dict)






    def transform(self, img):
        ratio = 1.0
        if (img.shape[1] > 1280) or (img.shape[0] > 1280):
            if (img.shape[1] > img.shape[0]):
                ratio = 1280 / img.shape[1]
            else:
                ratio = 1280 / img.shape[0]
        # new_width = 500
        # ratio = new_width / img.shape[1]
        img = mmcv.imresize(img, (np.int32(ratio * img.shape[1]), np.int32(ratio * img.shape[0])), return_scale=False)

        return img

    def count_classes(self, model, result):
        if type(result) == list:
            bbox_result = result
        else:
            # if instance segmentation
            bbox_result, segm_result = result

        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]

        labels = np.concatenate(labels)
        bboxes = np.vstack(bbox_result)

        # Номера боксов c уверенностью выше
        labels_impt = np.where(bboxes[:, -1] > self.score_threshold)[0]

        # Уверенность для этих номеров
        confidence_imp = bboxes[bboxes[:, -1] > self.score_threshold][:, 4]

        # Коды классов для этих номеров
        labels_impt_list = [labels[i] for i in labels_impt]

        classes_names = model.CLASSES

        # Названия классов для номеров боксов
        labels_class = [classes_names[i] for i in labels_impt_list]

        result_dict = Counter(labels_class)

        return result_dict


    def detect_YOLO3(self, img):
        result = inference_detector(self.model_YOLO3, img)
        img = self.model_YOLO3.show_result(img, result, score_thr=self.score_threshold, show=False)

        return img

    def detect_SEG(self, img):
        result = inference_detector(self.model_SEG, img)
        img = self.model_SEG.show_result(img, result, score_thr=self.score_threshold, show=False)

        return img




    def get_result(self, image_name):
        fig = plt.figure()

        img = mmcv.imread(image_name)
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

            self.models[i]['count_classes'] = self.count_classes(model, result)

            result_name = f'result{i}.jpg'
            self.models[i]['result_name'] = result_name
            self.models[i]['score_threshold'] = self.score_threshold
            self.models[i]['time'] = f'{work_time:.4f}'

            model.show_result(img, result, score_thr=self.score_threshold, show=False, out_file=self.path_images + result_name)

            plt.close(fig)

            # plt.imshow(model.show_result(img[:, :, ::-1], result, score_thr=0.3, show=False, out_file = result_name))
            # plt.show()

    # >>> import os
    # >>> base=os.path.basename('/root/dir/sub/file.ext')
    # >>> base
    # 'file.ext'
    # >>> os.path.splitext(base)

    if __name__ == '__main__':
        detector = Detector(model_IDs=[0, 1, 2], path_images='F:/python/web-detector/app/static/uploads/')
        detector.get_result(r'F:/python/web-detector/app/static/uploads/demo.jpg')
