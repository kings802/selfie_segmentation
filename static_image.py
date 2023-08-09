
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
# 图片人物抠图:
IMAGE_FILES = ["face_example_2.jpg"]
BG_COLOR = (0, 255, 0) # 背景颜色也可以使用其他的照片，要求与原照片尺寸一致
bg_image = cv2.imread('seasand3.jpg')
height, width ,_ = bg_image.shape
MASK_COLOR = (255, 255, 255) # mask图片颜色

with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        image = cv2.resize(image,(width,height))
        # 在处理之前需要转换图片到RGB颜色空间
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = selfie_segmentation.process(image)
        # 在背景图像上绘制分割图
        #为了改善边界周围的分割，可以考虑在 results.segmentation_mask进行双边过滤
        # np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1最后参数越小，包括的边缘越多
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.3
        #生成纯色图像,白色的mask图纸
        #fg_image = np.zeros(image.shape, dtype=np.uint8)
        #fg_image[:] = MASK_COLOR
        fg_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if bg_image is None:
            # 背景为纯色
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
        output_image = np.where(condition, fg_image, bg_image)
        cv2.imshow('output_image',output_image)
        cv2.waitKey(0)
        #cv2.imwrite('selfie0.png', output_image)