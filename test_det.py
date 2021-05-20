import sys
sys.path.append('Detection')

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import cv2

config_file = 'Detection/configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py'

checkpoint_file = '../Data/mask_rcnn_swin_tiny_patch4_window7.pth'

device = 'cuda:0'

model = init_detector(config_file, checkpoint_file, device=device)

# show_result_pyplot(model, 'test.jpg',  inference_detector(model, 'test.jpg'), score_thr=0.3, title='result', wait_time=0)

result = inference_detector(model, 'test.jpg')

print(result[0][0])
# bbox = []
# for r in result[0][0]:
# 	if r[4] > 0.5:
# 		bbox.append(r[0:4].astype(int))	

# img = cv2.imread('test.jpg')

# print(len(bbox))

# for bb in bbox:
# 	img = cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (255, 0, 0), 2)

# cv2.imshow('img', img)
# cv2.waitKey(0)