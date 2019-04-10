# encoding: utf-8
"""
@author: xyliao
@contact: xyliao1993@qq.com
"""

import requests
import argparse
import cv2
import numpy as np
import os

# Initialize the PyTorch REST API endpoint URL.
PyTorch_REST_API_URL = 'http://172.20.62.233:5000/predict'


def vis_detections(im, dets):
    """Visual debugging of detections."""
    cnt = 0
    for i in range(np.minimum(500, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        cnt += 1
        cv2.circle(im, (int((bbox[2]+bbox[0])/2), int((bbox[3]+bbox[1])/2)), 12, (0, 0, 255), -1)

    cv2.putText(im, str(cnt), (40, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 204, 0), 2)
    return im


def predict_result(image_path):
    # Initialize image path
    image = open(image_path, 'rb').read()
    payload = {'image': image}

    # Submit the request.
    r = requests.post(PyTorch_REST_API_URL, files=payload).json()

    # Ensure the request was successful.
    if r['success']:
        det_box = []
        for (i, result) in enumerate(r['predictions']):
            box = []
            if i > 0:
                [box.append(int(j)) for j in result["BoxList"]]
            det_box.append(box)
        del[det_box[0]]
        im2show = vis_detections(cv2.imread(image_path), np.array(det_box))
        result_path = os.path.join(image_path[:-4] + "_det.jpg")
        cv2.imwrite(result_path, im2show)
        # cv2.imwrite('./result.jpg', im2show)
        # cv2.imshow('imshow', im2show)

        cv2.destroyAllWindows()
    else:
        print('Request failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification demo')
    parser.add_argument('--file', type=str, help='test image file')

    args = parser.parse_args()
    predict_result(args.file)
