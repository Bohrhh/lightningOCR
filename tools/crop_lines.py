import os
import cv2
import json
import lmdb
import numpy as np
from tqdm import tqdm
from multiprocessing.pool import Pool


def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [
        points[index_1], points[index_2], points[index_3], points[index_4]
    ]
    return box


def get_rotate_crop_image(img, points):
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def get_img_txt(data_dir, k, v):
    print(k)
    s = ''
    for i in v:
        txt = i['transcription']
        coords = np.array(i['points']).reshape(-1)
        s += ','.join([str(i) for i in coords])+'\t'+txt+'\n'
    with open(os.path.join(data_dir, k+'.txt'), 'w') as f:
        f.write(s)


def multi_img_txt():
    with open('train_full_labels.json', 'r') as f:
        j = json.load(f)
    pool = Pool(48)
    for k,v in tqdm(j.items()):
        pool.apply_async(get_img_txt, ('train/labels', k, v))
    pool.close()
    pool.join()


def get_crop_img(data_dir, outimg_dir, outlabel_dir, k, v):
    print(k)
    img_path = os.path.join(data_dir, k+'.jpg')
    img = cv2.imread(os.path.join(data_dir, k+'.jpg'))
    height, width = img.shape[:2]
    crops = []
    txts = []
    for idx, i in enumerate(v):
        txt = i['transcription']
        poly = np.array(i['points'])
        if txt in ['#', '###']:
            continue
        box = get_mini_boxes(poly.astype(np.int32))
        box = np.array(box)
        box[:,0] = np.clip(box[:,0], 0, width)
        box[:,1] = np.clip(box[:,1], 0, height)
        crop = get_rotate_crop_image(img, box)
        crops.append(crop)
        txts.append(txt)
    return crops, txts
        # cv2.imwrite(os.path.join(outimg_dir, f'{k}_{idx:03d}.jpg'), crop)
        # with open(os.path.join(outlabel_dir, f'{k}_{idx:03d}.txt'), 'w') as f:
        #     f.write(txt)


def multi_crop_img():
    with open('train_full_labels.json', 'r') as f:
        j = json.load(f)

    pool = Pool(48)
    results = []
    for k,v in tqdm(j.items()):
        results.append(pool.apply_async(get_crop_img, ('train/images', 'rec/images', 'rec/labels', k, v)))

    pool.close()
    pool.join()

    env = lmdb.open('./rec', map_size=1099511627776)
    txn = env.begin(write=True)

    idx = 0
    for res in results:
        crops, txts = res.get()
        for crop, txt in zip(crops, txts):
            ret, crop = cv2.imencode('.jpg', crop)
            if not ret:
                continue
            txn.put(key=f'image-{idx:09d}'.encode(encoding='utf-8'), value=crop.tobytes())
            txn.put(key=f'label-{idx:09d}'.encode(encoding='utf-8'), value=txt.encode(encoding='utf-8'))
            idx += 1
    txn.put(key='num-samples'.encode(encoding='utf-8'), value=b'%d' % idx)
    txn.commit()
    env.close()


if __name__ == '__main__':
    # generate det labels
    # multi_img_txt()

    # generate crop text imgs for rec training
    multi_crop_img()