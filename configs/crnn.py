# ========================
# data
dataset_type = 'RecDataset'
train_root = [
    '../data/rec/mtwi2018/train',
    '../data/rec/icdar2019_lsvt/train',
    '../data/rec/icdar2019_rects/train',
    # '../data/rec/YCG09/train_images',
    # '../data/rec/YCG09/train_syn_images',
    # '../data/rec/YCG09/test_syn_images',
    # '../data/rec/ccpd2019/train',
    # '../data/rec/ccpd2020/train',
    # '../data/rec/document/train',
    # '../data/rec/document/val',
    # '../data/rec/document/test',
    # '../data/rec/ctw/train',
    # '../data/rec/ctw/val',
    # '../data/rec/synth/icdar2019_lsvt_weak',
]
val_root = '../data/rec/mtwi2018/test',

character_dict_path = './lightningOCR/common/rec_keys.txt'
fontfile = './lightningOCR/common/Arial.Unicode.ttf'

img_norm_cfg = dict(
    mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

train_pipeline = {'transforms':[
                        dict(type='CTCLabelEncode',
                             max_text_length=25,
                             min_text_length=3,
                             character_dict_path=character_dict_path,
                             character_type='ch',
                             use_space_char=True),
                        dict(type='RecTIA', p=0.5),
                        dict(type='Flipud', p=0.02),
                        dict(type='ColorJitter', brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
                        # dict(type='GaussianBlur', blur_limit=(5, 5), sigma_limit=1, p=0.5),  # cost too much time
                        dict(type='GaussNoise', var_limit=(10.0, 50.0), p=0.5),
                        dict(type='Reverse', p=0.5),
                        dict(type='Normalize', **img_norm_cfg),
                        dict(type='TextLineResize', height=32, width=320, p=1)]}

val_pipeline  =  {'transforms':[
                        dict(type='CTCLabelEncode',
                             max_text_length=25,
                             character_dict_path=character_dict_path,
                             character_type='ch',
                             use_space_char=True),
                        dict(type='Normalize', **img_norm_cfg),
                        dict(type='TextLineResize', height=32, width=320, p=1)]}

test_pipeline =  {'transforms':[ 
                        dict(type='Normalize', **img_norm_cfg)]}


postprocess = dict(type='CTCLabelDecode',
                   character_dict_path=character_dict_path,
                   character_type='ch',
                   use_space_char=True)


data = dict(
    pin_memory=True,
    train=dict(
        type=dataset_type,
        data_root=train_root,
        character_dict_path=character_dict_path,
        pipeline=train_pipeline,
        length=None,
        fontfile=fontfile,
        postprocess=postprocess),
    val=dict(
        type=dataset_type,
        data_root=val_root,
        character_dict_path=character_dict_path,
        pipeline=val_pipeline,
        length=None,
        fontfile=fontfile),
    test=dict(
        type=dataset_type,
        data_root=val_root,
        character_dict_path=character_dict_path,
        pipeline=test_pipeline))


# ========================
# strategy
strategy = dict(
    warmup_epochs=3,
    lrf=0.1,
    momentum=0.937,
    weight_decay=0.0005,
    cos_lr=True
)


# ========================
# model
model = dict(
    type='Recognizer',
    architecture=dict(type='CRNN', return_feats=True),
    loss=dict(type='CombinedLoss',
              loss_ctc=dict(type='CTCLoss', use_focal_loss=True, weight=1.0),
              loss_ace=dict(type='ACELoss', weight=1.0),
              loss_center=dict(type='CenterLoss', center_file_path='center.pth', weight=0.05)),
    # loss=dict(type='CTCLoss'),
    strategy=strategy,
    data=data,
    metric=dict(type='RecF1'),  # RecF1 or RecAcc
    postprocess=postprocess
)


# ========================
# callbacks

ckpt = dict(monitor='acc/val' if model['metric']['type'] == 'RecAcc' else 'f1/val', mode='max')
