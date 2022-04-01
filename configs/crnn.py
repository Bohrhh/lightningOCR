# ========================
# data
dataset_type = 'RecDataset'
data_root = '../data/icdar2019_lsvt/train/rec'
character_dict_path = './lightningOCR/common/rec_keys.txt'
fontfile = './lightningOCR/common/Arial.Unicode.ttf'

img_norm_cfg = dict(
    mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

train_pipeline = {'transforms':[
                        dict(type='CTCLabelEncode',
                             max_text_length=25,
                             character_dict_path=character_dict_path,
                             character_type='ch',
                             use_space_char=True),
                        dict(type='RecTIA', p=0.8),
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

data = dict(
    pin_memory=True,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=train_pipeline,
        length=None,
        fontfile=fontfile),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=val_pipeline,
        length=None,
        fontfile=fontfile),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline))


# ========================
# strategy
strategy = dict(
    warmup_epochs=3,
    lr0=0.01,
    lrf=0.1,
    momentum=0.937,
    weight_decay=0.0005,
    cos_lr=True
)


# ========================
# model
model = dict(
    type='Recognizer',
    architecture=dict(type='CRNN'),
    loss_cfg=dict(type='CTCLoss'),
    strategy=strategy,
    data_cfg=data,
    metric_cfg=dict(type='RecF1'),
    postprocess_cfg=dict(
        type='CTCLabelDecode',
        character_dict_path=character_dict_path,
        character_type='ch',
        use_space_char=True)
)


# ========================
# callbacks

ckpt = dict(monitor='acc/val' if model['metric_cfg']['type'] == 'RecAcc' else 'f1/val', mode='max')