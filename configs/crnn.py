# ========================
# data
dataset_type = 'RecDataset'
data_root = '../data/icdar2019_lsvt/train/rec'
img_norm_cfg = dict(
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

train_pipeline = {'transforms':[
                        dict(type='CTCLabelEncode',
                             max_text_length=25,
                             character_dict_path='./lightningOCR/common/rec_keys.txt',
                             character_type='ch',
                             use_space_char=False),
                        dict(type='TextLineResize', height=32, width=320, p=1),
                        dict(type='Normalize', **img_norm_cfg)]}

val_pipeline  =  {'transforms':[
                        dict(type='CTCLabelEncode',
                             max_text_length=25,
                             character_dict_path='./lightningOCR/common/rec_keys.txt',
                             character_type='ch',
                             use_space_char=False),
                        dict(type='TextLineResize', height=32, width=320, p=1),
                        dict(type='Normalize', **img_norm_cfg)]}

test_pipeline =  {'transforms':[ 
                        dict(type='Normalize', **img_norm_cfg)]}

data = dict(
    pin_memory=True,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=train_pipeline,
        length=None),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=val_pipeline,
        length=None),
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
    data_cfg=data
)