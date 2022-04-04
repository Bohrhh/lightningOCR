# ========================
# data
dataset_type = 'ClsDataset'
data_root = '../data/document/train/rec'
img_norm_cfg = dict(
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

train_pipeline = {'transforms':[
                      dict(type='ClsRotate180', p=0.5),
                      dict(type='TextLineResize', height=48, width=192, p=1),
                      dict(type='Normalize', **img_norm_cfg)]}

val_pipeline =  {'transforms':[
                      dict(type='TextLineResize', height=48, width=192, p=1),
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
    type='Classifier',
    architecture=dict(type='resnet18', pretrained=True, num_classes=2),
    loss_cfg=dict(type='CrossEntropyLoss'),
    strategy=strategy,
    data_cfg=data,
    metric_cfg=dict(type='ClsAcc')
)


# ========================
# callbacks

ckpt = dict(monitor='acc/val', mode='max')