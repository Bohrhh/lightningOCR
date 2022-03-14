# ========================
# data
dataset_type = 'ClsDataset'
data_root = '../data/rec/icdar2019_lsvt'
img_norm_cfg = dict(
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

train_pipeline = {'transforms':[
                      dict(type='ClsRotate180', p=0.5),
                      dict(type='ClsResize', height=48, width=192, p=1),
                      dict(type='Normalize', **img_norm_cfg)]}

val_pipeline =  {'transforms':[
                      dict(type='ClsResize', height=48, width=192, p=1),
                      dict(type='Normalize', **img_norm_cfg)]}

test_pipeline =  {'transforms':[ 
                      dict(type='Normalize', **img_norm_cfg)]}

data = dict(
    batch_size_per_gpu=128,
    workers_per_gpu=16,
    pin_memory=True,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=train_pipeline),
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
    gpus=1,
    epochs=10,
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
)