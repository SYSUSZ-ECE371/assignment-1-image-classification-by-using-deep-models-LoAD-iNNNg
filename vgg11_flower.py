auto_scale_lr = dict(base_batch_size=256)
bgr_mean = [
    103.53,
    116.28,
    123.675,
]
bgr_std = [
    57.375,
    57.12,
    58.395,
]
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    num_classes=5,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
dataset_type = 'ImageNet'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='VisualizationHook'))
default_scope = 'mmcls'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth'
log_level = 'INFO'
model = dict(
    backbone=dict(
        depth=11,
        init_cfg=dict(
            checkpoint=
            'D:/model_cache/hub/checkpoints/vgg11_batch256_imagenet_20210208-4271cd6c.pth',
            prefix='backbone',
            type='Pretrained'),
        num_classes=5,
        type='VGG'),
    head=dict(
        loss=dict(loss_weight=1.0, type='CrossEntropyLoss'),
        topk=(1, ),
        type='ClsHead'),
    neck=None,
    type='ImageClassifier')
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0001))
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=5,
        start_factor=0.001,
        type='LinearLR'),
    dict(
        T_max=295,
        begin=5,
        by_epoch=True,
        end=100,
        eta_min=1e-06,
        type='CosineAnnealingLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=100,
        eta_min=1e-05,
        param_name='weight_decay',
        type='CosineAnnealingParamScheduler'),
]
randomness = dict(deterministic=False, seed=None)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=64,
    dataset=dict(
        ann_file='val.txt',
        classes='data/flower_dataset/classes.txt',
        data_prefix='val',
        data_root='data/flower_dataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=224, type='Resize'),
            dict(type='PackClsInputs'),
        ],
        type='ImageNet'),
    num_workers=4,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(topk=(1, ), type='Accuracy')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(scale=224, type='Resize'),
    dict(type='PackClsInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
train_dataloader = dict(
    batch_size=64,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='train.txt',
        classes='data/flower_dataset/classes.txt',
        data_prefix='train',
        data_root='data/flower_dataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                interpolation='bicubic',
                scale=224,
                type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(
                hparams=dict(pad_val=[
                    104,
                    116,
                    124,
                ]),
                policies='imagenet',
                type='AutoAugment'),
            dict(
                erase_prob=0.15,
                fill_color=[
                    103.53,
                    116.28,
                    123.675,
                ],
                fill_std=[
                    57.375,
                    57.12,
                    58.395,
                ],
                max_area_ratio=0.3333333333333333,
                min_area_ratio=0.02,
                mode='rand',
                type='RandomErasing'),
            dict(type='PackClsInputs'),
        ],
        type='ImageNet'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        backend='pillow',
        interpolation='bicubic',
        scale=224,
        type='RandomResizedCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(
        hparams=dict(pad_val=[
            104,
            116,
            124,
        ]),
        policies='imagenet',
        type='AutoAugment'),
    dict(
        erase_prob=0.15,
        fill_color=[
            103.53,
            116.28,
            123.675,
        ],
        fill_std=[
            57.375,
            57.12,
            58.395,
        ],
        max_area_ratio=0.3333333333333333,
        min_area_ratio=0.02,
        mode='rand',
        type='RandomErasing'),
    dict(type='PackClsInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=64,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='val.txt',
        classes='data/flower_dataset/classes.txt',
        data_prefix='val',
        data_root='data/flower_dataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=224, type='Resize'),
            dict(type='PackClsInputs'),
        ],
        type='ImageNet'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(topk=(1, ), type='Accuracy')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ClsVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'work_dir/vgg11_flower'
