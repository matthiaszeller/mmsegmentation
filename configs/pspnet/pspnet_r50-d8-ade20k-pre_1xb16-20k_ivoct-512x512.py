_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/ivoct.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    # pretrained: download manually and use --cfg-options model.init_cfg.checkpoint="..."
    # NOTE: here we use the full pretrained mode, with decoding heads
    pretrained=None,
    init_cfg=dict(type='Pretrained', checkpoint=None),
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2)
)


# Default 20k schedule optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

# Add linear warmup
_warmup_iter = 180
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=_warmup_iter),
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=_warmup_iter,
        end=20000,
        by_epoch=False,
    )
]

# training schedule for 20k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
