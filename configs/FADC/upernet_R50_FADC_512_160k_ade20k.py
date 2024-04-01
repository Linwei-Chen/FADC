# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


_base_ = [
    '../_base_/models/upernet_r50.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
model = dict(
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        with_cp=False,
        dcn=dict( 
            type='AdaDilatedConv',
            offset_freq=None,
            epsilon=1e-4,
            use_zero_dilation=False,
            # offset_freq='SLP_res',
            deformable_groups=1, 
            padding_mode='repeat',
            kernel_decompose='both',
            # kernel_decompose=None,
            # pre_fs=False,
            pre_fs=True,
            # conv_type='multifreqband',
            conv_type='conv',
            # fs_cfg=None,
            fs_cfg={
                'k_list':[2,4,8],
                'fs_feat':'feat',
                'lowfreq_att':False,
                'lp_type':'freq',
                # 'lp_type':'laplacian',
                'act':'sigmoid',
                'spatial':'conv',
                'spatial_group':1,
            },
            sp_att=False,
            fallback_on_stride=False),
        # dcn=dict( #在最后三个block加入可变形卷积 
        stage_with_dcn=(False, True, True, True),
        # stage_with_dcn=(False, False, False, False),
    ),
    decode_head=dict(
        num_classes=150), 
    auxiliary_head=dict(num_classes=150),
    test_cfg = dict(mode='slide', crop_size=crop_size, stride=(341, 341)),
)

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=4, workers_per_gpu=4,)

optimizer = dict(constructor='LearningRateDecayOptimizerConstructorHorNet', _delete_=True, type='AdamW', 
                 lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg={'decay_rate': 0.9,
                                'decay_type': 'stage_wise',
                                'num_layers': 12})

checkpoint_config = dict(max_keep_ckpts=2)
evaluation = dict(save_best='mIoU')

optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
runner = dict(type='IterBasedRunner')

# do not use mmdet version fp16
fp16 = None
# optimizer_config = dict(
#     type="DistOptimizerHook",
#     update_interval=1,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=True,
# )
