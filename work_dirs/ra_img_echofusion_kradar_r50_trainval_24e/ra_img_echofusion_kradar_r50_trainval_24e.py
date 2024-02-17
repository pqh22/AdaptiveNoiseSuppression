dataset_type = 'KRadarDataset'
data_root = 'data/k-radar/'
class_names = ['Sedan']
point_cloud_range = [0, -6.4, -2.0, 72.0, 6.4, 6.0]
input_modality = dict(use_lidar=False, use_camera=True, use_radar=True)
db_sampler = dict(
    data_root='data/kitti/',
    info_path='data/kitti/kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(filter_by_difficulty=[-1], filter_by_min_points=dict(Car=5)),
    classes=['Car'],
    sample_groups=dict(Car=15))
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        type='LoadKRadarsFromFile',
        modality_type='RA',
        coord_type='LIDAR',
        convert2polar=False,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[0, -6.4, -2.0, 72.0, 6.4, 6.0]),
    dict(type='ObjectNameFilter', classes=['Sedan']),
    dict(
        type='NormalizeMultiviewImage',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=['Sedan']),
    dict(
        type='Collect3D',
        keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'radars_ra'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                   'cam2lidar', 'cam_intrinsic', 'Trv2c', 'depth2img',
                   'cam2img', 'pad_shape', 'scale_factor', 'flip',
                   'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
                   'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
                   'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                   'transformation_3d_flow'))
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadKRadarsFromFile',
        modality_type='RA',
        coord_type='LIDAR',
        file_client_args=dict(backend='disk')),
    dict(
        type='NormalizeMultiviewImage',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=['Sedan'],
                with_label=False),
            dict(
                type='Collect3D',
                keys=['img', 'radars_ra'],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                           'cam2lidar', 'cam_intrinsic', 'Trv2c', 'depth2img',
                           'cam2img', 'pad_shape', 'scale_factor', 'flip',
                           'pcd_horizontal_flip', 'pcd_vertical_flip',
                           'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                           'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                           'pcd_rotation', 'pts_filename',
                           'transformation_3d_flow'))
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=dict(backend='disk')),
    dict(type='DefaultFormatBundle3D', class_names=['Car'], with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type='KRadarDataset',
            data_root='data/k-radar/',
            ann_file='data/k-radar/kradar_infos_trainval.pkl',
            split='',
            pts_prefix='os2-64',
            pipeline=[
                dict(type='LoadMultiViewImageFromFiles', to_float32=True),
                dict(type='PhotoMetricDistortionMultiViewImage'),
                dict(
                    type='LoadKRadarsFromFile',
                    modality_type='RA',
                    coord_type='LIDAR',
                    convert2polar=False,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                    with_attr_label=False),
                dict(
                    type='ObjectRangeFilter',
                    point_cloud_range=[0, -6.4, -2.0, 72.0, 6.4, 6.0]),
                dict(type='ObjectNameFilter', classes=['Sedan']),
                dict(
                    type='NormalizeMultiviewImage',
                    mean=[103.53, 116.28, 123.675],
                    std=[1.0, 1.0, 1.0],
                    to_rgb=False),
                dict(type='PadMultiViewImage', size_divisor=32),
                dict(type='DefaultFormatBundle3D', class_names=['Sedan']),
                dict(
                    type='Collect3D',
                    keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'radars_ra'],
                    meta_keys=('filename', 'ori_shape', 'img_shape',
                               'lidar2img', 'cam2lidar', 'cam_intrinsic',
                               'Trv2c', 'depth2img', 'cam2img', 'pad_shape',
                               'scale_factor', 'flip', 'pcd_horizontal_flip',
                               'pcd_vertical_flip', 'box_mode_3d',
                               'box_type_3d', 'img_norm_cfg', 'pcd_trans',
                               'sample_idx', 'pcd_scale_factor',
                               'pcd_rotation', 'pts_filename',
                               'transformation_3d_flow'))
            ],
            modality=dict(use_lidar=False, use_camera=True, use_radar=True),
            classes=['Sedan'],
            test_mode=False,
            box_type_3d='LiDAR',
            radar_prefix='radar_ra')),
    val=dict(
        type='KRadarDataset',
        data_root='data/k-radar/',
        ann_file='data/k-radar/kradar_infos_val.pkl',
        split='',
        pts_prefix='lidars',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='LoadKRadarsFromFile',
                modality_type='RA',
                coord_type='LIDAR',
                file_client_args=dict(backend='disk')),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=['Sedan'],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['img', 'radars_ra'],
                        meta_keys=('filename', 'ori_shape', 'img_shape',
                                   'lidar2img', 'cam2lidar', 'cam_intrinsic',
                                   'Trv2c', 'depth2img', 'cam2img',
                                   'pad_shape', 'scale_factor', 'flip',
                                   'pcd_horizontal_flip', 'pcd_vertical_flip',
                                   'box_mode_3d', 'box_type_3d',
                                   'img_norm_cfg', 'pcd_trans', 'sample_idx',
                                   'pcd_scale_factor', 'pcd_rotation',
                                   'pts_filename', 'transformation_3d_flow'))
                ])
        ],
        modality=dict(use_lidar=False, use_camera=True, use_radar=True),
        classes=['Sedan'],
        test_mode=True,
        box_type_3d='LiDAR',
        radar_prefix='radar_ra'),
    test=dict(
        type='KRadarDataset',
        data_root='data/k-radar/',
        ann_file='data/k-radar/kradar_infos_test.pkl',
        split='',
        pts_prefix='lidars',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='LoadKRadarsFromFile',
                modality_type='RA',
                coord_type='LIDAR',
                file_client_args=dict(backend='disk')),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=['Sedan'],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['img', 'radars_ra'],
                        meta_keys=('filename', 'ori_shape', 'img_shape',
                                   'lidar2img', 'cam2lidar', 'cam_intrinsic',
                                   'Trv2c', 'depth2img', 'cam2img',
                                   'pad_shape', 'scale_factor', 'flip',
                                   'pcd_horizontal_flip', 'pcd_vertical_flip',
                                   'box_mode_3d', 'box_type_3d',
                                   'img_norm_cfg', 'pcd_trans', 'sample_idx',
                                   'pcd_scale_factor', 'pcd_rotation',
                                   'pts_filename', 'transformation_3d_flow'))
                ])
        ],
        modality=dict(use_lidar=False, use_camera=True, use_radar=True),
        classes=['Sedan'],
        test_mode=True,
        box_type_3d='LiDAR',
        radar_prefix='radar_ra'))
evaluation = dict(
    interval=1,
    pipeline=[
        dict(type='LoadMultiViewImageFromFiles', to_float32=True),
        dict(
            type='LoadKRadarsFromFile',
            modality_type='RA',
            coord_type='LIDAR',
            file_client_args=dict(backend='disk')),
        dict(
            type='NormalizeMultiviewImage',
            mean=[103.53, 116.28, 123.675],
            std=[1.0, 1.0, 1.0],
            to_rgb=False),
        dict(type='PadMultiViewImage', size_divisor=32),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1333, 800),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=['Sedan'],
                    with_label=False),
                dict(
                    type='Collect3D',
                    keys=['img', 'radars_ra'],
                    meta_keys=('filename', 'ori_shape', 'img_shape',
                               'lidar2img', 'cam2lidar', 'cam_intrinsic',
                               'Trv2c', 'depth2img', 'cam2img', 'pad_shape',
                               'scale_factor', 'flip', 'pcd_horizontal_flip',
                               'pcd_vertical_flip', 'box_mode_3d',
                               'box_type_3d', 'img_norm_cfg', 'pcd_trans',
                               'sample_idx', 'pcd_scale_factor',
                               'pcd_rotation', 'pts_filename',
                               'transformation_3d_flow'))
            ])
    ])
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/ra_img_echofusion_kradar_r50_trainval_24e/'
load_from = None
resume_from = None
workflow = [('train', 1)]
pi = 3.141592653589793
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
polar_pcd_range = [0, -0.3490658503988659, -2.0, 72.0, 0.3490658503988659, 6.0]
polar_voxel_size = [0.375, 0.01090830782496456, 0.4]
radius_range = [0.0, 72.0, 0.375]
grid_res = 0.4
voxel_size = [0.4, 0.4, 0.4]
output_size = [256, 64, 10]
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
model = dict(
    type='EchoFusion_RA_IMG',
    use_grid_mask=True,
    pretrained=dict(img='ckpts/resnet50_bev.pth'),
    pts_backbone=dict(
        type='RadarResNet',
        stem_stride=1,
        depth=50,
        in_channels=20,
        num_stages=4,
        strides=(1, 1, 2, 2),
        out_indices=(1, 2, 3),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch'),
    pts_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        num_outs=3,
        norm_cfg=dict(type='SyncBN', eps=0.001, momentum=0.01)),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch'),
    img_neck=dict(
        type='FPN_KRADAR',
        num_encoder=0,
        num_decoder=3,
        num_levels=3,
        polar_range=[
            0, -0.3490658503988659, -2.0, 72.0, 0.3490658503988659, 6.0
        ],
        radius_range=[0.0, 72.0, 0.375],
        use_different_res=True,
        use_bev_aug=False,
        output_multi_scale=True,
        grid_res=0.4,
        pc_range=[0, -6.4, -2.0, 72.0, 6.4, 6.0],
        output_size=[256, 64, 10],
        fpn_cfg=dict(
            in_channels=[512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_output',
            num_outs=3,
            relu_before_extra_convs=True,
            norm_cfg=dict(type='SyncBN', requires_grad=True))),
    pts_bbox_head=dict(
        type='EchoFusionHead',
        num_query=30,
        num_classes=1,
        in_channels=256,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        polar_range=[
            0, -0.3490658503988659, -2.0, 72.0, 0.3490658503988659, 6.0
        ],
        radius_range=[0.0, 72.0, 0.375],
        code_size=8,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='PolarTransformer',
            num_feature_levels=3,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_levels=3),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='PolarTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                pc_range=[0, -6.4, -2.0, 72.0, 6.4, 6.0],
                radius_range=[0.0, 72.0, 0.375],
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            num_levels=3)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[0, -6.4, -2.0, 72.0, 6.4, 6.0],
            pc_range=[0, -6.4, -2.0, 72.0, 6.4, 6.0],
            max_num=10,
            voxel_size=[0.4, 0.4, 0.4],
            score_threshold=0.3,
            num_classes=1),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=[0.4, 0.4, 0.4],
            point_cloud_range=[0, -6.4, -2.0, 72.0, 6.4, 6.0],
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=[0, -6.4, -2.0, 72.0, 6.4, 6.0]))))
optimizer = dict(
    type='AdamW',
    lr=5e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            img_backbone=dict(lr_mult=0.1),
            sampling_offsets=dict(lr_mult=0.1),
            reference_points=dict(lr_mult=0.1))),
    weight_decay=0.075)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001)
total_epochs = 24
runner = dict(type='EpochBasedRunner', max_epochs=24)
gpu_ids = range(0, 1)
