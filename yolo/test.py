import torch
from torch.utils.data import DataLoader
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.utils import DEFAULT_CFG, yaml_load
from ultralytics.yolo.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.yolo.data import build_yolo_dataset, build_dataloader

overrides = {'task': 'detect',
             'data': 'data.yaml',
             'imgsz': 640,
             'workers': 4
             }
cfg = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
data_info = check_det_dataset(cfg.data)

ds_train = build_yolo_dataset(cfg, img_path=data_info['train'], batch=cfg.batch,
                              data_info=data_info, mode='train', rect=False, stride=32)

ds_val = build_yolo_dataset(cfg, img_path=data_info['val'], batch=cfg.batch, data_info=data_info,
                            mode='val', rect=False, stride=32)

# dl_train = build_dataloader(ds_train,batch=cfg.batch,workers=0)
# dl_val = build_dataloader(ds_val,batch=cfg.batch,workers =0,shuffle=False)


dl_train = DataLoader(ds_train, batch_size=cfg.batch, num_workers=cfg.workers,
                      collate_fn=ds_train.collate_fn)

dl_val = DataLoader(ds_val, batch_size=cfg.batch, num_workers=cfg.workers,
                    collate_fn=ds_val.collate_fn)

for batch in dl_val:
    break

batch.keys()
