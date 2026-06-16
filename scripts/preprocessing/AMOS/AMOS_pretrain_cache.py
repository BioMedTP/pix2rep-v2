from pix2repv2.data.amos import AMOSPatchSSL, build_amos_pretraining_pipeline
from pix2repv2.utils.utils import Config
from torch.utils.data import DataLoader

cfg = Config(config_name="config3D").cfg

pipeline = build_amos_pretraining_pipeline()
# pipeline = pipeline._drop_cache()
ds = AMOSPatchSSL(cfg=cfg, pipeline=pipeline)

dataloader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=10)

# Warmup the cache
for _ in range(len(ds)):
    batch = ds[_]
    print(batch[0].shape)
    print(batch[1].shape)
    print("===============")
