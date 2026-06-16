from pix2repv2.data.amos import AMOSPatchSupervised, build_amos_finetuning_pipeline
from pix2repv2.utils.utils import Config

cfg = Config(config_name="config3D").cfg

pipeline = build_amos_finetuning_pipeline()
pipeline = pipeline._drop_cache()
ds = AMOSPatchSupervised(cfg=cfg, pipeline=pipeline)

# Warmup the cache
for _ in range(len(ds)):
    batch = ds[_]
