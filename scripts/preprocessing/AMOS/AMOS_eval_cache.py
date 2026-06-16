from pix2repv2.data.amos import AMOSSupervisedEval, build_amos_eval_pipeline
from pix2repv2.utils.utils import Config

cfg = Config(config_name="config3D").cfg

pipeline = build_amos_eval_pipeline()
# pipeline = pipeline._drop_cache()
ds = AMOSSupervisedEval(cfg=cfg, pipeline=pipeline)

# Warmup the cache
for _ in range(len(ds)):
    batch = ds[_]
