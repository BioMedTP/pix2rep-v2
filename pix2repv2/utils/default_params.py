from pix2repv2.utils import utils

cfg = utils.Config(config_name="config3D").cfg

BODY_THRESHOLD_HU = cfg.data.BODY_THRESHOLD_HU
SPACING = tuple(cfg.data.SPACING)
WINDOW_HU = tuple(cfg.data.WINDOW_HU)
PATCH_SIZE = tuple(cfg.data.PATCH_SIZE)
