import math
import sys
from collections import OrderedDict

import numpy as np
from loguru import logger

from carbs import CARBS
from carbs import CARBSParams
from carbs import LogSpace
from carbs import LogitSpace
from carbs import ObservationInParam
from carbs import ParamDictType
from carbs import Param

logger.remove()
logger.add(sys.stdout, level="DEBUG", format="{message}")



def run_test_fn(input_in_param: ParamDictType):
    # A noisy function minimized at lr=1e-3, max hidden_dim
    result = (math.log10(input_in_param["learning_rate"]) + 3) ** 2 * 512 / input_in_param[
        "epochs"
    ] + np.random.uniform() * 0.1
    return result

param_spaces = [
    Param(name="learning_rate", space=LogSpace(scale=0.5), search_center=1e-4),
    Param(name="momentum", space=LogitSpace(), search_center=0.9),
    Param(name="epochs", space=LogSpace(is_integer=True, min=2, max=512), search_center=10),
]

carbs_params = CARBSParams(
    better_direction_sign=-1,
    is_wandb_logging_enabled=False,
    resample_frequency=0,
)
carbs = CARBS(carbs_params, param_spaces, db_path="observations.db")
for i in range(4):
    suggestion = carbs.suggest().suggestion
    observed_value = run_test_fn(suggestion)
    obs_out = carbs.observe(ObservationInParam(input=suggestion, output=observed_value, cost=suggestion["epochs"]))
    logger.info(f"Observation {obs_out.logs['observation_count']}")
    logger.info(
        f"Observed lr={obs_out.logs['observation/learning_rate']:.2e}, "
        f"epochs={obs_out.logs['observation/epochs']}, "
        f"output {obs_out.logs['observation/output']:.3f}"
    )
    logger.info(
        f"Best lr={obs_out.logs['best_observation/learning_rate']:.2e}, "
        f"epochs={obs_out.logs['best_observation/epochs']}, "
        f"output {obs_out.logs['best_observation/output']:.3f}"
    )

carbs_reloaded = CARBS.load_from_db(carbs_params, param_spaces, db_path="observations.db")
logger.info(f"Reloaded from DB. Observations so far: {carbs_reloaded.observation_count}")