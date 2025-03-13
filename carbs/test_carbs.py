import os
from typing import List

import pytest
import wandb

from carbs import LogitSpace
from carbs import ObservationInParam
from carbs.carbs import CARBS
from carbs.utils import CARBSParams
from carbs.utils import LinearSpace
from carbs.utils import LogSpace
from carbs.utils import Param

# Set wandb to dryrun mode for testing
os.environ["WANDB_MODE"] = "dryrun"

# Initialize wandb
wandb.init(project="my_project", job_type="train")

# Initialize the database name
db_name = "test_observations.db"

@pytest.fixture
def carbs_config() -> CARBSParams:
    return CARBSParams(is_wandb_logging_enabled=False, is_saved_on_every_observation=False)


@pytest.fixture
def params() -> List[Param]:
    return [
        Param("p1", LogSpace(scale=1), 1e-2),
        Param("p2", LinearSpace(scale=2), 0),
        Param("p3", LogitSpace(scale=0.5), 0.5),
    ]

@pytest.fixture
def db_path() -> str:
    return db_name

@pytest.fixture
def carbs_instance(carbs_config: CARBSParams, params: List[Param], db_path: str) -> CARBS:
    return CARBS(carbs_config, params, db_path)


def test_suggest_one(carbs_instance: CARBS) -> None:
    start_suggestions = len(carbs_instance.outstanding_suggestions)
    suggestion = carbs_instance.suggest()
    assert len(carbs_instance.outstanding_suggestions) == start_suggestions + 1
    assert suggestion is not None
    # assert "suggestion_uuid" in suggestion.suggestion
    for param in carbs_instance.params:
        assert param.name in suggestion.suggestion

    os.remove(db_name)

def test_suggest_observe_loop(carbs_instance):
    initial_outstanding_suggestions = len(carbs_instance.outstanding_suggestions)
    initial_success_observations = len(carbs_instance.success_observations)

    num_iterations = 4
    for i in range(num_iterations):
        suggest_output = carbs_instance.suggest()
        row_id = suggest_output.suggestion['row_id']
        suggestion_input = suggest_output.suggestion

        # Simulate creating an observation with incrementing output
        new_observation = ObservationInParam(
            input=suggestion_input,
            output=i * 10,  # example outputs
            cost=(i + 1) * 0.5,
            is_failure=False,
        )

        carbs_instance.observe(new_observation)

    # Check that the number of outstanding suggestions hasn't changed
    assert len(carbs_instance.outstanding_suggestions) == initial_outstanding_suggestions

    # Check that the number of successful observations increased by num_iterations
    assert len(carbs_instance.success_observations) == initial_success_observations + num_iterations

    os.remove(db_name)

def test_observe(carbs_instance: CARBS) -> None:
    start_success_obs = len(carbs_instance.success_observations)
    start_failure_obs = len(carbs_instance.failure_observations)
    obs_success = ObservationInParam(input={x.name: x.search_center for x in carbs_instance.params}, output=1, cost=1)
    obs_failure = ObservationInParam(
        input={x.name: x.search_center for x in carbs_instance.params}, output=1, cost=1, is_failure=True
    )
    obs_success_output = carbs_instance.observe(obs_success)
    obs_failure_output = carbs_instance.observe(obs_failure)
    assert len(carbs_instance.success_observations) == start_success_obs + 1
    assert len(carbs_instance.failure_observations) == start_failure_obs + 1

    os.remove(db_name)


def test_forget(carbs_instance: CARBS) -> None:
    start_suggestions = len(carbs_instance.outstanding_suggestions)
    # Make a new suggestion
    suggest_output = carbs_instance.suggest()

    # Grab the row_id from the suggestion
    row_id = suggest_output.suggestion['row_id']

    # Now call forget_suggestion with that row_id
    carbs_instance.forget_suggestion(row_id)

    # We expect to revert to the original suggestion count
    assert len(carbs_instance.outstanding_suggestions) == start_suggestions

    os.remove(db_name)

def test_load_from_db(carbs_instance: CARBS) -> None:
    initial_outstanding_suggestions = len(carbs_instance.outstanding_suggestions)
    initial_success_observations = len(carbs_instance.success_observations)    
    
    # Make a few suggestions/observations
    for i in range(3):
        suggest_output = carbs_instance.suggest()
        obs = ObservationInParam(
            input=suggest_output.suggestion,
            output=float(i),
            cost=1.0,
            is_failure=False,
        )
        carbs_instance.observe(obs)

    # Now we create a fresh instance from the same DB
    new_instance = CARBS.load_from_db(
        carbs_instance.config,
        carbs_instance.params,
        carbs_instance.db_path,
    )

    # Compare the success observations count
    assert len(new_instance.success_observations) == len(carbs_instance.success_observations)

    # Compare outstanding suggestions count
    assert len(new_instance.outstanding_suggestions) == len(carbs_instance.outstanding_suggestions)

    os.remove(db_name)

    # Optionally compare actual outputs
    # for old_obs, new_obs in zip(carbs_instance.success_observations, new_instance.success_observations):
    #     assert old_obs.output == new_obs.output
    #     assert torch.allclose(old_obs.real_number_input, new_obs.real_number_input)