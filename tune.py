import time
import hydra
import logging
import os
import sys
from omegaconf import DictConfig
from syne_tune import Tuner, StoppingCriterion
from syne_tune.optimizer.baselines import (
    RandomSearch,
    BayesianOptimization,
    ASHA,
    MOBSTER,
)
from syne_tune.backend.local_backend import LocalBackend
from syne_tune.callbacks.tensorboard_callback import TensorboardCallback
from syne_tune.results_callback import StoreResultsCallback

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from deeptune.search_space.search_space import SearchSpace
from deeptune.utils.syne_tune_utils import adapt_search_space


@hydra.main(config_path="config", config_name="deeptune_config")
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # print the current working directory
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Current file path: {os.path.dirname(__file__)}")

    ss = SearchSpace(os.path.join(os.path.dirname(__file__), cfg.search_space_path))
    config_space = adapt_search_space(ss)
    
    # add the dataset name to the config
    config_space["dataset_name"] = cfg.dataset_name
    
    logger.info(f"Config space: {config_space}")

    backend = LocalBackend(entry_point=os.path.join(os.path.dirname(__file__), cfg.train_script),
                           pass_args_as_json=True)

    method_kwargs = dict(
        metric=cfg.metric,
        mode=cfg.mode,
        random_seed=cfg.random_seed,
        max_resource_attr=cfg.max_resource_attr,
        search_options={"num_init_random": cfg.n_workers + 2},
        resource_attr=cfg.resource_attr,
    )
    
    if cfg.opt_type == "RS":
        scheduler = RandomSearch(config_space, **method_kwargs)
    elif cfg.opt_type == "BO":
        scheduler = BayesianOptimization(config_space, **method_kwargs)
    elif cfg.opt_type == "ASHA":
        scheduler = ASHA(config_space, type=cfg.sch_type, **method_kwargs)
    elif cfg.opt_type == "MOBSTER":
        scheduler = MOBSTER(config_space, type=cfg.sch_type, **method_kwargs)
    
    
    callback = [
        TensorboardCallback(target_metric=cfg.metric, mode=cfg.mode),
        StoreResultsCallback(),
    ]
    
    trial_name = cfg.dataset_name + "-" + time.strftime("%Y%m%d-%H%M%S")
    
    tuner = Tuner(
        trial_backend=backend,
        scheduler=scheduler,
        n_workers=cfg.n_workers,
        stop_criterion=StoppingCriterion(max_wallclock_time=cfg.max_wallclock_time),
        trial_backend_path=trial_name,
        callbacks=callback,
        tuner_name="deeptune-" + trial_name,
        metadata={"trial_name": trial_name},
    )
    
    tuner.run()
    
    logger.info(f"Best config: {tuner.best_config()}")
    # save the best config on th trial folder
    with open(f"{trial_name}/best_config.json", "w") as f:
        f.write(str(tuner.best_config()))

if __name__ == "__main__":
    main()