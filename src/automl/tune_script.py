from syne_tune import Tuner, StoppingCriterion
from syne_tune.config_space import randint, loguniform, choice
from syne_tune.optimizer.schedulers import FIFOScheduler
from syne_tune.backend.local_backend import LocalBackend
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from search_space.search_space import SearchSpace
from src.utils.syne_tune_utils import adapt_search_space
ss = SearchSpace("search_space/search_space_v1.yml")



config_space = adapt_search_space(ss)
print(config_space)
# Path to the training script
train_script = "/home/werbya/QTB/src/automl/train_script.py"

# Create a backend
backend = LocalBackend(entry_point=train_script, pass_args_as_json=True)

mode = "max"
metric = "loss"
resource_attr = "epoch"
max_resource_attr = "epochs"
random_seed = 42
n_workers = 4
max_wallclock_time = 600
stop_criterion = StoppingCriterion(max_wallclock_time=max_wallclock_time)

# Create a scheduler
method_kwargs = dict(
    metric=metric,
    mode=mode,
    random_seed=random_seed,
    max_resource_attr=max_resource_attr,
    search_options={"num_init_random": n_workers + 2},
)
scheduler = FIFOScheduler(config_space=config_space, **method_kwargs)

# Create a tuner
tuner = Tuner(
    trial_backend=backend,
    scheduler=scheduler,
    n_workers=n_workers,
    stop_criterion=stop_criterion,
    trial_backend_path="trail_backend",
)

# Run the tuning process
tuner.run()

# print the best configuration
print(tuner.best_config())
