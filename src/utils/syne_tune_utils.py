from syne_tune.config_space import randint, loguniform, choice


def adapt_search_space(search_space):

    original_config_space = {
        #'lr': loguniform(1e-5, 1e-1),
        # "batch_size": randint(1, 8),
        # "model": choice(["edgenext_x_small", "volo_d5_512"]),
        "epochs": 50,
        "report_synetune": 1,
        "val-split": "val",
        "train-split": "train",
    }

    omit_args = [
        "epochs",
        # "data_augmentation",
        "amp",
        "stoch_norm",
        "linear_probing",
        "trivial_augment",
    ]

    for key, values in search_space.data.items():
        if key not in omit_args:
            if isinstance(values, dict):
                values = values["options"]
            values = [x for x in values if x is not None and x != "None"]
            if "True" in values:
                values = [1 if x == "True" else 0 for x in values]
            original_config_space[key] = choice(values)

    return original_config_space
