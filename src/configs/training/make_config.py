from configs import model_config, ds_prepare_config


def make_config(is_debug, mode):
    if mode == "segmentation":
        import configs.training.segmentation as training_config
    elif mode == "contours":
        import configs.training.contours as training_config
    else:
        raise ValueError("Некорректный мод", mode)
    try:
        compile_params = training_config._compile_configs[model_config.OUT_SIZE]
        compile_params["optimizer"] = training_config.OPTIMIZER
        compile_params["run_eagerly"] = is_debug
        tr_conf = {
            "batch_size": ds_prepare_config.BATCH_SIZE,
            "epochs": training_config.EPOCHS,
            "validation_split": ds_prepare_config.VALIDATION_TRAIN_SIZE,
            "compile_params": compile_params,
        }
    except AttributeError:
        raise
    return tr_conf
