from configs import io_config, ds_prepare_config
from DSPreparerFactory import DSPreparerFactory, InMemPrepFactory


__factories = {"in_mem": InMemPrepFactory}
__ds_prep_factory = "in_mem"


def _create_prep_fact(is_mini, split, mask, type, **fact_kwargs) -> DSPreparerFactory:
    images_dir = io_config.get_samples_dir(is_mini, split)
    masks_dir = io_config.get_samples_dir(is_mini, split, mask)
    ds_path = images_dir.parent / masks_dir.with_suffix(".npz").name
    return __factories[type](
        images_dir, masks_dir, ds_path, ds_prepare_config.RANDOM_STATE, **fact_kwargs
    )


def create_train_fact(mask, is_mini=True) -> DSPreparerFactory:
    return _create_prep_fact(
        is_mini=is_mini, split="train", mask=mask, type=__ds_prep_factory, shuffle=True
    )


def create_val_fact(mask, is_mini=True) -> DSPreparerFactory:
    return _create_prep_fact(
        is_mini=is_mini, split="val", mask=mask, type=__ds_prep_factory, shuffle=False
    )


def create_test_fact(mask, is_mini=True):
    return _create_prep_fact(
        is_mini=is_mini, split="test", mask=mask, type=__ds_prep_factory, shuffle=False
    )
