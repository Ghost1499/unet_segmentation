from ds_prepare.ds_prep_fact import create_train_fact, create_val_fact
from training.ModelTrainer import ModelTrainer

__mask = "contours_ls"
__model_name = "unet0cls"


def main():
    train_fact = create_train_fact(__mask)
    test_fact = create_val_fact(__mask)
    trainer = ModelTrainer(
        __model_name,
    )


if __name__ == "__main__":
    main()
