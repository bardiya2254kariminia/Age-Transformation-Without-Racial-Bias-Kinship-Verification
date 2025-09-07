from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
    "ffhq_encode": {
        "transforms": transforms_config.EncodeTransforms,
        "train_source_root": dataset_paths["ffhq"],
        "train_target_root": dataset_paths["ffhq"],
        "test_source_root": dataset_paths["celeba_test"],
        "test_target_root": dataset_paths["celeba_test"],
    },
    "celeba_10000": {
        "transforms": transforms_config.EncodeTransforms,
        "annotations": dataset_paths["celeba_10000_annotations"],
        "train_source_root": dataset_paths["train_celeba_10000"],
        "train_target_root": dataset_paths["train_celeba_10000"],
        "test_source_root": dataset_paths["test_celeba_10000"],
        "test_target_root": dataset_paths["test_celeba_10000"],
    },
    "rage_gan": {
        "transforms": transforms_config.EncodeTransforms,
        "annotations": dataset_paths["rage_gan_train_annotations"],
        "train_source_root": dataset_paths["rage_gan_train"],
        "train_target_root": dataset_paths["rage_gan_train"],
        "test_source_root": dataset_paths["rage_gan_test"],
        "test_target_root": dataset_paths["rage_gan_test"],
    },
}
