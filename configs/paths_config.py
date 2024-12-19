dataset_paths = {
    #  Face Datasets (In the paper: FFHQ - train, CelebAHQ - test)
    "ffhq": "",
    "celeba_test": "",
    # celeba_10000
    "train_celeba_10000": rf"downloads\img_align_celeba",
    "test_celeba_10000": "",
    "celeba_10000_annotations": rf"downloads\annotation_10000_celeba.csv",
}

model_paths = {
    "stylegan_ffhq": "/content/ebrahimi_moghadam_refactored/criteria/pretrained_models/stylegan2-ffhq-config-f.pt",
    "ir_se50": "/content/ebrahimi_moghadam_refactored/criteria/pretrained_models/model_ir_se50.pth",
    "shape_predictor": "/content/ebrahimi_moghadam/criteria/pretrained_models/shape_predictor_68_face_landmarks.dat",
    "moco": "/content/ebrahimi_moghadam_refactored/criteria/pretrained_models/moco_v2_800ep_pretrain.pt",
    "generator_style_gan2": "/content/ebrahimi_moghadam_refactored/criteria/pretrained_models/stylegan2-ffhq-config-f.pt",
    "Encoder4Editing": "/content/ebrahimi_moghadam_refactored/criteria/pretrained_models/e4e_ffhq_encode.pt",
    "Discriminator_nvlab": "",
}
