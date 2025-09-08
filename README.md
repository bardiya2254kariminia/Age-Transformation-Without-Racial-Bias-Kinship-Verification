# A Race Biass Free Face aging model
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.11.2-blue)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-orange)


Kinship verification is a challenging task due to the age gap between parent and child photos, as their same-age images are rarely available. Existing face aging models aim to address this gap but often suffer from racial bias and poor identity preservation, which reduce fairness and accuracy in kinship verification. To tackle these issues, we introduce RA-GAN, a race-unbiased face aging model that incorporates two novel components: RACEpSp, which mitigates racial bias, and a feature mixer, which enhances identity preservation. The unbiased synthesized faces are then used to transform parent and child images into the same age group for verification. Experimental results on the KinFaceW-I and KinFaceW-II datasets show that RA-GAN outperforms prior methods, achieving an average improvement of 13.14% over SAM-GAN across all age groups and 9.1% over CUSP-GAN in the 60+ age group in terms of racial accuracy. Furthermore, RA-GAN consistently preserves identity features better than competing models and improves verification accuracy across all kinship relationships. These results demonstrate that RA-GAN provides a more fair, accurate, and identity-preserving solution for kinship verification.
Be aware that the paper is still under revision in the journal.

<p align="center">
<img src="images\Intro1.jpg" width="1000px"/>
<!-- <img src="images\Intro2.jpg" width="1000px"/>     -->
<img src="images\Intro3.jpg" width="1000px"/>    
<img src="images\Intro4.jpg" width="1000px"/>    
<img src="images\Intro5.jpg" width="1000px"/>    
</p>

#### Description
Official implementation of the model RA-GAN from the paper "A Race Biass Free Face aging model"
from the first image to the end Races are white, Asian , Black , Indian respectivly.

### Project Tree
```
├── configs/              # Specifying the path's for
|   ├── path_config       # fro setting the path of the datasets and the models
├── models/               # Model architectures (GAN, encoders, etc.)
|   ├── psp.py            # setting up the entire model and modules except generator
|   ├── race_net.py       # the racenet for the training
├── criteria/           # the losses and their implementations
├── utils/              # Helper functions
├── requirements.txt    # Python dependencies
├── cmd_options.txt     # for internal  command running
├── config.json         # the entire setup for hyperparams of the project
├── main.py             # main file for training and inferencing 
└── README.md           # Project description
```


### Installation and setup
for the installation and project setup please use the following setup :
- `Python  3.11.3`
- `NVIDIA GPU`
- `CUDA CuDNN` package the lattest version (important)
then run the following command in the command line (cmd):

```
sudo apt update
sudp apt upgrade
git clone https://github.com/bardiya2254kariminia/An-Age-Transformation-whitout-racial-bias-for-Kinship-verification.git
cd An-Age-Transformation-whitout-racial-bias-for-Kinship-verification
pip install -r requirements.txt
```
also for this project we used Shahid Beheshti university Gitlab server storage for the datas.
if you don't have access to it please comment the ussage of them in the `main.py` file.

### Pretrained Model & Dataset's
you can download the following dataset used for this project from [Here](URL).
the Dataset has been gathered from  [UTKFACE](https://susanqq.github.io/UTKFace)
and for high resoulation we used [GFPGAN](https://github.com/TencentARC/GFPGAN).

#### Dataset

| Path | Description
| :--- | :----------
|[ِِDataset]() |the Dataset has been gathered from  [UTKFACE](https://susanqq.github.io/UTKFace) and for high resoulation we used [GFPGAN](https://github.com/TencentARC/GFPGAN).
|[Kinface1 Age Transformation(from 20-80)]() | The output for the [Kinface1](https://www.kinfacew.com/datasets.html) dataset on the RA-GAN (Our) model
|[Kinface2 Age Transformation(from 20-80)]() | The output for the [Kinface2](https://www.kinfacew.com/datasets.html) dataset on the RA-GAN (Our) model

#### pretrained models and weights

| Path | Description
| :--- | :----------
|[RA-GAN weights]() | The core of the paper contribution, Train on our Dataset for ethnicity fairness..
|[FFHQ StyleGAN](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view?usp=sharing) | StyleGAN model pretrained on FFHQ taken from [rosinality](https://github.com/rosinality/stylegan2-pytorch) with 1024x1024 output resolution.
|[IR-SE50 Model](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view?usp=sharing) | Pretrained IR-SE50 model taken from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) for use in our ID loss during training.
|[VGG Age Classifier](https://drive.google.com/file/d/1atzjZm_dJrCmFWCqWlyspSpr3nI6Evsh/view?usp=sharing) | VGG age classifier from DEX and fine-tuned on the FFHQ-Aging dataset for use in our aging loss
|[Resnet_34_7](https://drive.google.com/drive/folders/1F_pXfbzWvG-bhCpNsRj6F_xsdjpesiFu?usp=sharing) | The Resnet34 has been trained on the [FairFace](https://github.com/dchen236/FairFace/tree/master?tab=readme-ov-file) dataset for with 7 output type White,Black,Indian,Asian and others (for more information visit [HERE](https://github.com/dchen236/FairFace/tree/master?tab=readme-ov-file)). 
|[Resnet_34_4](https://drive.google.com/drive/folders/1F_pXfbzWvG-bhCpNsRj6F_xsdjpesiFu?usp=sharing) | The Resnet34 has been trained on the [FairFace](https://github.com/dchen236/FairFace/tree/master?tab=readme-ov-file) dataset for with 4 output type White,Black,Indian,Asian. We used it for the Race preservation evavluation metric.  

### Motiation
Kinship verification aims to determine whether two people are biologically related based on facial images. A major challenge in this task is the age gap between parents and children, since their photos are usually taken at different life stages. Existing face aging models try to reduce this gap by synthesizing same-age faces, but most of them suffer from two critical issues:

-   Racial bias – Many face aging models are trained on datasets dominated by specific races, which leads to distorted or less realistic results for underrepresented groups. This bias negatively impacts kinship verification fairness.

-   Identity preservation – Some existing GAN-based models fail to maintain the unique identity features of individuals while performing age transformations, reducing verification reliability.

To overcome these issues, we propose RA-GAN, a race-unbiased face aging GAN that integrates two novel modules — RACEpSp and a feature mixer — to generate realistic, identity-preserving, and racially unbiased age-progressed faces. By aligning parent and child images to the same age group, RA-GAN significantly improves kinship verification accuracy and ensures fair performance across racial groups.

### Method overview

#### Age Transformation
An overview of this phase is depicted in the  following image:
<p align="center">
<img src="images\method_overview1.png" width="1000px"/>
</p>

-   `Age_Encoder`: used for extracting corrolations between desired age and base image. 
Our method conducts using 4 seperated module
-   `RACEPsP` : used for getting and embbeded vector of input image that contains facial and identity concepts.
although the dataset for different tasks cannot guarantee balance  racial ratio. we have racenet which is used to extract race related features from the base image and give us 3 set of tensor, which give us different abstraction and detail for representing these informations. also we have a Pyramidnet which is a Pixle2Style2Pixle encoder for extracting facial infromation adn idenetity of the base image.Also Racenet anotehr self specified module is goinf to used for morphing the informations of both racenet and Pyramidnet.
<p align="center">
<img src="images\method_overview2.png" width="1000px"/>
</p>

-   `Feature_Mixer`: in the end after getting concept embeddings from both RACEPsP and Age_Encoder,each value in aging vector can have different impact on different face embedding value, thus ,we have  to mix them in a manner to get good aging and facial preservation quality.
<p align="center">
<img src="images\method_overview3.png" width="1000px"/>
</p>

-   `StyleGan_V2`: in the end phase, we have to construct the image based on the feature_mixer's output and we will use stylegan because of its capability on creating high quality and fidelity image conditioned on a latent vector.

#### Kinship verification
In this phase we will use out Age Transformation framework for changinf the age of the KinfaceI and KinfaceII dataset
and givve them to the Kinship verificator model and get the output.
<p align="center">
<img src="images\method_overview4.png" width="1000px"/>
</p>

one of the challenges here is the images in those datasets arent  covering full head thus our model cannot generate usefull images and we will get unreliable outputs.
To overcome  this obstacle, first, we will use a data augmentation technique called  mirror augmentation. we believed putting the face of a person in the middle of the image and using using a good  mirror augmentation can somehow imply to out desired base image for the network. Still the image is not good enough for being base image for us thus we used a pretrained Pixle2Style2Pixle model trained on our dataset for getting a full head output for each imahe in the Kinface datasets.
Then we will feed the generated images to RA-GAN and get the transformed images  for each ages and give them to the Kinship verificator. Here the Kinship verificator is a [D4ML](https://www.researchgate.net/publication/379187260_Kinship_verification_based_on_multi-scale_feature_fusion?utm_source=chatgpt.com) model which has the best  capability  for extracting kinship information and correlations.
