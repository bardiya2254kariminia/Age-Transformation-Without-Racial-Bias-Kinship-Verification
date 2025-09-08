# A Race Biass Free Face aging model
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" height=22.5></a>
> The age gap in kinship verification addresses the time difference between the
photos of the parent and the child. Moreover, their same-age photos are often
unavailable, and face aging models are racially biased, which impacts the likeness
of photos. Therefore, we propose a face aging GAN model, RA-GAN, consisting
of two new modules, RACEpSp and a feature mixer, to produce racially unbiased images. The unbiased synthesized photos are used in kinship verification to investigate the results of verifying same-age parent-child images. The experiments
demonstrate that our RA-GAN outperforms SAM-GAN on an average of 13.14%
across all age groups, and CUSP-GAN in the 60+ age group by 9.1% in terms of
racial accuracy. Moreover, RA-GAN can preserve subjects’ identities better than
SAM-GAN and CUSP-GAN across all age groups.Our method conduct a fine grained module called race mixer in-order to
Generalize the models capability on Generating desirable image with preserved ethnicity and with few amount of data.
Be aware that the paper is still under revision in the journal.

<p align="center">
<img src="images\Intro1.jpg" width="1000px"/>
<!-- <img src="images\Intro2.jpg" width="1000px"/>     -->
<img src="images\Intro3.jpg" width="1000px"/>    
<img src="images\Intro4.jpg" width="1000px"/>    
<img src="images\Intro5.jpg" width="1000px"/>    
</p>

## Description
Official implementation of the model RA-GAN from the paper "A Race Biass Free Face aging model"
from the first image to the end Races are white, Asian , Black , Indian respectivly.


### Installation and setup
for the installation and project setup please use the following setup :
- `Python  3.11.3`
- `NVIDIA GPU`
- `CUDA CuDNN` package the lattest version (important)
then run the following command in the command line (cmd):

```
sudo apt update
sudp apt upgrade
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


