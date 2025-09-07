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
for the installation and project setup please use the latest cuda setup version and use `Python  3.11.3` or later version.
then run the following command in the command line (cmd):

```
sudo apt update
sudp apt upgrade
pip install -r requirements.txt
```
also for this project we used Shahid Beheshti university Gitlab server storage for the datas.
if you don't have access to it please comment the ussage of them in the `main.py` file.

### Dataset's
you can download the following dataset used for this project from [Here](URL).
the Dataset has been gathered from  [UTKFACE](https://susanqq.github.io/UTKFace)
and for high resoulation we used [GFPGAN](https://github.com/TencentARC/GFPGAN).

