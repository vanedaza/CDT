# CDT: COMPOUND DECISION TREE

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)


<div class="alert alert-block alert-warning" align="center">
<table><tr><td><img src='CDT_log.png' width="250" height="250"></td></tr></table>
</div>

CDT is a customised model, which is composed of 3 models (M1, M2 and M3). In turn, the Mi models are composed of commonly used machine learning models and a voting system to classify binary eclipsing systems in contact (C), semi-detached (SD) and detached (D) using as input time series.

    - M1 classifies C and D EBs
    
    - M2 classifies D and SD EBs
    
    - M3 classifies C and SD EBs 
    

              ALGORITHM:
              The input x enter to model M1
                      if M1(x) = D => classify M2(x)
                      if M1(x) = C => classify M3(x)
                      


## Prerequisites and Installation

<div class="alert alert-block alert-warning">
    <b>If you want to install the packages in your usual user, you can use the comand:</b>
           
           
        $ pip install -r /path/to/requirements.txt
</div>


If you prefer, you can create a  virtual environment of conda o pip:

### Conda

<div class="alert alert-block alert-warning">
    <b>create conda enviroment:</b>
           
           
        $ conda create --name environment_name python=3.7

</div>


<div class="alert alert-block alert-warning">
    <b>activate your environment:</b>
           
           
        $ source activate environment_name
</div>

<div class="alert alert-block alert-warning">
    <b>install requirements:</b>
           
           
       $ conda install --force-reinstall -y -q --name cdt -c conda-forge --file requirements.txt
</div>


<div class="alert alert-block alert-warning">
    <b>deactivate your environment:</b>
           
           
       $ conda deactivate
</div>



### Pip

<div class="alert alert-block alert-warning">
    <b>create conda enviroment:</b>
           
           
        $ virtualenv environment_name -p python3

</div>


<div class="alert alert-block alert-warning">
    <b>activate your environment:</b>
           
           
        $ source environment_name/bin/activate
</div>

<div class="alert alert-block alert-warning">
    <b>install requirements:</b>
           
           
       $ pip install -r /path/requirements.txt
</div>


<div class="alert alert-block alert-warning">
    <b>deactivate your environment:</b>
           
           
       $ deactivate
</div>


## Use

This repository provides tools exemplified in notebooks for the preprocessing, curation and extraction of time series features used in the classification done by CDT and a notebook to generate light curves and tables with the most relevant characteristics of the classified eclipsing binary system. The order and use of each of them is described below:

### **Data**

1_curacion.ipynb In this notebook is generated:
 
- Folder containing the time series information of the eclipsing binary systems to be classified into Detached, Semi-detached and Contact.
- File with the id of the eclipsing binary system, right ascension, declination magnitude and the periods calculated from the frequencies                     'FreqKfi2', 'FreqLfl2', 'FreqLSG', 'FreqPDM' and, 'FreqSTR' contained in the vivace table.


2_features.ipynb In this notebook for each eclipsing binary system we extract the features with feets and join them with the periods. We add a new        feature, the difference of the minima and also perform a new curation, pre-process the data, and prepare the data for input to the classification        model.


### **Model**

Preprocessing Contains some models that do data preprocessing such as minimising and feature selection.
 
Classifier Contains everything needed for classification.

CDT.ipynob It is the notebook that I run to do the classifications.
 
 
### **Report**

CL.ipynb  This notebook is used to generate a .tex containing the light curves of the eclipsing binary systems and a table with system and classification information.
 
 
 
For more details about the training and calibration of ROGER please read [Automated classification of eclipsing binary systems in the VVV Survey](https://arxiv.org/abs/2302.01200).
## Authors

Daza-Perilla, I. V. (IATE-OAC-UNC) <a itemprop="sameAs"  href="https://orcid.org/my-orcid?orcid=0000-0001-6216-9053" target="orcid.widget" rel="noopener noreferrer" style="vertical-align:top;"> <img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="width:1em;margin-right:.5em;" alt="ORCID iD icon"></a>

Gramajo, L. V. (OAC-UNC)
 
Lares, M.      (IATE-OAC-UNC)

Palma, T.      (OAC-UNC)

Ferreira Lopes, C. E. (Millennium Institute of Astrophysics)

Minniti, D.   (Vatican Observatory)

Clariá, J. J. (OAC-UNC)



# Citation

If you use this software please cite [Automated classification of eclipsing binary systems in the VVV Survey](https://arxiv.org/abs/2302.01200).
