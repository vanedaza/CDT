# CDT: COMPOUND DECISION TREE

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)


<div class="alert alert-block alert-warning">

<table><tr><td><img src='CDT_log.png'></td></tr></table>
</div>

CDT is a customised model, which is composed of 3 models (M1, M2 and M3). In turn, the Mi models are composed of commonly used machine learning models and a voting system.

    - M1 classifies C and D EBs
    
    - M2 classifies D and SD EBs
    
    - M3 classifies C and SD EBs 
    

              ALGORITHM:
              The input x enter to model M1
                      if M1(x) = D => classify M2(x)
                      if M1(x) = C => classify M3(x)


## Use

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

# Authors

Daza-Perilla, I. V. (IATE-OAC-UNC) <a itemprop="sameAs"  href="https://orcid.org/my-orcid?orcid=0000-0001-6216-9053" target="orcid.widget" rel="noopener noreferrer" style="vertical-align:top;"> <img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="width:1em;margin-right:.5em;" alt="ORCID iD icon"></a>

Gramajo, L. V. (OAC-UNC)
 
Lares, M.      (IATE-OAC-UNC)

Palma, T.      (OAC-UNC)

Ferreira Lopes, C. E. (Millennium Institute of Astrophysics)

Minniti, D.   (Vatican Observatory)

Clariá, J. J. (OAC-UNC)



# Citation

If you use this software please cite [Article](https://arxiv.org/abs/2302.01200).
