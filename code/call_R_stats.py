# -------------------------------------------------------------------------------------
# Code to import and call R packages in Python
# Marta Blanco-Pozo, 2023

# Rpy2 TUTORIAL
# https://www.youtube.com/watch?v=GvmoOHkABNA
# https://www.marsja.se/r-from-python-rpy2-tutorial/
# -------------------------------------------------------------------------------------

import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector, IntVector
from rpy2.robjects import pandas2ri, Formula
import rpy2.robjects as robjects
from rpy2.robjects.lib import grdevices
from IPython.display import Image, display
from rpy2.robjects.conversion import localconverter


package_names = ('parallel', 'tidyverse', 'knitr', 'data.table', 'broom', 'car', 'afex', 'multcomp', 'emmeans', 'lme4',
                 'ggResidpanel', 'optimx', 'dfoptim', 'sjPlot', 'sjlabelled', 'sjmisc', 'glmmTMB')

base = rpackages.importr('base')
utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)
packnames_to_install = [x for x in package_names if not rpackages.isinstalled(x)]
if len(packnames_to_install) > 0:
  utils.install_packages(StrVector(package_names))

afex = rpackages.importr('afex')
afex.afex_options(emmeans_model = "multivariate")  # Change system default to use multivariate model for all follow-up tests [makes traditional Repeated Measures ANOVA simple-effects analysis consistent with SPSS/STATA]
emmeans = rpackages.importr('emmeans', robject_translations = {"recover.data.call": "recover_data_call1"})  # translations is necessary to be able to use the package
parallel = rpackages.importr('parallel')
lme4 = rpackages.importr('lme4')
ggResidpanel = rpackages.importr('ggResidpanel')
broom = rpackages.importr('broom')
car = rpackages.importr('car')

def convert_df_to_R(df):
  '''
  Convert pandas dattaframe into R
  '''
  with localconverter(robjects.default_converter + pandas2ri.converter):
    r_from_pd_df = robjects.conversion.py2rpy(df)
    return r_from_pd_df

