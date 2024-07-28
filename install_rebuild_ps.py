import re
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
f=open(script_dir+'/PhenotypeSeeker/.PSenv/bin/phenotypeseeker','r')
lines = f.readlines()
lines[0]='#!'+script_dir+'/PhenotypeSeeker/.PSenv/bin/pss_py38/bin/python\n'
#o=open()
with open(script_dir+'/PhenotypeSeeker/.PSenv/bin/phenotypeseeker', 'w') as file:
    file.writelines(lines)
