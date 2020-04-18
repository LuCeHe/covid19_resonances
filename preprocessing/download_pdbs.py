
import os
import urllib.request


# here some covid19 proteins in .pdb
protein_url = 'https://files.rcsb.org/download/6Y84.pdb'
protein_url = 'https://files.rcsb.org/download/6LU7.pdb'
protein_url = 'https://files.rcsb.org/download/6M03.pdb'

# in this website more proteins can be found in .cif file
# https://pdbj.org/featured/covid-19

# check this website to turn .cif into pdb with python
# https://gist.github.com/sbliven/b7cc2c5305aa3652a75a580ae7c6ce33


print('Beginning file download with urllib2...')

_, protein_name = os.path.split(protein_url)
urllib.request.urlretrieve(protein_url, r'data/cristal_structure/' + protein_name)