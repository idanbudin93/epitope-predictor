__author__ = 'Smadar Gazit'

#download all human proteins from swissprot:https://www.uniprot.org/uniprot/?query=*&fil=organism%3A%22Homo+sapiens+%28Human%29+%5B9606%5D%22+AND+reviewed%3Ayes&desc=no&sort=organism

import os
import pathlib 
import urllib
import urllib.request
import shutil 
import random


URL_FOR_DOWNLOAD = "https://www.iedb.org/downloader.php?file_name=doc/tcell_full_v3.zip"
OUT_PATH = "./samples"

def download_data(out_path, url, force=False):
    
    """ downloads the data to the specified out_path """
    
    pathlib.Path(out_path).mkdir(exist_ok=True)
    out_filename = os.path.join(out_path, "proteins.fasta")
    
    if os.path.isfile(out_filename) and not force:
        print(f'Proteins file {out_filename} exists, skipping download.')
    else:
        print(f'Downloading {url}...')
        with urllib.request.urlopen(url) as response, open(out_filename, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print(f'Saved to {out_filename}.')

    import zipfile
    with zipfile.ZipFile(out_filename, 'r') as zip_ref:
        zip_ref.extractall(out_path)
    return out_filename

download_data(out_path, url_for_download)