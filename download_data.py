import os
import pathlib 
import urllib
import urllib.request
import shutil 
import random
import zipfile
TEMP_FILENAME = "tmp.zip"
def download_data(out_path, csv_filename, url, force=False):
    
    """ downloads the data to the specified out_path """
    dir_path = pathlib.Path(out_path)
    dir_path.mkdir(exist_ok=True)
    tmp_path = dir_path.joinpath(TEMP_FILENAME)
    csv_path = dir_path.joinpath(csv_filename)
    
    if csv_path.is_file() and not force:
        print(f'csv file {str(csv_path)} exists, skipping download.')
    else:
        if tmp_path.is_file() and not force:
            print(f'zip file {str(tmp_path)} exists, skipping download.')  
        else:
            print(f'Downloading {url}...')
            with urllib.request.urlopen(url) as response, open(str(tmp_path), 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            print(f'Saved to {str(tmp_path)}.')
        print(f'Extracting zip {str(tmp_path)}.')
        with zipfile.ZipFile(str(tmp_path), 'r') as zip_ref:        
            zipinfo = zip_ref.infolist()[0]
            zipinfo.filename = str(csv_path)
            zip_ref.extract(zipinfo)
        tmp_path.unlink()

    return csv_filename