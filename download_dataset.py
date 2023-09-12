from urllib.request import urlopen
from zipfile import ZipFile
from pathlib import Path
import os

def download_dataset(output_dir):
    zipresp = urlopen("http://kvfrans.com/static/32_emoji.zip")
    tempzip = open("temp.zip", "wb")
    tempzip.write(zipresp.read())
    tempzip.close()
    zf = ZipFile("temp.zip")
    zf.extractall(path = output_dir)
    zf.close()
    
    os.remove("temp.zip")

if __name__ == "__main__":
    download_dataset(Path("./pic"))