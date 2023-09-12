from urllib.request import urlopen
from zipfile import ZipFile
from pathlib import Path
import os

zipresp = urlopen("http://kvfrans.com/static/32_emoji.zip")
tempzip = open("temp.zip", "wb")
tempzip.write(zipresp.read())
tempzip.close()
zf = ZipFile("temp.zip")
zf.extractall(path = Path("./pic"))
zf.close()

os.remove("temp.zip")