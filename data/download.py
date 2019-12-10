import urllib.request
import tarfile
from tqdm import tqdm
import os

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

url = 'http://homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz'
output_path = 'VCTK-Corpus.tar.gz'

download_url(url, output_path)
print('Download complete')
print('Extracting tar file. This will take a while...')
tf = tarfile.open(output_path)
tf.extractall()
print('Extract complete!')
os.remove(output_path)
print("Deleted " + output_path)
