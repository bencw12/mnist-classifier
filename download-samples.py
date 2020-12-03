import os
import urllib.request
import gzip
import shutil

try:
	os.mkdir('samples')
except:
	pass
os.chdir('samples')
url1 = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
url2 = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
url3 = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
url4 = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

urllib.request.urlretrieve(url1, 'train-images-idx3-ubyte.gz')
urllib.request.urlretrieve(url2, 'train-labels-idx1-ubyte.gz')
urllib.request.urlretrieve(url3, 't10k-images-idx3-ubyte.gz')
urllib.request.urlretrieve(url4, 't10k-labels-idx1-ubyte.gz')\

with gzip.open('train-images-idx3-ubyte.gz', 'rb') as f_in:
    with open('train-images-idx3-ubyte', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

with gzip.open('train-labels-idx1-ubyte.gz', 'rb') as f_in:
    with open('train-labels-idx1-ubyte', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)\

with gzip.open('t10k-images-idx3-ubyte.gz', 'rb') as f_in:
    with open('t10k-images-idx3-ubyte', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

with gzip.open('t10k-labels-idx1-ubyte.gz', 'rb') as f_in:
    with open('t10k-labels-idx1-ubyte', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

