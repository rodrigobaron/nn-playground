import os
import os.path as P
import urllib.request
import numpy as np
class Downloadable:
    def __init__(self, url):
        self.url = url

    def try_download(self, filename, work_directory, silent=False):
        if not P.exists(work_directory):
            os.makedirs(work_directory)
        filepath = P.join(work_directory, filename)
        if not os.path.exists(filepath):
            filepath, _ = urllib.request.urlretrieve(self.url + filename, filepath)
            statinfo = os.stat(filepath)
            if not silent:
                print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        return filepath

    def read_numpy32(self, bytestream):
        dt = np.dtype(np.uint32).newbyteorder('>')
        return np.frombuffer(bytestream.read(4), dtype=dt)