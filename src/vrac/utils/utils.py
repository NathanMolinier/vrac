import tempfile
import datetime

def tmp_create(basename):
    """Create temporary folder and return its path
    Based on https://github.com/spinalcordtoolbox/spinalcordtoolbox/
    """
    prefix = f"{basename}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    tmpdir = tempfile.mkdtemp(prefix=prefix)
    print(f"Creating temporary folder ({tmpdir})")
    return tmpdir

##
def normalize(arr):
    '''
    Normalize image
    '''
    ma = arr.max()
    mi = arr.min()
    return ((arr - mi) / (ma - mi + 0.00001))