try:
    import numpy as np
except:
    np = None

try:
    import pandas as pd
except:
    pd = None

try:
    import sklearn as sk
except:
    sk = None


assert np != None, "Numpy is not installed"
assert pd != None, "Pandas is not installed"
assert sk != None, "Sklearn is not installed"

assert np.__version__ == '1.13.0', "Your version is: {} if {} is newer it is ok".format(np.__version__, "0.13.0")
assert pd.__version__ == '0.19.2', "Your version is: {} if {} is newer it is ok".format(pd.__version__, "0.19.2")
assert sk.__version__ == '0.18.1', "Your version is: {} if {} is newer it is ok".format(sk.__version__, "0.18.1")

print("------------------------------------\nYou are ready for saturday's Dojo\n---------------------------------------------")
