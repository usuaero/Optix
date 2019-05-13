import optix as opt
from random import random
import sys
import numpy as np
import os
import subprocess

subprocess.call("rm ./test/*.txt",shell=True)
test_files = [f for f in os.listdir('./test/') if os.path.isfile(os.path.join('./test/',f))]
for test_file in test_files:
    print(test_file)
    exec(open("./test/"+test_file).read())
