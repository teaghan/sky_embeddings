import numpy as np
import os
import sys

# Command line arguments
start_n = int(sys.argv[1])
end_n = int(sys.argv[2])

for n in range(start_n, end_n+1):
    os.system('scancel %i' % n)

    #os.system('chmod u+x %s' % os.path.join(results_dir, 'combine_root.sh'))

