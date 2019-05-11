import numpy as np
import sys

np.random.seed(0)

ans = np.random.randint(0, 2, (1000000))
with open(sys.argv[3], 'w') as f:
    f.write('id,label\n' + '\n'.join(
    ['{},{}'.format(*a) for a in enumerate(ans)]) + '\n')