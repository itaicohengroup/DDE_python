# Imports
from lena_DDE import DDEStack
import dill

# Create test DDE stack
LMkwargs = {'damping': 100.,
            'max_iter': 20,
            'ptol': 1e-6}
kwargs = {
    'filename': 'test_short2_crop.tif',
    'regionsize': 35,
    'euler': True,
    'LMkwargs': LMkwargs
    }
stack = DDEStack(**kwargs)

# Displace grid displacements, saving each frame
basename = 'test_output/frame'
savekwargs = {'dpi': 150}
stack.show_deformation(basename=basename, savekwargs=savekwargs,
                       straincolormap='bwr', strainlim=0.25, alpha=0.5)

# Also use dill (pickle) to save the stack instance
stackname = 'test_output/stack.p'
f = open(stackname, 'wb')
dill.dump(stack, f)
f.close()

## Profile DDEStack creation via cProfile
#import cProfile
#import pstats
#cProfile.run('DDEStack(**DDEparams)', 'runstats', sort='cumulative')
#stream = open('runstats.txt', 'w');
#stats = pstats.Stats('runstats', stream=stream)
#stats.sort_stats('cumulative').print_stats()
#stream.close()