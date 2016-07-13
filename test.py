# Imports
from lena_DDE import DDEStack
#import cProfile
#import pstats

# Create test DDE stack
LMkwargs = {'damping': 1.,
            'max_iter': 20,
            'ptol': 1e-6}
kwargs = {
    'filename': 'test_short.tif',
    'regionsize': 25,
    'regionspacing': 50,
    'euler': True,
    'LMkwargs': LMkwargs
    }
stack = DDEStack(**kwargs)



# Displace grid displacements, saving each frame
#basename = 'test_output/frame'
#savekwargs = {'dpi': 150}
#stack.show_deformation(basename=basename, savekwargs=savekwargs,
#                       straincolormap='bwr', strainlim=0.25, alpha=0.5)

## Profile DDEStack creation via cProfile
#cProfile.run('DDEStack(**DDEparams)', 'runstats', sort='cumulative')
#stream = open('runstats.txt', 'w');
#stats = pstats.Stats('runstats', stream=stream)
#stats.sort_stats('cumulative').print_stats()
#stream.close()