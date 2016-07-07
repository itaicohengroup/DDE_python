# Imports
from lena_DDE import DDEStack
import cProfile
import pstats

# Create test DDE stack
kwargs = {
    'filename': 'test_short.tif',
    'regionsize': 51,
    'regionspacing': 100,
    'euler': False,
    }
stack = DDEStack(**kwargs)

# Displace grid displacements, saving each frame
basename = 'test_output/frame'
savekwargs = {'dpi': 150}
stack.show_displacements(basename=basename, savekwargs=savekwargs)

# Profile DDEStack creation
cProfile.run('DDEStack(**DDEparams)', 'runstats', sort='cumulative')
stream = open('runstats.txt', 'w');
stats = pstats.Stats('runstats', stream=stream)
stats.sort_stats('cumulative').print_stats()
stream.close()
