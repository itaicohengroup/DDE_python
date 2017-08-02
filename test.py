# Imports
from DDE import DDEStack
import dill

# Create test DDE stack
LMkwargs = {'damping': 100.,
            'max_iter': 50,
            'ptol': 1e-7}
kwargs = {
    'filename': 'test_short2_crop_blur.tif',
    'regionsize': 25,
    'regionspacing': 5,
    'euler': True,
    'LMkwargs': LMkwargs
    }
stack = DDEStack(**kwargs)

# Displace grid displacements, saving each frame
basename = 'test_output_euler/frame'
savekwargs = {'dpi': 150}
stack.show_deformation(stack, basename=basename, savekwargs=savekwargs,
                       showboxes=False,
                       straincolormap='bwr', strainlim=0.25, alpha=0.25, 
                       xlim=(0, 512), ylim=(0, 512))

# Also use dill (pickle) to save the stack instance
stackname = 'test_output_euler/stack.p'
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