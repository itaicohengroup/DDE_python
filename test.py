
# Imports
from lena_DDE import DDEStack

# Create DDE stack
kwargs = {
    'filename': 'test_short.tif',
    'regionsize': 51,
    'regionspacing': 100,
    'euler': False,
    }
stack = DDEStack(**kwargs)

# Displace grid displacements, saving each frame
basename = 'test_output/frame'
stack.show_displacements(basename=basename)