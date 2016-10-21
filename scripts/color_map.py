import fcn
import numpy as np

if __name__ == '__main__':
    """"""
    cmap = fcn.util.labelcolormap()
    cmap = (cmap * 255).astype(np.uint8)
    with open('color_map.txt', 'a') as the_file:
      for i, colr in enumerate(cmap):
          the_file.write(str(i)+' - '+str(colr)+'\n')