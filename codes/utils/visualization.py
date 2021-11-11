import os
import copy
import random
import matplotlib.pyplot as plt
from matplotlib import image

def show_images(filenames, stop, mode = "matplotlib", selection = "top"):
    """[summary]

    Args:
        filenames ([type]): [description]
        stop ([type]): [description]
        mode (str, optional): [description]. Defaults to "matplotlib".
        selection (str, optional): [description]. Defaults to "top".

    Returns:
        [type]: [description]
    """

    files = copy.deepcopy(filenames)
    if selection == "bottom":
        files = files[::-1]
    elif selection == "random":
        random.shuffle(files)
    else:
        pass

    if stop:
        files = files[0:stop]
    
    n = len(files)

    if mode == "matplotlib":
        if n >= 2:
            ncols = 5
            nrows = int(n/ncols)
        
            fig, axes = plt.subplots(nrows = nrows, ncols=ncols, figsize=(5*ncols, nrows*5))
            for file, ax in zip(files, axes.ravel()):
                img = image.imread(file)
                ax.imshow(img)
                ax.set_title(file.split(os.sep)[-1])
            plt.tight_layout()
            plt.show()
        else:
            img = image.imread(files[0])
            plt.imshow(img)

    elif mode == "PIL":
        for file in files:
            im = Image.open(file)
            im.show()

    else: # mode == "cv2"
        for file in files:
            im = cv2.imread(file,0)
            cv2.startWindowThread()
            cv2.imshow(file.split(os.sep)[-1], im)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

        cv2 .destroyAllWindows()

        # In macOS there is a known bug that the last image will freeze unless this sequence of waitkey(1) 
        # is included in the code. 
        # Source: https://stackoverflow.com/questions/6116564/destroywindow-does-not-close-window-on-mac-using-python-and-opencv

        for i in range(5):
            cv2.waitKey(1)
            
    return
