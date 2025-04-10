import os
import matplotlib.pyplot as plt
import csv

class saveResult:
    @classmethod
    def saveFigures(cls,directory,name, lib='plt' , ext='png', close=True, verbose=True):
        """Save a figure from pyplot.
            Parameters
            ----------
            path : string
                The path (and filename, without the extension) to save the
                figure to.
            ext : string (default='png')
                The file extension. This must be supported by the active
                matplotlib backend (see matplotlib.backends module).  Most
                backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.
            close : boolean (default=True)
                Whether to close the figure after saving.  If you want to save
                the figure multiple times (e.g., to multiple formats), you
                should NOT close it in between saves or you will have to
                re-plot it.
            verbose : boolean (default=True)
                Whether to print information about when and where the image
                has been saved.
            """
        # Extract the directory and filename from the given path
        #directory = os.path.split(path)[0]
        #filename = "%s.%s" % (os.path.split(path)[1], ext)
        if directory == '':
            directory = '.'
        filename = "%s.%s" % (name, ext)
        # If the directory does not exist, create it
        if not os.path.exists(directory):
            os.makedirs(directory)
        # The final path to save to
        savepath = os.path.join(directory, filename)
        if verbose:
            print("Saving figure to '%s'..." % savepath),
        # Actually save the figure
        if lib =='plt':
            plt.savefig(savepath)
        else:
            print('Please check your Plot library')
        # Close it
        if close:
            if lib == 'plt':
                plt.close()
        if verbose:
            print("Done")


    @classmethod
    def saveData(cls,totalList,directory,name,ext='csv',verbose=True):
        filename = "%s.%s" % (name, ext)
        if directory == '':
            directory = '.'
        # If the directory does not exist, create it
        if not os.path.exists(directory):
            os.makedirs(directory)
        # The final path to save to
        savepath = os.path.join(directory, filename)
        if verbose:
            print("Saving data to '%s'..." % savepath),
        data= [ [i, totalList[i]] for i in range(len(totalList))]
        with open(savepath, 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv.DictWriter(csvfile,fieldnames=["Iteration", "Value"]).writeheader()
            filewriter.writerows(data)
        csvfile.close()
        if verbose:
            print("Done")
