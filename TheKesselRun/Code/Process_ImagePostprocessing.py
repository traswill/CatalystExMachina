import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

def stitch_KDE():
    params = ['temperature',
     'Ru Loading',
     'Rh Loading',
     'Second Ionization Energy_wt-mad',
     'Second Ionization Energy_wt-mean',
     'Number d-shell Valence Electrons_wt-mean',
     'Number d-shell Valence Electrons_wt-mad',
     'Periodic Table Column_wt-mean',
     'Periodic Table Column_wt-mad',
     'Electronegativity_wt-mean',
     'Electronegativity_wt-mad',
     'Number Valence Electrons_wt-mean',
     'Number Valence Electrons_wt-mad',
     'Conductivity_wt-mean',
     'Conductivity_wt-mad',
     ]

    pths = glob.glob(r'C:\Users\quick\PycharmProjects\CatalystExMachina\TheKesselRun\Results\v34*\figures\*.png')

    fig1, ax1 = plt.subplots(nrows=2, ncols=3) # Conductivity
    fig2, ax2 = plt.subplots(nrows=2, ncols=3) # Electronegativity
    fig3, ax3 = plt.subplots(nrows=2, ncols=3) # N d-shell valence
    fig4, ax4 = plt.subplots(nrows=2, ncols=3) # 2nd Ion

    for pth in pths:
        if 'Conductivity' in pth:
            if 'mad' in pth:
                rw = 1
            else:
                rw = 0

            if 'ru_filter=1' in pth:
                col = 0
            elif 'ru_filter=2' in pth:
                col = 1
            else:
                col=2

            img = plt.imread(pth)
            plt.sca(ax=ax1[rw, col])
            plt.imshow(img)
            plt.axis('off')

        if 'Electronegativity' in pth:
            if 'mad' in pth:
                rw = 1
            else:
                rw = 0

            if 'ru_filter=1' in pth:
                col = 0
            elif 'ru_filter=2' in pth:
                col = 1
            else:
                col=2

            img = plt.imread(pth)
            plt.sca(ax=ax2[rw, col])
            plt.imshow(img)
            plt.axis('off')

        if 'Number d-shell Valence Electrons' in pth:
            if 'mad' in pth:
                rw = 1
            else:
                rw = 0

            if 'ru_filter=1' in pth:
                col = 0
            elif 'ru_filter=2' in pth:
                col = 1
            else:
                col = 2

            img = plt.imread(pth)
            plt.sca(ax=ax3[rw, col])
            plt.imshow(img)
            plt.axis('off')

        if 'Second Ionization Energy' in pth:
            if 'mad' in pth:
                rw = 1
            else:
                rw = 0

            if 'ru_filter=1' in pth:
                col = 0
            elif 'ru_filter=2' in pth:
                col = 1
            else:
                col = 2

            img = plt.imread(pth)
            plt.sca(ax=ax4[rw, col])
            plt.imshow(img)
            plt.axis('off')

    fig1.subplots_adjust(hspace=0, wspace=0)
    fig1.savefig('..\\Figures\\Conductivity.png', dpi=400)

    fig2.subplots_adjust(hspace=0, wspace=0)
    fig2.savefig('..\\Figures\\Electronegativity.png', dpi=400)

    fig3.subplots_adjust(hspace=0, wspace=0)
    fig3.savefig('..\\Figures\\ndshell.png', dpi=400)

    fig4.subplots_adjust(hspace=0, wspace=0)
    fig4.savefig('..\\Figures\\secondionenergy.png', dpi=400)

if __name__ == '__main__':
    stitch_KDE()
