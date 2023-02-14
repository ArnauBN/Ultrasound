# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 13:22:05 2023
Python version: Python 3.8

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def annot_position(p, fig, ax, annot_height=25, annot_width=100, margin=10):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi

    if p.lower() == 'northwest':
        return (0, height - annot_height - margin)
    elif p.lower() == 'northeast':
        return (width - annot_width - margin, height - annot_height - margin)
    elif p.lower() == 'southwest':
        return (0, margin)
    elif p.lower() == 'southeast':
        return (width - annot_width - margin, margin)
    elif p.lower() == 'north':
        return (int((width - annot_width)//2), height - annot_height - margin)
    elif p.lower() == 'south':
        return (int((width - annot_width)//2), margin)

def pltGUI(x, x4, CL, Cs, L, PE, TT, img):
    fig = plt.figure(figsize=(12,8))
    gs = fig.add_gridspec(4,3)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1:])
    ax2 = fig.add_subplot(gs[1, 1:])
    ax3 = fig.add_subplot(gs[2, 1:])
    ax4 = fig.add_subplot(gs[3, 1:])
    
    ax0.imshow(img, extent=[0, img.shape[1], x[-1], x[0]], aspect="auto")
    ax0.get_xaxis().set_ticks([])
    line1, = ax1.plot(x, CL, 'b')
    line2, = ax2.plot(x, Cs, 'r')
    line3, = ax3.plot(x, L)
    line4_PE, = ax4.plot(x4, PE[:,0])
    line4_TT, = ax4.plot(x4, TT[:,0])
    ax4.set_ylim([np.min([PE,TT])*1.1, np.max([PE,TT])*1.1])
    
    ax0.set_ylabel('Position (mm)')
    ax1.set_xlabel('Position (mm)')
    ax2.set_xlabel('Position (mm)')
    ax3.set_xlabel('Position (mm)')
    ax4.set_xlabel('Time (\u03bcs)')
    
    ax1.set_ylabel('Long. vel. (m/s)')
    ax2.set_ylabel('Shear vel. (m/s)')
    ax3.set_ylabel('Thickness (mm)')
    ax4.set_ylabel('Amplitude (V)')
    
    hline0, = ax0.plot([0, img.shape[1]], [x[0], x[0]], c='k')
    vline1 = ax1.axvline(x[0], c='k')
    vline2 = ax2.axvline(x[0], c='k')
    vline3 = ax3.axvline(x[0], c='k')
    
    annot = ax1.annotate(f"{round(x[0], 2)}, {round(CL[0], 2)}", xy=(x[0], CL[0]), xytext=annot_position('northwest', fig, ax1), textcoords="offset pixels",
                        bbox=dict(boxstyle="round", fc="w"))
    annot.set_visible(True)
    
    plt.tight_layout()
    
    def update(idx, l, shear):
        line1.set_marker('')
        line2.set_marker('')
        line3.set_marker('')
        
        x, y = l.get_data()
        xval, yval = round(x[idx], 2), round(y[idx], 2)
        
        # annot.xy = (x[idx], y[idx])
        text = f"{xval}, {yval}"
    
        annot.set_text(text)
        
        hline0.set_ydata(x[idx])
        vline1.set_xdata(x[idx])
        vline2.set_xdata(x[idx])
        vline3.set_xdata(x[idx])
        
        if shear:
            auxPE = PE[:,::-1]
            auxTT = TT[:,::-1]
            line4_PE.set_ydata(auxPE[:,idx])
            line4_TT.set_ydata(auxTT[:,idx])
        else:
            line4_PE.set_ydata(PE[:,idx])
            line4_TT.set_ydata(TT[:,idx])
        
        l.set_marker('.')
        l.set_markerfacecolor('k')
        l.set_markeredgecolor('k')
        l.set_markevery([idx])
    
    
    def hover(event):
        shear = False
        if event.inaxes == ax1:
            l = line1
        elif event.inaxes == ax2:
            l = line2
            shear = True
        elif event.inaxes == ax3:
            l = line3
        else:
            return
        cont, ind = l.contains(event)
        if cont:
            if len(ind["ind"]) != 0:
                update(ind["ind"][0], l, shear)
            fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect("motion_notify_event", hover)
    
    plt.show()

if __name__ == '__main__':
    x1 = np.sort(np.random.rand(15))
    x4 = x1
    y1 = np.sort(np.random.rand(15))*2
    y2 = -y1
    y3 = np.sort(np.random.rand(15))*3
    y4 = np.random.rand(len(x1), len(y1))
    y4_2 = np.random.rand(len(x1), len(y1))
    img = mpimg.imread(r'G:\Unidades compartidas\Proyecto Cianocrilatos\Data\Scanner\Methacrylate\methacrylate_photo.jpg')

    pltGUI(x1, x4, y1, y2, y3, y4, y4_2, img)