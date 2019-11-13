import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import colors
import re

from lspsuite.plots.cmaps import pastel_clear

def angular_electron(s, phi, e, fig=None, ropts=None, oap_deg=60, labels=None, ax=None, log_q=False, phi_spacing=180, cmap=pastel_clear, E_spacing=40, colorbar=True, maxE=4.0, maxQ=None, minQ=None, Estep=1.0, clabel=r'$Charge/MeV/rad (a.u.)$', ltitle=None, rtitle=None, title=None):
    '''
    Plot an angularly-resolved energy spectrum plot. Tailored specifically to case of backward going electrons, with the off-axis-paraboloid area marked.
    
    EDITED BY SCOTT FOR FUNCTIONAL PYTHON INPUT SYNTAX
    And also changed in other ways, e.g. no keV option, and adding a title.
    
    Arguments:
      s   -- the charges.
      phi -- the angles of ejection.
      e   -- energies of each charge.

    Keyword Arugments:
      max_e       -- Maximum energy.
      max_q       -- Maximum charge per angle/s.r. (max of cbar)
      angle_bins  -- Set the number of angle bins.
      radial_bins -- Set the number of radial (energy) bins.
      clabel      -- Set the colorbar label.
      colorbar    -- If true, plot the colorbar.
      e_step      -- Set the steps of the radius contours.
      labels      -- Set the angular labels.
      KeV         -- Use KeV isntead of MeV.
      fig         -- If set, use this figure, Otherwise,
                     make a new figure.
      ax          -- If set, use this axis. Otherwise,
                     make a new axis.
      ltitle      -- Make a plot on the top left.
      rtitle      -- Make a plot on the top right.
      log_q       -- log10 the charges.
      cmap        -- use the colormap cmap.
      rgridopts   -- pass a dictionary that sets details for the
                     rgrid labels.
      title        -- an actual title. Added by scott.
    '''
    phi_bins = np.linspace(-np.pi,np.pi,phi_spacing+1)
    E_bins   = np.linspace(0, maxE, E_spacing+1)
            
    PHI,E = np.mgrid[ -np.pi : np.pi : phi_spacing*1j,
                      0 : maxE : E_spacing*1j]
    S,_,_ = np.histogram2d(phi,e,bins=(phi_bins,E_bins),weights=s)
    fig = fig if fig else plt.figure(1,facecolor=(1,1,1))
    ax  = ax if ax else plt.subplot(projection='polar',axisbg='white')
    norm = matplotlib.colors.LogNorm() if log_q else None
    
    surf=plt.pcolormesh(PHI,E,S,norm=norm, cmap=cmap,vmin=minQ,vmax=maxQ)
    #making radial guides. rgrids only works for plt.polar calls
    full_phi = np.linspace(0.0,2*np.pi,100)
    for i in np.arange(0.0,maxE,Estep)[1:]:
        plt.plot(full_phi,np.ones(full_phi.shape)*i,
                 c='gray',alpha=0.9,
                 lw=1, ls='--')
    ax.set_theta_zero_location('N')
    ax.patch.set_alpha(0.0)
    ax.set_axis_bgcolor('red')
    #making rgrid
    if ropts:
        test = lambda d,k: k in d and d[k]

        if test(ropts, 'unit'):
            runit = ropts['unit']
        else:
            runit = 'MeV'
        if test(ropts, 'angle'):
            rangle = ropts['angle']
        else:
            rangle = 45
        if test(ropts, 'size'):
            rsize = ropts['size']
        else:
            rsize = 10.5
        if test(ropts, 'invert'):
            c1,c2 = "w","black"
        else:
            c1,c2 = "black","w"
    else:
        runit = 'MeV'
        rangle = 45
        rsize = 10.5
        c1,c2 = "black","w"
    rlabel_str = '{} ' + runit
    rlabels    = np.arange(0.0,maxE,Estep)[1:]
    _,ts=plt.rgrids(rlabels,
                    labels=map(rlabel_str.format,rlabels),
                    angle=rangle)
    for t in ts:
        t.set_path_effects([
            pe.Stroke(linewidth=1.5, foreground=c2),
            pe.Normal()
        ])
        t.set_size(rsize)
        t.set_color(c1)
    if oap_deg:
        oap = oap_deg/2 * np.pi/180
        maxt = oap+np.pi
        mint = np.pi-oap
        maxr  = maxE*.99
        minr=.12
        ths=np.linspace(mint, maxt, 20)
        rs =np.linspace(minr, maxr, 20)
        mkline = lambda a,b: plt.plot(a,b,c=(0.8,0.8,0.8),ls='-',alpha=0.5)
        mkline(ths, np.ones(ths.shape)*minr)
        mkline(mint*np.ones(ths.shape), rs)
        mkline(maxt*np.ones(ths.shape), rs)
    if labels:
        ax.set_xticks(np.pi/180*np.linspace(0,360,len(labels),endpoint=False))
        ax.set_xticklabels(labels)
    if colorbar:
        c=fig.colorbar(surf,pad=0.1)
        c.set_label(clabel)
    if ltitle:
        if len(ltitle) <= 4:
            ax.set_title(ltitle,loc='left',fontdict={'fontsize':28})
        else:
            ax.text(np.pi/4+0.145,maxE+Estep*2.5,ltitle,fontdict={'fontsize':28})
    if rtitle:
        if '\n' in rtitle:
            fig.text(0.60,0.875,rtitle,fontdict={'fontsize':22})
        else:
            plt.title(rtitle,loc='right',fontdict={'fontsize':22})
    if title:
        plt.suptitle(title, fontsize=20)
        plt.subplots_adjust(top=0.85)
    return (surf, ax, fig, (phi_bins, E_bins))