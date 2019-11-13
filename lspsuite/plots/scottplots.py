# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 19:05:18 2016

@author: Scott
"""

import sys
import numpy as np # local
import lstools as ls # local
import lspreader2 as rd # local
from special import tlmb # local

# Matplotlib stuff
import matplotlib as mpl
#mpl.use('Agg') # Let's call this at an earlier point, shall we?
import matplotlib.pyplot as plt
from distutils.version import LooseVersion
if LooseVersion(mpl.__version__) < LooseVersion('1.5.0'):    
    # http://stackoverflow.com/questions/11887762/how-to-compare-version-style-strings and 
    print "Matplotlib", mpl.__version__, "might not have colormap 'viridis'. Importing from local colormaps.py."
    import colormaps as cmaps
    plt.register_cmap(name='viridis', cmap=cmaps.viridis)
    plt.register_cmap(name='inferno', cmap=cmaps.inferno)
    plt.register_cmap(name='magma', cmap=cmaps.magma)
    plt.register_cmap(name='plasma', cmap=cmaps.plasma)

import os
import sftools as sf # local
import scipy.constants as sc

def addCrit(ax, edens, xgv_um, zgv_um):
    """ Add the critical and quarter-critical density contours over top of current figure axis. Not the best function, but works for now.
    Inputs:    
        ax: axis handle
        edens: electron density NumPy array
        xgv_um: x grid vector, in microns
        zgv_um: z grid vector, in microns
    Outputs:
        Modifies axis 'ax' in place by adding critical density contours to the plot.
    """
    crit = 1.742e21 # critical density (800 nm), electrons per cc
    qcrit = crit/4.0
    
    C = edens
    X, Z = np.meshgrid(xgv_um, zgv_um)
    CS = ax.contour(X, Z, C, [qcrit, crit], linewidths=1.0, colors=('g','w','black'))
    
    fmt = {}
    strs = ['$n_{cr}/4$', '$n_{cr}$', '$n_{s}$']
    for l, s in zip(CS.levels, strs):
        fmt[l] = s
    
    # Label every other level using strings
    ax.clabel(CS, CS.levels, inline_spacing = 10, inline=True, fmt=fmt, fontsize=15, rightside_up=True, manual=[(-20, 13), (-6.2, -10)])
    return CS
    
def addQuiv(ax, vec, xgv, zgv, divsp = 8):
    """ Add a quiver plot of a vector overtop a figure axis.
    Inputs:    
        ax: axis handle
        vec: NumPy array, a vector with three components at every point in 2D (Space x Space x Vector)
        xgv_um: x grid vector, in microns
        zgv_um: z grid vector, in microns
    Outputs:
        Modifies axis 'ax' in place by adding quiver arrows, normalized to themselves, over the plot.
    """
    
    myvecmag = sf.vecMag(vec)
    norm = np.max(myvecmag)
    Vx = vec[::divsp,::divsp,0]/norm
    Vz = vec[::divsp,::divsp,2]/norm
    X, Z = np.meshgrid(xgv[::divsp],zgv[::divsp])
    qu = ax.quiver(X, Z, Vx, Vz, pivot='mid', units='xy', scale_units='xy', scale=0.3, color='white')
    return qu

def mypcolor(C, xgv, zgv, cmin = 0,  cmax = None, title='', tstring = '', clabel = '', fld_id = '', sticker ='', rfooter = '', cmap='viridis', edens = np.zeros(0), vec = np.zeros(0), fig=None, color='white'):
    """ A custom-tailored wrapper for pcolorfast(). Somewhat general, meant for any 2D sim colorplot.
    Inputs:
        C: 2D NumPy array, the data to be visualized
        xgv: x dimension grid vector (1D), in microns
        zgv: z dimension grid vector (1D), in microns
        ...
        edens: 2D NumPy array, electron density data for adding critical density contours. If omitted, no contours are added to plot.
    """    
    # xgv, ygv should be in microns
    
    if not cmax:
        cmax = np.max(C)
    if not fig:
        fig = plt.figure()
    plt.clf() # Clear the figure
    ax = plt.subplot(111)
    xr = [xgv[0],xgv[-1]] # min and max of xgv, needed for pcolorfast
    zr = [zgv[0],zgv[-1]]
    im = ax.pcolorfast(xr, zr, C, cmap=cmap)
    ax.set_xlabel(r'X ($\mu m$)')
    ax.set_ylabel(r'Z ($\mu m$)')
    ax.set_title(title, fontsize=20)
    ax.text(0.05, 0.95, tstring, fontsize=24, color=color, transform=ax.transAxes, horizontalalignment='left', verticalalignment='top') # Upper left within axis (transform=ax.transAxes sets it into axis units 0 to 1)
    ax.text(0.95, 0.03, fld_id, fontsize=44, color=color, transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom') # Lower right within axis
    ax.text(0.05, 0.03, sticker, fontsize=44, color=color, transform=ax.transAxes, horizontalalignment='left', verticalalignment='bottom') # Lower left within axis
    cbar = fig.colorbar(im, label=clabel)
    im.set_clim(vmin=cmin, vmax=cmax)
    if len(edens) > 1: # Did the user specify an electron density array?
        addCrit(ax, edens, xgv, zgv) # Highlight the critical density surface
    if len(vec) > 1: # Did the user specify a poynting vector to plot overtop as quivers?
        addQuiv(ax, vec, xgv, zgv)
    fig.text(0.99, 0.01, rfooter, horizontalalignment='right') # Lower right in figure units
    return fig

def emFFT(data, kind = "EB", trange = (60, 1.0e9)):
    """ Outputs the energy breakdowns
    Inputs:
        data: the usual data dict, with enough frames to resolve the desired frequency characteristics. Must contain all three components of electric and magnetic fields
        kind: (optional) string, one of three strings: "EB" for all components together, "E" for just electric field, and "B" for just magnetic. If kind = "Backscat", performs analysis over only the X index 0, and only over certain times.
        trange: (applies only when kind = "Backscat") A two-element tuple of min and max times for the analysis in femtoseconds. This allows you to cut over just the times you want. E.g. trange = (0, 100) should catch the main pulse, while trange = (100, 1.0e9) should catch the rest of the sim (backreflection). trange[0] <= t < trange[1]
    Outputs:
        maps: dict containing multiple 2D arrays, energy density cut by frequency, in units of "J/m^3"
        cuts: dict with same keys as maps, but containing the frequency cuts (in units of the fundamental) corresponding to maps
        pwrsum: 1D array, power spectrum Y, units are J/m/(freq. fund)
        freq: 1D array, power spectrum X, units are frequency (in units of the fundamental)
        df: number, the frequency step size (in units of the fundamental)
        nframes: number, the number of time steps analyzed for the analysis
    """

    
    # Assume equal spacing in time and space, and get the deltas
    dt = np.mean(np.diff(data['times']))*1e-9 # dt in seconds
    dx = np.mean(np.diff(data['xgv']))*1e-2 # dx in m
    dz = np.mean(np.diff(data['zgv']))*1e-2 # dz in m
    
    
    ## Build EMvector to analyze (time x space x space x component)
    # Field energy
    # LSP unit conversion to SI
    tesla = 1e-4 # LSP to SI conversion. 'X Gauss' * tesla = 'Y Tesla'
    vpm = 1e5 # LSP to SI conversion. 'X kV/cm' * vpm = 'Y V/m'

    eps = sc.epsilon_0 # Call the dielectric constant equal to vacuum permittivity, which is wrong within a plasma
    mu = sc.mu_0 # Call the magnetic permeability the vacuum permeability, which is wrong within a plasma

    Efact = vpm * np.sqrt(0.5*eps) # Factor defined by (data['Ex']*Efact)**2 => Electric field energy density in J/m
    Bfact = tesla * np.sqrt(0.5/mu) # Factor defined by (data['Bx']*Bfact)**2 => Magnetic field energy density in J/m
    if kind == "E": # Electric field energy analysis
        EMvec = np.stack((data['Ex'], data['Ey'], data['Ez']), axis=-1) * Efact  # Give the EM vector 3 electric field components as its last axis
    elif kind == "B": # Magnetic field energy analyis
        EMvec = np.stack((data['Bx'], data['By'], data['Bz']), axis=-1) * Bfact # Give the EM vector 3 magnetic field components as its last axis
    elif kind == "EB": # Combined electric and magnetic field energy analysis
        EMvec = np.stack((data['Ex'] * Efact, data['Ey'] * Efact, data['Ez'] * Efact, data['Bx'] * Bfact, data['By'] * Bfact, data['Bz'] * Bfact), axis=-1) # Give the EM vector all 6 components
    elif kind == "Backscat": # Backscatter analyisis. Extract from the X-coordinate index 0 only, and over specific times only.
        tcdt = np.logical_and(data['times'] >= trange[0]*1e-6, data['times'] < trange[1]*1e-6) # Condition on times
        if not np.any(tcdt):
            raise Exception("There aren't any values in the backscatter analysis time range to analyze.")
        xidx = int(np.floor(2.0e-6/dx)) # Index of lineout extraction, along the x dimension. Place 2 microns away from edge. (e.g. xidx = 8, such that xgv[xidx] = 8 microns)
        EMvec = np.stack((data['Ex'][tcdt,:,xidx:(xidx+1)] * Efact, data['Ey'][tcdt,:,xidx:(xidx+1)] * Efact, data['Ez'][tcdt,:,xidx:(xidx+1)] * Efact, data['Bx'][tcdt,:,xidx:(xidx+1)] * Bfact, data['By'][tcdt,:,xidx:(xidx+1)] * Bfact, data['Bz'][tcdt,:,xidx:(xidx+1)] * Bfact), axis=-1) # Give the EM vector all 6 components, but only at specific times and at a single X coordinate
    else:
        raise Exception("Invalid field kind specificied at input of function.")
    
    ## Assess simulation parameters
    nframes = EMvec.shape[0] # Number of frames (times) to be analyzed

    Jvecmean = np.mean(EMvec**2, 0) # Average the field energy density, for each component, along the time axis (axis 0). Now (space x space x vector component)
    Jvectot = np.sum(Jvecmean, axis=(0,1))*dx*dz # 1D array of total field energy per component, in Joules/m per component; Array dims: (component)
 
    # Calculate the frequency of the laser from its wavelength (800 nm)
    wl = 0.8e-6 # Wavelength of laser in m
    fr_fund = sc.c/wl # Laser frequency (the 'fundamental'), in Hz (Freq = Speed of light / wavelength)
    
    ## Break down by energy presence by frequency
    freqHz = np.fft.rfftfreq(nframes, d = dt) # Makes equally spaced frequency bins
    freq = freqHz/fr_fund # Re-define frequency from Hz to units of the laser frequency
    df = np.mean(np.diff(freq)) # Get the frequency step df

    EMvecft = np.fft.rfft(EMvec, axis=0) # Perform real frequency Fourier transform along time axis (axis 0)
    pwrvec = np.absolute(EMvecft)**2 # Non-normalized value (freq x space x space x vector component)

    pwrvectot = np.sum(pwrvec, axis=(0,1,2))*df*dx*dz # Integral across space and time, for each vector component. 1D Arr dims: (component)
    for i in range(pwrvec.shape[3]): # Iterate over vector components
        pwrvec[:,:,:,i] = pwrvec[:,:,:,i] * (Jvectot[i] / pwrvectot[i]) # Normalized-to-J power. Each component integrates, over all frequencies and space, to the total energy (J/m) of that component
    # Now, each vector component of pwrvec integrates, over all frequencies and space, to the energy of that component
    # In other words, pwrvec is still (freq x space x space x vector), but an integral over all space, frequencies, and components gives the total field energy (J/m)
    
    print "Powervec shape:", pwrvec.shape
    
    cuts = {}
    cuts['1_0'] = (0.9, 1.1) # Frequency min and max for integration, in units of the fundamental. min <= freq < max
    cuts['0_5'] = (0.3, 0.7)
    cuts['1_5'] = (1.3, 1.7)
    cuts['2_0'] = (1.8, 2.3)
    cuts['0_8'] = (0.7, 0.9)
    cuts['0'] = (0, 0.2)
    cuts['all'] = (0, 1.0e9)
    
    maps = {}
    for k in cuts.keys(): # Make maps
        cdt = np.logical_and(freq >= cuts[k][0], freq < cuts[k][1])
        maps[k] = np.sum(pwrvec[cdt], axis=(0,3))*df # Integrate over frequency, and add up the contributions from all vector components
        # maps is a 2D array. (space x space), in units of J/m^3

    print "Map shape:", maps['1_5'].shape

    pwrsum = np.sum(pwrvec, axis=(1,2,3))*dx*dz # 1D array, (freq), units are J/m/(freq. fund)
    
    print "Pwrsum shape:", pwrsum.shape
    return maps, cuts, pwrsum, freq, df, nframes


def poyntAnlz(data):
    """ Perform a poynting analysis on a stack of data frames
    Inputs:
        data: the usual data dict; must contain all of the E and B components.
    Outputs:
        Svecmean: array, Full Poynting vector (Space x Space x Vector), averaged across time steps (J/m^2)
        Smagmean: array, Magnitude of Poynting vector (Space x Space), averaged across time steps (J/m^2)
        JEmean: array, Electric field energy density (Space x Space), averaged across time steps (J/m^3)
        JBmean: array, Magnetic field energy density (Space x Space), averaged across time steps (J/m^3)
        Jtotal: number, the total of all electric and magnetic field energy (J/m), averaged across time steps
        """
    # Field energy
    # LSP unit conversion to SI
    tesla = 1e-4 # LSP to SI conversion. 'X Gauss' * tesla = 'Y Tesla'
    vpm = 1e5 # LSP to SI conversion. 'X kV/cm' * vpm = 'Y V/m'
    
    # Poynting vector info: http://hyperphysics.phy-astr.gsu.edu/hbase/waves/emwv.html#c2
    # Energy of electric and magnetic fields: http://hyperphysics.phy-astr.gsu.edu/hbase/electric/engfie.html
    
    Evec = np.stack((data['Ex'], data['Ey'], data['Ez']), axis=-1) * vpm # Electric field vector, in V/m
    Emag = sf.vecMag(Evec)
    Bvec = np.stack((data['Bx'], data['By'], data['Bz']), axis=-1) * tesla # Magnetic field vectors, in Tesla
    Bmag = sf.vecMag(Bvec)
    Svec = (1/sc.mu_0)*np.cross(Evec,Bvec) # Poynting vector, in J/m^2
    Smag = sf.vecMag(Svec)
        
    Svecmean = np.mean(Svec, axis=0) # Mean of the Poynting vector (all components), in J/m^2
    Smagmean = np.mean(Smag, axis=0) # Mean of the Poynting vector's magnitude, in J/m^2
    
    eps = sc.epsilon_0 # Call the dielectric constant equal to vacuum permittivity, which is wrong within a plasma
    mu = sc.mu_0 # Call the magnetic permeability the vacuum permeability, which is wrong within a plasma
    JE = (0.5*eps)*Emag**2 # Electric field energy density, Joule / m^3
    JB = (0.5/mu)*Bmag**2 # Magnetic field energy density, Joule / m^3
    JEmean = np.mean(JE, 0) # Mean Electric field energy density, in J/m^3
    JBmean = np.mean(JB, 0) # Mean Magnetic field energy density, in J/m^3
    
    dx = np.mean(np.diff(data['xgv']))*1e-2 # dx in m
    dz = np.mean(np.diff(data['zgv']))*1e-2 # dz in m
    dA = dx*dz # Area of a cell, in m^2
    Jtotal = np.sum(JEmean + JBmean)*dA # Total energy, in J/m  (Add up each (J/m^3) times cell area = Joules in cell, over all cells)
    print "Simulation total energy: ", Jtotal*1e-3, "mJ / micron."

    return Svecmean, Smagmean, JEmean, JBmean, Jtotal

def timeStrings(data, alltime=False):
    """ Subroutine that outputs a nice string and label for input list of times
    Inputs:
        data: the usual data dict
        alltime: bool, should this be labeled in a special way to indicate it is an analysis of all time steps, rather than a subset?
    Outputs:
        tstring: string, # '00512' for mean time = 51.2342 fs, or "00000" if alltime=True
        tlabel: string, "t = XX fs +/- 20 fs", or "All times" if alltime=True
    """
    times = data['times']*1e6 # times in femtosecond

    # This timestring computation was copied and pasted from freqanalysis.plotme()
    if alltime: # If this is the all-time rather than time-resolved analysis, put a different label on plot
        maxtime = np.max(times)
        mintime = np.min(times)
        #tstring = r"All time ($\Delta$t =" + "{:.0f}".format(maxtime - mintime) + ' fs)'
        tstring = r"All time (" + "{:.0f}".format(maxtime - mintime) + ' fs total)' # To lay onto the plot
        tlabel = ''.zfill(5) # If this is the "All times" analysis, label the file with "00000.*"
    else: # Otherwise, put the standard "t = XX fs +/- 20 fs" label.
        meantime = np.mean(times)
        tplus = np.max(times) - np.mean(times)
        tstring = 't=' + "{:.1f}".format(meantime) + " fs $\pm$ " + "{:.1f}".format(tplus) + " fs"
        tlabel = "{:.0f}".format(round(meantime*10)).zfill(5) # Make the file label be '00512.*' for t = 51.2342 fs
    return tstring, tlabel

def plotDens(data, outdir='.', shortname = '', alltime=False):
    """ Make Scott's set of custom density-related plots for this batch. """
    xgv = data['xgv']*1e4 # x values in microns
    zgv = data['zgv']*1e4
    dx = np.mean(np.diff(xgv))# dx in microns
    dz = np.mean(np.diff(zgv))
    
    ## CALCULATIONS
    # Mean electron density
    edens = np.mean(data['RhoN10'],0)

    # Mean ion density
    pdens = np.mean(data['RhoN11'],0)
    
    # Mean oxygen ionization state
    old_settings = np.seterr(divide='ignore', invalid='ignore') # Set to ignore divide by zero error. (We will divide by zero where no ions exist)
    ionstate = np.mean((0*data['RhoN1'] + 1*data['RhoN2'] + 2*data['RhoN3'] + 3*data['RhoN4'] + 4*data['RhoN5'] + 5*data['RhoN6'] + 6*data['RhoN7'] + 7*data['RhoN8'] + 8*data['RhoN9'])/(data['RhoN1'] + data['RhoN2'] + data['RhoN3'] + data['RhoN4'] + data['RhoN5'] + data['RhoN6'] + data['RhoN7'] + data['RhoN8'] + data['RhoN9']), 0)
    ionstate = np.nan_to_num(ionstate)
    np.seterr(**old_settings)  # reset divide by zero error to prior settings

    ## MAKE AND SAVE FIGURES
    pltdir = outdir
    tstring, tlabel = timeStrings(data, alltime=alltime) # Custom strings for labeling figures, based on the list of times in this dataset

    ## Plot 1: Electron density
    C = edens
    sticker = '$e^-$'
    title = 'Electron density'
    clabel = 'Density (number/cc)'
    fld_id = r'$\rho$'
    cmax = 3e21
    fig = mypcolor(C, xgv, zgv, cmin=0,  cmax=cmax, title=title, tstring=tstring, clabel=clabel, fld_id=fld_id, sticker=sticker, rfooter=shortname, edens=edens)
    fig.savefig(os.path.join(sf.subdir(pltdir, 'Electron density'), tlabel + '.png')) # Save into a subdirectory of 'outdir'
    
    ## Plot 2: Proton density
    C = pdens
    sticker = '$p^+$'
    title = 'Proton density'
    clabel = 'Density (number/cc)'
    fld_id = r'$\rho$'
    cmax = 3e21*.67
    fig = mypcolor(C, xgv, zgv, cmin=0,  cmax=cmax, title=title, tstring=tstring, clabel=clabel, fld_id=fld_id, sticker=sticker, rfooter=shortname, edens=edens)
    fig.savefig(os.path.join(sf.subdir(pltdir, 'Proton density'), tlabel + '.png')) # Save into a subdirectory of 'outdir'

    ## Plot 3: Oxygen, mean ionization
    C = ionstate
    sticker = '$O$'
    title = 'Ionization level of oxygen'
    clabel = 'Mean ionization state'
    fld_id = r'+'
    cmin = 0
    cmax = 7
    fig = mypcolor(C, xgv, zgv, cmin=cmin,  cmax=cmax, title=title, tstring=tstring, clabel=clabel, fld_id=fld_id, sticker=sticker, rfooter=shortname, edens=edens, cmap='inferno')
    fig.savefig(os.path.join(sf.subdir(pltdir, 'Oxygen ionization'), tlabel + '.png'))# Save into a subdirectory of 'outdir'

     ## Close the plots
    plt.close('all')


def bsFFT(data, trange = (80, 1.0e9)):
    """ A wrapper for emFFT() for the case of Backscatter analysis. Outputs are lines (rather than maps) and are in plotting units.
    Inputs:
        data: the usual data dict
        trange: (fs) the min and max time to look at for this "summing energy through the plane" analysis
    Outputs:
        lines: dict containing multiple 1D arrays, total energy as a function of Z position, cut by frequency, in units of "mJ/um^2"
        cuts: dict with same keys as lines, but containing the frequency cuts (in units of the fundamental) corresponding to lines
        pwrsum: 1D array, power spectrum Y, units are mJ/um/(freq. fund)
        freq: 1D array, power spectrum X, units are frequency (in units of the fundamental)
        Utots: dict with same keys as lines, but with each key giving a single number (the total energy at that frequency, in mJ/um)
    """
    # Assume equal spacing in time and space, and get the deltas
    dt = np.mean(np.diff(data['times']))*1e-9 # dt in seconds
    dx = np.mean(np.diff(data['xgv']))*1e-2 # dx in m
    dz = np.mean(np.diff(data['zgv']))*1e-2 # dx in m

    # Do the Fourier analysis
    maps, cuts, pwrsum, freq, df, nframes = emFFT(data, kind='Backscat', trange=trange)
    
    # All these energies returned by emFFT are the mean energy value, across the time period; if we want the sum, we need to multiply by number of frames analyzed
    # But there is now another problem. We have overcounted the energy by a factor of (dx*c/dt), because while dx should be c/dt long, it is actually larger and a delta pulse of light gets counted multiple frames in a row.
    pwrline = pwrsum*(nframes * sc.c * dt/dx) # Units are now J per y meter per freq. (integrated rather than averaged over all time)
    lines = {}
    for k in maps:
        lines[k] = np.squeeze(maps[k]*(nframes * sc.c * dt/dx)) # Units are now Joules per y meter per z meter (integrated rather than averaged over all time). Not really, I can't figure out what's wrong though.
    
    # That's better!

    print lines['0'].shape
    
    
    print "Total energy, in mJ/micron:", np.sum(lines['all'])*dx*dz*1e-3 
    print "Total energy, in mJ/micron:", np.sum(pwrline)*df*1e-3
    
    ## UNIT CONVERSION FROM SI to PLOTTING UNITS (mJ/micron, etc.)
    for k in lines:
        lines[k] = lines[k]*dx*1e-9 # units of lines (Energy in frequencies / Z axis) are now in mJ / z micron / y micron
    pwrline = pwrline * 1e-3 # pwrline (Power spectrum Y axis) is now in of mJ / freq. unit / y micron
    
    dt = dt * 1e15 # dt is now in fs
    dx = dx * 1e6 # dx is now in microns
    dz = dz * 1e6 # dz is now in microns
    
    Utots = {}
    for k in lines:
        Utots[k] = np.sum(lines[k])*dz # Total energy in the frequency band, in mJ/ y micron
    return lines, cuts, pwrline, freq, Utots

def wlenPlot(freq, Jfreq, wlen_fund=800):
    """ Make the backscatter wavelength plot from data for a frequency plot. Don't save to file, just return the handle.
    Inputs:
        freq: 1D array, list of frequencies (in units of the fundamental) (the X axis for a power spectrum plot)
        Jfreq: 1D array, same length as freq, list of mJ/freq./um at those frequencies (the Y axis for a power spectrum plot)
        wlen_fund: (optional) number, wavelength of the fundamental, in nm
    Outputs:
        fig: handle to the plot where X axis is wavelength (in nm), and Y axis is Joules/wlen. (which can be saved to file, e.g. with fig.savefig())
    """
    ## Make a chop above 0.1 (because below that is essentially electrostatic), and then transform the array
    chp = (freq > 0.1) # Chopping condition
    wlen = (1/freq[chp] * wlen_fund)[::-1] # wlen = 1 / frequency. The -1 is to reverse the arrays, since wavelength and frequency are reversed
    Jwlen = (freq[chp]**2 * Jfreq[chp] / wlen_fund)[::-1] # Adjust the Y axis, based on the fact that dwlen/dfreq = -1/freq^2
    
    ## Adjust by the TLMB transmission
    T_tlmb = tlmb.trans(wlen)
    Jwlen2 = Jwlen * T_tlmb
    
    fig = plt.figure(2)
    plt.clf()
    ymin = 0
    ymax = np.max(Jwlen2[(wlen > 100) & (wlen < 600)])
    ax = plt.subplot(2,1,1)
    ax.set_title('Blue-green reflected (w/ TLMB)')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('$mJ / optical nm / \mu m$')
    ax.plot(wlen, Jwlen2, 'b')
    ax.set_xlim(200, 900)
    ax.vlines([wlen_fund*2, wlen_fund, wlen_fund * 2./3., wlen_fund / 2.0], ymin, ymax, colors=[(0.6,0,0),'r','g','b'], linestyle=':') # vertical lines, colored by frequency
    ax.set_ylim(0, ymax)
    
    ax = plt.subplot(2,1,2)
    ymin = 0
    ymax = np.max(Jwlen2[(wlen > 950) & (wlen < 1700)])
    ax.set_title('Mid-IR reflected (w/ TLMB)')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('$mJ / optical nm / \mu m$')
    ax.plot(wlen, Jwlen2, 'b')
    ax.set_xlim(900, 1800)
    ax.vlines([wlen_fund*2, wlen_fund, wlen_fund * 2./3., wlen_fund / 2.0], ymin, ymax, colors=[(0.6,0,0),'r','g','b'], linestyle=':') # vertical lines, colored by frequency
    ax.set_ylim(ymin, ymax)
    plt.tight_layout()
    
    return fig

def plotBScat(data, tcut=80, outdir='.', shortname = ''):
    """ Plot relevant backscattered light plots. Energy calculation makes some iffy assumptions about angles of light, so some degree of systematic error.
    Inputs:        
        data: the usual data dict
        tcut: number, a time (fs) that splits forward light from backward. I.e. after this time (in fs), we consider it backscattered light. Before this time, we consider it incident light.
        outdir: directory where PNG plots will be saved
        shortname: a short string describing the run, e.g. 'a40f-14_so'
    Outputs:
        Saves PNG plots into a subdirectory of outdir.
    """
    linesT, _, pwrlineT, freqT, U_T = bsFFT(data, trange=(0,tcut)) # Input light (T)ransmission characteristics
    linesR, cuts, pwrlineR, freqR, U_R = bsFFT(data, trange=(tcut,1.0e9)) # Backscatter (R)eflection characteristics
    
    zgv = data['zgv']*1e4 # Z grid vector, in microns
    
    R = U_R['all']/U_T['all'] # Reflectivity, roughly
    print "Reflectivity: " + str(np.round(R*100, 2)) + "%"
    Rstring = '(R = ' + str(np.round(R*100,2)) + '%)'
    
    ## CSV: Write some reflectivity info to file.
    with open(os.path.join(outdir, shortname + ' - Reflectivity.csv'), 'w') as f:
        f.write("FREQ_BAND, PERCENT_ENERGY_IN_REFLECTED, FREQ_CUT1, FREQ_CUT2, TRANSMITTED_MILLIJOULE_PER_MICRON, REFLECTED_MILLIJOULE_PER_MICRON\n")        
        for k in U_T:
            f.write(k + ", " + str(np.round(U_R[k]/U_T['all']*100, 4)) + ", " + str(cuts[k][0]) + ", " + str(cuts[k][1]) + ", " + str(U_T[k]) + ", " + str(U_R[k]) + "\n")

    ## Plot 0: Backscattered light through TLMB
    fig = wlenPlot(freqR, pwrlineR, wlen_fund=800)
    fig.text(0.99, 0.01, shortname, horizontalalignment='right')
    fig.savefig(os.path.join(outdir, shortname + ' - BScat Wavelengths.png'))

    ## Plot 1: Backscatter power spectrum
    title = 'Backscattered light power spectrum ' + Rstring
    fld_id = r'$L$'
    fig = plt.figure(1)
    plt.clf() # Clear the figure
    ymax = 100
    ymin = 10**(np.log10(ymax) - 4) # Give 4 orders of magnitude
    xmin = 0
    xmax = 2.5
    ax = plt.subplot(111)
    ax.plot(freqT, pwrlineT, '-.k', label='T')
    ax.plot(freqR, pwrlineR, '-k', label='R')
    ax.set_xlabel('Angular frequency / $\omega_{laser}$', fontsize=20)
    ax.set_ylabel('Energy passing plane ($mJ / \omega_{laser} / \mu m$)')
    ax.legend(ncol=1, loc=1, borderaxespad=0.)
    ax.set_yscale('log')
    plt.xticks(fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xlim(xmin, xmax)
    ax.vlines([0.5,1.0,1.5,2.0], -1.0e30, 1.0e30, colors=[(0.6,0,0),'r','g','b'], linestyle=':') # vertical lines, colored by frequency
    ax.yaxis.grid(which='minor', color='0.92', linestyle='--') # horizontal lines
    ax.yaxis.grid(which='major', color='0.92', linestyle='-') # horizontal lines
    ax.set_axisbelow(True) # Move the grid marks under the axes
    ax.text(0.9 ,0.05, fld_id, fontsize=44, color='black', transform=ax.transAxes) # http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes.text
    ax.set_ylim(ymin, ymax)
    fig.text(0.99, 0.01, shortname, horizontalalignment='right')
    fig.savefig(os.path.join(outdir, shortname + ' - BScat Spectrum.png'))

    ## Plot 2: Backscatter power spectrum
    title = 'Backscattered light vs. Z axis ' + Rstring
    fld_id = r'$L$'
    fig = plt.figure(2)
    plt.clf() # Clear the figure
    ymax = 1
    ymin = 10**(np.log10(ymax) - 4) # Give 4 orders of magnitude
    xmin = np.min(zgv)
    xmax = np.max(zgv)
    ax = plt.subplot(111)
    ax.plot(zgv, linesT['all'], '-.k', label='T')
    ax.plot(zgv, linesR['all'], '-k', label='R')
    ax.plot(zgv, linesR['0_5'], color=(0.6,0,0), label='$\omega/2$')
    ax.plot(zgv, linesR['0_8'], '-.r', label='$.8\omega$')
    ax.plot(zgv, linesR['1_0'], 'r', label='$\omega$')
    ax.plot(zgv, linesR['1_5'], 'g', label='$3\omega/2$')
    ax.plot(zgv, linesR['2_0'], 'b', label='$2\omega$')
    ax.legend(ncol=1, loc=1, borderaxespad=0.)
    ax.set_xlabel('Z ($\mu m$)', fontsize=20)
    ax.set_ylabel('Energy at Z position ($mJ / \mu m^2$)')
    ax.set_yscale('log')
    ax.set_title(title, fontsize=16)
    ax.xaxis.grid(which='both', color='0.92', linestyle='-') # vertical lines
    ax.yaxis.grid(which='both', color='0.87', linestyle='--') # horizontal lines
    ax.set_axisbelow(True) # Move the grid marks under the axes
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)
    ax.text(0.9 ,0.05, fld_id, fontsize=44, color='black', transform=ax.transAxes)
    fig.text(0.99, 0.01, shortname, horizontalalignment='right')
    fig.savefig(os.path.join(outdir, shortname + ' - BScat vs Z.png'))
    
    plt.close('all') # Close all the figures


  
def plotEM(data, outdir='.', shortname = '', alltime=False):
    """ Make Scott's set of custom electromagnetic field (and frequency) plots for this batch """
    xgv = data['xgv']*1e4 # x values in microns
    zgv = data['zgv']*1e4
    dx = np.mean(np.diff(xgv)) # dx in microns
    dz = np.mean(np.diff(zgv))

    ## CALCULATIONS
    # Mean electron density (used for critical density contours)
    edens = np.mean(data['RhoN10'],0)

    # Poynting vector analysis
    Svec, Smag, JE, JB, Jtot = poyntAnlz(data) # Units are Joules and meters

    # Frequency analysis
    maps, cuts, pwrsum, freq, df, _ = emFFT(data, kind='EB')
    
    # Compare the many ways of calculating energy, in mJ/ um
    print "SUMFFTmap", np.sum(maps['all'])*dx*dz*1e-15
    print "SUMFFTpwr", np.sum(pwrsum) * df * 1e-3
    print "SUMpoyntJEB", np.sum(JE + JB)*dx*dz*1e-15
    print "SUMpoyntJtot", Jtot*1e-3

    ## MAKE AND SAVE FIGURES
    pltdir = outdir
    tstring, tlabel = timeStrings(data, alltime=alltime) # Custom strings for labeling figures, based on the list of times in this dataset

    ## Plot 1: EM energy density, all frequencies
    C = maps['all']*1e-12 # Convert from J/m^3 to uJ/um^3  ; Conversion is 10^6 * 10^-18 => 10^-12
    sticker = r'EM'
    title = 'EM field energy: ' + str(np.round(np.sum(C)*dx*dz*1e-3, 2)) + " $mJ/\mu m$"
    clabel = 'Energy density ($\mu J/\mu m^3$)'
    fld_id = r'$J$'
    cmax = None if alltime else 150
    fig = mypcolor(C, xgv, zgv, cmin=0,  cmax=cmax, title=title, tstring=tstring, clabel=clabel, fld_id=fld_id, sticker=sticker, rfooter=shortname, edens=edens, cmap='inferno', vec=Svec)
    fig.savefig(os.path.join(sf.subdir(pltdir, 'EM All Energy'), tlabel + '.png'))# Save into a subdirectory of 'outdir'
    
    ## Plot 2: EM energy density, omega
    C = maps['1_0']*1e-12 # Convert from J/m^3 to uJ/um^3  ; Conversion is 10^6 * 10^-18 => 10^-12
    sticker = r'$\omega$'
    title = 'EM field energy: ' + str(np.round(np.sum(C)*dx*dz, 1)) + " $\mu J/\mu m$"
    clabel = 'Energy density ($\mu J/\mu m^3$)'
    fld_id = r'$J$'
    cmax = None if alltime else 90
    fig = mypcolor(C, xgv, zgv, cmin=0,  cmax=cmax, title=title, tstring=tstring, clabel=clabel, fld_id=fld_id, sticker=sticker, rfooter=shortname, edens=edens, cmap='plasma')
    fig.savefig(os.path.join(sf.subdir(pltdir, 'EM Omega'), tlabel + '.png'))# Save into a subdirectory of 'outdir'

    ## Plot 3: EM energy density, three-halves omega
    C = maps['1_5']*1e-12 # Convert from J/m^3 to uJ/um^3  ; Conversion is 10^6 * 10^-18 => 10^-12
    sticker = r'$3\omega/2$'
    title = 'EM field energy: ' + str(np.round(np.sum(C)*dx*dz, 1)) + " $\mu J/\mu m$"
    clabel = 'Energy density ($\mu J/\mu m^3$)'
    fld_id = r'$J$'
    cmax = None if alltime else 15
    fig = mypcolor(C, xgv, zgv, cmin=0,  cmax=cmax, title=title, tstring=tstring, clabel=clabel, fld_id=fld_id, sticker=sticker, rfooter=shortname, edens=edens, cmap='plasma')
    fig.savefig(os.path.join(sf.subdir(pltdir, 'EM Three-halves'), tlabel + '.png'))# Save into a subdirectory of 'outdir'

    ## Plot 4: EM energy density, half omega
    C = maps['0_5']*1e-12 # Convert from J/m^3 to uJ/um^3  ; Conversion is 10^6 * 10^-18 => 10^-12
    sticker = r'$\omega/2$'
    title = 'EM field energy: ' + str(np.round(np.sum(C)*dx*dz, 1)) + " $\mu J/\mu m$"
    clabel = 'Energy density ($\mu J/\mu m^3$)'
    fld_id = r'$J$'
    cmax = None if alltime else 25
    fig = mypcolor(C, xgv, zgv, cmin=0,  cmax=cmax, title=title, tstring=tstring, clabel=clabel, fld_id=fld_id, sticker=sticker, rfooter=shortname, edens=edens, cmap='plasma')
    fig.savefig(os.path.join(sf.subdir(pltdir, 'EM Half'), tlabel + '.png'))# Save into a subdirectory of 'outdir'

    ## Plot 5: EM energy density, static
    C = maps['0']*1e-12 # Convert from J/m^3 to uJ/um^3  ; Conversion is 10^6 * 10^-18 => 10^-12
    sticker = r'$static$'
    title = 'EM field energy: ' + str(np.round(np.sum(C)*dx*dz, 1)) + " $\mu J/\mu m$"
    clabel = 'Energy density ($\mu J/\mu m^3$)'
    fld_id = r'$J$'
    cmax = None if alltime else 15
    fig = mypcolor(C, xgv, zgv, cmin=0,  cmax=cmax, title=title, tstring=tstring, clabel=clabel, fld_id=fld_id, sticker=sticker, rfooter=shortname, edens=edens, cmap='plasma')
    fig.savefig(os.path.join(sf.subdir(pltdir, 'EM Static'), tlabel + '.png'))# Save into a subdirectory of 'outdir'

    ## Plot 6: EM power spectrum
    title = 'Power spectrum, ' + tstring
    fld_id = r'$J$'
    fig = plt.figure()
    x = freq
    y = pwrsum*1e-3 # convert from J/omega/m to mJ/omega/um
    plt.clf() # Clear the figure
    #ymax = np.max(y)
    ymax = 10
    ymin = 10**(np.log10(ymax) - 4) # Give 4 orders of magnitude
    xmin = 0
    xmax = 2.5
    ax = plt.subplot(111)
    ax.plot(x, y)
    ax.set_xlabel('Angular frequency / $\omega_{laser}$')
    ax.set_ylabel('mJ / $\omega_{laser} / \mu m$')
    ax.set_yscale('log')
    ax.set_title(title, fontsize=16)
    ax.set_xlim(xmin, xmax)
    ax.xaxis.grid() # vertical lines
    ax.text(xmax - (xmax - xmin)/7, ymin + (ymax - ymin)/16, fld_id, fontsize=44, color='black')
    ax.set_ylim(ymin, ymax)
    fig.text(0.99, 0.01, shortname, horizontalalignment='right')
    fig.savefig(os.path.join(sf.subdir(pltdir, 'EM Power spectrum'), tlabel + '.png'))# Save into a subdirectory of 'outdir'
    
    ## Close the plots
    plt.close('all')
