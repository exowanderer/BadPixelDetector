fig = figure(figsize=(15,15))

ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

darkMed = darks_flat_med_axis0 - np.min(darks_flat_med_axis0)
darksMed_scaled = darks_flat_med_axis0 / median(darks_flat_med_axis0)# pp.scale(darks_flat_med_axis0)

diff_darks_flat_med_axis0     = np.zeros(darks_flat_med_axis0.size)
diff_darks_flat_med_axis0[1:] = diff(darks_flat_med_axis0)

classLabels = {1:'Noisy', 2:'HP', 3:'IHP', 4:'LHP', 5:'SHP', 6:'CR', 7:'RTN0', 8:'RTN1'}

for iQuirk, quirkNow in enumerate(quirks_store):
    if quirkCheck[iQuirk]:
        continue
    
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    
    classNow = classes_store[iQuirk]
    classOut = classNow
    if classNow == 3:
        classOut = 7
    if classNow == 4:
        classOut = 6
    if classNow == 5:
        classOut = 5
    if classNow == 6:
        classOut = 3
    
    # ax1.plot(darks_reshaped[irow][1:] / median(darks_reshaped[irow][1:]) - darks_flat_med_axis0_norm);
    
    # Plot Subtraction frame: (Darknow - min(Darknow)) - (DarkMed - min(DarkMed))
    # darkNowMinusMed = (darks_reshaped[irow][1:] - np.min(darks_reshaped[irow][1:])) - \
    #                   (darks_flat_med_axis0[1:] - np.min(darks_flat_med_axis0[1:]))
    
    quirkNow_scaled = pp.scale(quirkNow)
    
    quirkMinusMed = (quirkNow_scaled - np.min(quirkNow_scaled)) - (darksMed_scaled - np.min(darksMed_scaled))

    kde1 = sm.nonparametric.KDEUnivariate((quirkNow - np.min(quirkNow))-(darks_flat_med_axis0 - np.min(darks_flat_med_axis0)))
    kde1.fit(kernel='uni', fft=False)
    
    ax1.hist((quirkNow - np.min(quirkNow))-(darks_flat_med_axis0 - np.min(darks_flat_med_axis0)), bins=20, normed=True, alpha=0.50)
    ax1.plot(kde1.support, kde1.density, lw=2, color=rcParams['axes.color_cycle'][0])
    
    if classNow == 1:
        ax1.hist((darks_flat_med_axis0 - np.median(darks_flat_med_axis0)), bins=20, normed=True, alpha=0.25)
        kde2 = sm.nonparametric.KDEUnivariate((darks_flat_med_axis0 - np.median(darks_flat_med_axis0)))
        kde2.fit(kernel='uni', fft=False)
        ax1.plot(kde2.support, kde2.density, lw=2, color=rcParams['axes.color_cycle'][1])
    
    ylims = ax1.get_ylim()
    xlims = ax1.get_xlim()
    xyNow1 = [np.min(xlims) + 0.1*diff(xlims),
              np.min(ylims) + 0.9*diff(ylims)]
    ax1.annotate(str(classOut) + ': ' + classLabels[classOut], xyNow1, fontsize=75)
    # ax1.plot()
    # ax1.axvline(median(rtnNow), linestyle='--', color='k')
    ax1.set_xlabel('Subtraction Hist')
    # ax1.plot(darks_reshaped[irow][1:] / median(darks_reshaped[irow][1:]) / darks_flat_med_axis0_norm - 1);
    
    # Plot Normalized Frame: DarkNow vs DarMed
    ax2.plot((quirkNow - np.min(quirkNow))/darksMed_scaled,'o-');
    ylims = ax2.get_ylim()
    xlims = ax2.get_xlim()
    xyNow2 = [np.min(xlims) + 0.1*diff(xlims),
              np.min(ylims) + 0.9*diff(ylims)]
    ax2.annotate(str(classOut) + ': ' + classLabels[classOut], xyNow2, fontsize=75)
    #ax2.plot(darksMed_scaled,'o-')
    ax2.set_xlabel('Normalized Frame')
    
    # Plot Common Mode Correlation Frame
    # ax3.plot((quirkNow - np.min(quirkNow))-(darks_flat_med_axis0 - np.min(darks_flat_med_axis0)), darksMed_scaled,'o')
    # ax3.plot(darksMed_scaled, darksMed_scaled,'o')
    ax3.plot(diff(quirkNow),'o-')
    ax3.plot(diff_darks_flat_med_axis0, 'o-')
    ylims = ax3.get_ylim()
    xlims = ax3.get_xlim()
    xyNow3 = [np.min(xlims) + 0.1*diff(xlims),
              np.min(ylims) + 0.9*diff(ylims)]
    ax3.annotate(str(classOut) + ': ' + classLabels[classOut], xyNow3, fontsize=75)
    ax3.set_xlabel('Diff Mode')
    
    # Plot Raw DN minus Min Dark Ramp: DarkNow - min(DarkNow) vs DarkMed - min(DarkMed)
    flagNow = flags_loc[iQuirk]
    dark0   = quirkNow - np.min(quirkNow)
    ax4.plot(rescale((dark0 + diff_darks_flat_med_axis0)),'o-')
    
    avgCnt   = 0
    darksAvg = np.zeros(quirkNow.size)
    if flagNow[0] > 0:
        avgCnt   += 1
        darksAvg += darks[:,flagNow[0]-1, flagNow[1]  ] - np.min(darks[:,flagNow[0]-1, flagNow[1]  ]) *\
                    std(darks[:,flagNow[0]-1, flagNow[1]  ] - np.min(darks[:,flagNow[0]-1, flagNow[1]  ]))
        # ax4.plot(pp.scale(darks[:,flagNow[0]-1, flagNow[1]  ] - np.min(darks[:,flagNow[0]-1, flagNow[1]  ])\
        #          + diff_darks_flat_med_axis0),'o-')
    if flagNow[0] < darks.shape[0]:
        avgCnt   += 1
        darksAvg += darks[:,flagNow[0]+1, flagNow[1]  ] - np.min(darks[:,flagNow[0]+1, flagNow[1]  ]) *\
                    std(darks[:,flagNow[0]+1, flagNow[1]  ] - np.min(darks[:,flagNow[0]+1, flagNow[1]  ]))
        # ax4.plot(pp.scale(darks[:,flagNow[0]+1, flagNow[1]  ] - np.min(darks[:,flagNow[0]+1, flagNow[1]  ])\
        #          + diff_darks_flat_med_axis0),'o-')
    if flagNow[1] > 0:
        avgCnt   += 1
        darksAvg += darks[:,flagNow[0]  , flagNow[1]-1] - np.min(darks[:,flagNow[0]  , flagNow[1]-1]) *\
                    std(darks[:,flagNow[0]  , flagNow[1]-1] - np.min(darks[:,flagNow[0]  , flagNow[1]-1]))
        # ax4.plot(pp.scale(darks[:,flagNow[0]  , flagNow[1]-1] - np.min(darks[:,flagNow[0]  , flagNow[1]-1])\
        #          + diff_darks_flat_med_axis0),'o-')
    if flagNow[1] < darks.shape[0]:
        avgCnt   += 1
        darksAvg += darks[:,flagNow[0]  , flagNow[1]+1] - np.min(darks[:,flagNow[0]  , flagNow[1]+1]) *\
                    std(darks[:,flagNow[0]  , flagNow[1]+1] - np.min(darks[:,flagNow[0]  , flagNow[1]+1]))
        
        # ax4.plot(pp.scale(darks[:,flagNow[0]  , flagNow[1]+1] - np.min(darks[:,flagNow[0]  , flagNow[1]+1])\
        #          + diff_darks_flat_med_axis0),'o-')
    # meandark = np.mean([darks[:,159, 335]*std(darks[:,159, 335]),darks[:,161, 335]*std(darks[:,161, 335]), \
    #          darks[:,160, 334]*std(darks[:,160, 334]),darks[:,160, 336]*std(darks[:,160, 336])], axis=0)
    
    # plot(rescale(darks[:,160, 335]), lw=4)
    # plot(rescale(meandark), lw=4)

    ax4.plot(rescale(darksAvg / avgCnt), 'ko-')
    # ax4.plot((darkMed - diff_darks_flat_med_axis0) - np.min(darkMed - diff_darks_flat_med_axis0)\
    #          / ((np.max(darkMed - diff_darks_flat_med_axis0) - np.min(darkMed - diff_darks_flat_med_axis0))), 'ro-')
    
    ylims = ax4.get_ylim()
    xlims = ax4.get_xlim()
    xyNow4 = [np.min(xlims) + 0.1*diff(xlims),
              np.min(ylims) + 0.9*diff(ylims)]
    ax4.annotate(str(classOut) + ': ' + classLabels[classOut], xyNow4, fontsize=75)
    
    ax4.set_xlabel('Raw DN - Min ' + str(flagNow[0]) + ',' + str(flagNow[1]))
    # ax4.set_ylim(-5,5)
    # ax.plot(darks_flat_med_axis0[1:] / median(darks_flat_med_axis0[1:]))
    fig.suptitle('iQuirk: ' + str(iQuirk) + ' / ' + str(len(quirks_store)), fontsize=20)
    # ax1.set_ylim(ax2.get_ylim())
    fig.canvas.draw()
    display.display(plt.gcf())
    
    inputNow = input('[1:Noisy, 2:HP, 3:IHP, 4:LHP, 5:SHP, 6:CR, 7:RTN0, 8:RTN1]? ')
    
    # inputNowBak = np.copy(inputNow)
    quirkCheck[iQuirk] = int(classOut)
    if inputNow == '':
        pass
    else:
        classOut = int(inputNow)
        
        doubleCheck = input(str(classNow) + " -> " + str(classOut) + "? ")
        
        if {'y':True, 'n':False}[doubleCheck.lower()[0]]:
            pass
        else:
            quirkCheck[iQuirk] = int(classNow)
    
    display.clear_output(wait=True)