# darkMed = darks_flat_med_axis0 - np.min(darks_flat_med_axis0)
# darksMed_scaled = darks_flat_med_axis0 / median(darks_flat_med_axis0)# pp.scale(darks_flat_med_axis0)

diff_darks_flat_med_axis0     = np.zeros(darks_flat_med_axis0.size)
diff_darks_flat_med_axis0[1:] = diff(darks_flat_med_axis0)

dark_features  = []

for iDark, darkNow in enumerate(darks_flat_sample):
    
    darkNowRescaled = pp.scale((darkNow - np.min(darkNow))-(darks_flat_med_axis0 - np.min(darks_flat_med_axis0)))
    _, xhist = np.histogram(darkNowRescaled, bins=20, normed=True)#, alpha=0.50)
    
    kde1 = sm.nonparametric.KDEUnivariate(darkNowRescaled)
    kde1.fit(kernel='uni', bw=0.33*median(diff(xhist)), fft=False)
    
    leave1out = np.zeros(nFrames)
    for k in range(nFrames):
        leave1out[k] = np.std(hstack([darkNow[:k],darkNow[k+1:]]))
    
    leave1out_rescaled  = rescale(leave1out)
    
    kde3 = sm.nonparametric.KDEUnivariate(pp.scale(np.std(darkNow) - leave1out))
    kde3.fit(kernel='uni', fft=False)
    
    darkNowRescaled_KDE = rescale(kde1.density)
    l1o_Rescaled_KDE   = rescale(kde3.density)
    
    darkDiffRescale    = rescale(diff(darkNow))
    
    dark0    = darkNow - np.min(darkNow)
    
    dark0_rescaled = rescale(dark0 - diff_darks_flat_med_axis0)
    
    flagNow = flags_loc[iDark]
    avgCnt   = 0
    darksAvg = np.zeros(darkNow.size)
    if flagNow[0] > 0:
        avgCnt   += 1
        darksAvg += (darks[:,flagNow[0]-1, flagNow[1]+0] - diff_darks_flat_med_axis0) * std(darks[:,flagNow[0]-1, flagNow[1]+0])
        
        # ax4.plot(rescale(darks[:,flagNow[0]-1, flagNow[1]+0]),'o-')
    if flagNow[0] + 1 < darks.shape[1]:
        avgCnt   += 1
        darksAvg += (darks[:,flagNow[0]+1, flagNow[1]+0] - diff_darks_flat_med_axis0) * std(darks[:,flagNow[0]+1, flagNow[1]+0])
        
        # ax4.plot(rescale(darks[:,flagNow[0]+1, flagNow[1]+0]),'o-')
    if flagNow[1] > 0:
        avgCnt   += 1
        darksAvg += (darks[:,flagNow[0]+0, flagNow[1]-1] - diff_darks_flat_med_axis0) * std(darks[:,flagNow[0]+0, flagNow[1]-1])
        
        # ax4.plot(rescale(darks[:,flagNow[0]+0, flagNow[1]-1]),'o-')
    if flagNow[1] + 1 < darks.shape[1]:
        avgCnt   += 1
        darksAvg += (darks[:,flagNow[0]+0, flagNow[1]+1] - diff_darks_flat_med_axis0) * std(darks[:,flagNow[0]+0, flagNow[1]+1])
        
    darksAvg  = darksAvg / avgCnt
    
    darksAvg_rescaled = rescale(darksAvg)
    
    dark_features.append(hstack([dark0_rescaled, darkNowRescaled_KDE, l1o_Rescaled_KDE, leave1out_rescaled, darkDiffRescale, darksAvg_rescaled]))