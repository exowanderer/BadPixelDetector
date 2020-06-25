# %cpaste
#
# start = np.where(class_keep == 0)[0].min()
# for k in range(start,len(classme_array)):
#     # initial assumption
#     ax.clear()
#     ax.plot(quirks2[k])
#     ax.set_title(str(k+1)  + ' / ' + str(quirks2.shape[0]) + ';' + 'Current: ' + str(classme_array[k]))
#     ylims = ax.get_ylim()
#     xlims = ax.get_xlim()
#     xy_min  = 0.5*array([np.min(xlims),np.min(ylims)])
#     xy_diff = 1.0*array([np.diff(xlims),np.diff(ylims)])
#     xy_ann  = xy_min + 0.9*xy_diff
#     # ax.annotate(classme_array[k], xy_ann)
#     fig.canvas.draw()
#     new_input = input('Class Now? [1:Noisy, 2:HP, 3:RTN, 4:CR, 5:Other, 0:Oops] ')
#     if new_input == '':
#         # print('Keeping entry ' + str(k+1) + ' as ' + str(classme_array[k]))
#         class_keep[k] = np.copy(classme_array[k])
#     else:
#         if float(new_input) in arange(6):
#             class_keep[k] = np.float(new_input)
#             print('Changed ' + str(classme_array[k]) + ' to a ' + new_input + ' on the ' + str(k+1) + ' entry')
#         else:
#             print('That did make sense; input must be one of ', arange(6))
# --

    # if new_input == '':
    #     new_input = classme_array[k]
    # else:
    #     class_keep[k] = new_input
    #
    # if new_input != classme_array[k]:
    #     checkOW = input('Are you sure you want to overwrite? [y/n] ')
    #     if 'y' in checkOW.lower():
    #         class_keep[k] = new_input
    #         print('Changed ' + classme_array[k] + ' to a ' + new_input + ' on the ' + str(k+1) + ' entry')


# 376 == 1
# 377 == 2
# 378 == 4
# 379 == 2

# 723 == 1  ;; 741 is 1 ?  -- 18
# 1474 --> 1492?           -- 18


# %cpaste
#
# start = np.where(class_keep == 0)[0].min()
# for k in range(start,len(classme_array)):
#     # initial assumption
#     ax.clear()
#     ax.plot(quirks2[k])
#     kclassNow = k+koffset
#     ax.set_title(str(k+1)  + ' / ' + str(quirks2.shape[0]) + ';' + 'Current: ' + str(classme_array[kclassNow]))
#     ylims = ax.get_ylim()
#     xlims = ax.get_xlim()
#     ax.annotate(str(classme_array[kclassNow]), [min(xlims) + diff(xlims)[0]*0.6,min(ylims) + diff(ylims)[0]*0.2], fontsize=100)
#     fig.canvas.draw()
#     new_input = input('Class Now? [1:Noisy, 2:HP, 3:RTN, 4:CR, 5:Other, 0:Oops] ')
#     if new_input == '':
#         # print('Keeping entry ' + str(k+1) + ' as ' + str(classme_array[k]))
#         class_keep[k] = np.copy(classme_array[kclassNow])
#     else:
#         if float(new_input) in arange(6):
#             class_keep[k] = np.float(new_input)
#             print('Changed ' + str(classme_array[k]) + ' to a ' + new_input + ' on the ' + str(k+1) + ' entry')
#         else:
#             print('That did make sense; input must be one of ', arange(6))
# --

%cpaste

start = np.where(class_final == 0)[0].min()
for k in range(start,len(class_keep)):
    # initial assumption
    ax.clear()
    ax.plot(quirks2[k])
    kclassNow = k+koffset
    ax.set_title(str(k+1)  + ' / ' + str(quirks2.shape[0]) + ';' + 'Current: ' + str(class_keep[kclassNow]))
    ylims = ax.get_ylim()
    xlims = ax.get_xlim()
    ax.annotate(str(class_keep[kclassNow]), [min(xlims) + diff(xlims)[0]*0.6,min(ylims) + diff(ylims)[0]*0.2], fontsize=100)
    fig.canvas.draw()
    new_input = input('Class Now? [1:Noisy, 2:HP, 3:RTN, 4:CR, 5:Other, 0:Oops] ')
    if new_input == '':
        # print('Keeping entry ' + str(k+1) + ' as ' + str(class_keep[k]))
        class_final[k] = np.copy(class_keep[kclassNow])
    else:
        if float(new_input) in arange(6):
            class_final[k] = np.float(new_input)
            print('Changed ' + str(class_keep[k]) + ' to a ' + new_input + ' on the ' + str(k+1) + ' entry')
        else:
            print('That did make sense; input must be one of ', arange(6))
--
