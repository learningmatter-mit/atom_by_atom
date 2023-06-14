import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
#from scipy.stats import pearsonr

# import spearmanr
from scipy.stats import spearmanr

def plot_hexbin(targ, pred, fig, ax, scale="linear",
                inc_factor = 1.1, dec_factor = 0.9,
                bins=None, plot_helper_lines=False,
                cmap='viridis'):


    mae = mean_absolute_error(targ, pred)
    r, _ = spearmanr(targ, pred)

    if scale == 'log':
        pred = np.abs(pred) + 1e-8
        targ = np.abs(targ) + 1e-8

    lim_min = min(np.min(pred), np.min(targ))
    if lim_min < 0:
        if lim_min > -0.1:
            lim_min = -0.1
        lim_min *= inc_factor
    else:
        if lim_min < 0.1:
            lim_min = -0.1
        lim_min *= dec_factor
    lim_max = max(np.max(pred), np.max(targ))
    if lim_max <= 0:
        if lim_max > -0.1:
            lim_max = 0.2
        lim_max *= dec_factor
    else:
        if lim_max < 0.1:
            lim_max = 0.25
        lim_max *= inc_factor

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect('equal')

    ax.axline((0, 0), (1, 1),
           color='#000000',
           zorder=-1,
           linewidth=0.5)

    hb = ax.hexbin(
        targ, pred,
        cmap=cmap,
        gridsize=60,
        bins=bins,
        mincnt=1,
        edgecolors=None,
        linewidths=(0.1,),
        xscale=scale,
        yscale=scale,
        extent=(lim_min, lim_max, lim_min, lim_max))


    cb = fig.colorbar(hb, shrink=0.822)
    cb.set_label('Count')

    if plot_helper_lines:

        if scale == 'linear':
            x = np.linspace(lim_min, lim_max, 50)
            y_up = x + mae
            y_down = x - mae

        elif scale == 'log':
            x = np.logspace(np.log10(lim_min), np.log10(lim_max), 50)

            # one order of magnitude
            y_up = np.maximum(x + 1e-2, x * 10)
            y_down = np.minimum(np.maximum(1e-8, x - 1e-2), x / 10)

            # one kcal/mol/Angs
            y_up = x + 1
            y_down = np.maximum(1e-8, x - 1)


        for y in [y_up, y_down]:
            ax.plot(x,
                    y,
                    color='#000000',
                    zorder=2,
                    linewidth=0.5,
                    linestyle='--')


    ax.annotate("Spearman r: %.3f" % (r),
                (0.03, 0.88),
                xycoords='axes fraction',
                fontsize=12)

    return r, mae, ax, hb
