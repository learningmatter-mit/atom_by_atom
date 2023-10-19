import warnings
from csv import reader
from typing import List, Optional, Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bokeh.io import show as show_
from bokeh.models import (
    BasicTicker,
    ColorBar,
    ColumnDataSource,
    LinearColorMapper,
    LogColorMapper,
)
from bokeh.plotting import figure, output_file
from bokeh.sampledata.periodic_table import elements
from bokeh.transform import dodge
from matplotlib.cm import (
    ScalarMappable,
    cividis,
    inferno,
    magma,
    plasma,
    turbo,
    viridis,
)
from matplotlib.colors import LogNorm, Normalize, to_hex
from pandas import options
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

PROPERTIES = {
    "center_diff": "B 3d $-$ O 2p difference",
    "op": "O 2p $-$ $E_v$",
    "form_e": "Formation energy",
    "e_hull": "Energy above hull",
    "tot_e": "Energy per atom",
    "time": "Runtime",
    "magmom": "Magnetic moment",
    "ads_e": "Adsorption energy",
    "acid_stab": "Electrochemical stability",
    "bandcenter": "DOS band center",
    "bandwidth": "DOS band width",
    "bandfilling": "Amount of Bandfilling",
    "phonon": "Atomic vibration frequency",
    "bader": "Bader charge",
}

UNITS = {
    "center_diff": "eV",
    "op": "eV",
    "form_e": "eV",
    "e_hull": "eV/atom",
    "tot_e": "eV/atom",
    "time": "s",
    "magmom": "$\mu_B$",
    "ads_e": "eV",
    "acid_stab": "eV/atom",
    "bandcenter": "eV",
    "bandwidth": "eV",
    "phonon": "THz",
    "bandfilling": "$q_e$",
    "bader": "$q_e$"
}


def plot_scatter(
    targs,
    preds,
    prop_key: str,
    target_index: Optional[int] = None,
    test_ids: Optional[List] = None,
    structures: Optional[Dict[int, Structure]] = None,
    Z_range: Optional[List] = None,
    title="",
    scale="linear",
    inc_factor=1.1,
    dec_factor=0.9,
    facecolor="#219ebc",
    edgecolor="#ffffff",
    style='scifig'
):
    new_targ = []
    new_pred = []
    if test_ids is None:
        new_targ = targs
        new_pred = preds
    else:
        pred_dictionary: Dict[int, List] = {}
        dictionary: Dict[int, List] = {}
        for index in tqdm(range(len(test_ids))):
            id_ = test_ids[index]
            structure = structures.get(id_)
            elems = [Element(x.symbol).Z for x in structure.species]

            pred = preds[index]
            targ = targs[index]

            # get dictionary
            for i in range(len(elems)):

                elem = elems[i]
                y = targ[i][target_index]

                if elem in dictionary:
                    array = dictionary[elem]
                    array.append(y)
                    dictionary[elem] = array

                else:
                    dictionary[elem] = [y]

            # get pred_dictionary
            for i in range(len(elems)):

                elem = elems[i]
                y = pred[i][target_index]

                if elem in pred_dictionary:
                    array = pred_dictionary[elem]
                    array.append(y)
                    pred_dictionary[elem] = array

                else:
                    pred_dictionary[elem] = [y]

        if Z_range is None:
            for key in list(dictionary.keys()):
                new_targ += dictionary[key]
                new_pred += pred_dictionary[key]
        else:        
            for key in Z_range:
                new_targ += dictionary[key]
                new_pred += pred_dictionary[key]

    with plt.style.context(style):
        fig, ax = plt.subplots()
        ax.scatter(new_targ, new_pred, facecolors=facecolor, edgecolors=edgecolor, alpha=0.75, linewidth=0.5, s=15)
        mae = mean_absolute_error(new_targ, new_pred)
        r, _ = pearsonr(new_targ, new_pred)
        r_s = spearmanr(new_targ, new_pred).correlation

        if scale == "log":
            new_pred = np.abs(new_pred) + 1e-8
            new_targ = np.abs(new_targ) + 1e-8

        lim_min = min(np.min(new_pred), np.min(new_targ))
        if lim_min < 0:
            if lim_min > -0.1:
                lim_min = -0.1
            lim_min *= inc_factor
        else:
            if lim_min < 0.1:
                lim_min = -0.1
            lim_min *= dec_factor
        lim_max = max(np.max(new_pred), np.max(new_targ))
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
        ax.set_xticks(ax.get_yticks())
        ax.set_yticks(ax.get_yticks())
        ax.set_aspect("equal")

        ax.axline((0, 0), (1, 1), color="#000000", zorder=-1, linewidth=0.5)
        ax.set_title(title, fontsize=8)
        ax.set_ylabel("Predicted %s [%s]" % (PROPERTIES[prop_key], UNITS[prop_key]), fontsize=8)
        ax.set_xlabel("Calculated %s [%s]" % (PROPERTIES[prop_key], UNITS[prop_key]), fontsize=8)

        ax.annotate(
            "Pearson's r: %.3f \nSpearman's r: %.3f \nMAE: %.3f %s " % (r, r_s, mae, UNITS[prop_key]),
            (0.05, 0.85),
            xycoords="axes fraction",
            fontsize=6,
        )

    return fig, ax, r, mae


def plot_hexbin(
    targs,
    preds,
    prop_key: str,
    target_index: Optional[int] = None,
    test_ids: Optional[List] = None,
    structures: Optional[Dict[int, Structure]] = None,
    Z_range: Optional[List] = None,
    title="",
    scale="linear",
    inc_factor=1.1,
    dec_factor=0.9,
    bins=None,
    plot_helper_lines=False,
    cmap="viridis",
    style='scifig'
):
    new_targ = []
    new_pred = []
    if test_ids is None:
        new_targ = targs
        new_pred = preds
    else:
        pred_dictionary: Dict[int, List] = {}
        dictionary: Dict[int, List] = {}
        for index in tqdm(range(len(test_ids))):
            id_ = test_ids[index]
            structure = structures.get(id_)
            elems = [Element(x.symbol).Z for x in structure.species]

            pred = preds[index]
            targ = targs[index]

            # get dictionary
            for i in range(len(elems)):

                elem = elems[i]
                y = targ[i][target_index]

                if elem in dictionary:
                    array = dictionary[elem]
                    array.append(y)
                    dictionary[elem] = array

                else:
                    dictionary[elem] = [y]

            # get pred_dictionary
            for i in range(len(elems)):

                elem = elems[i]
                y = pred[i][target_index]

                if elem in pred_dictionary:
                    array = pred_dictionary[elem]
                    array.append(y)
                    pred_dictionary[elem] = array

                else:
                    pred_dictionary[elem] = [y]

        if Z_range is None:
            for key in list(dictionary.keys()):
                new_targ += dictionary[key]
                new_pred += pred_dictionary[key]
        else:        
            for key in Z_range:
                new_targ += dictionary[key]
                new_pred += pred_dictionary[key]
    # Use style in matplotlib.style.available
    with plt.style.context(style):
        fig, ax = plt.subplots()

        mae = mean_absolute_error(new_targ, new_pred)
        r, _ = pearsonr(new_targ, new_pred)
        r_s = spearmanr(new_targ, new_pred).correlation

        if scale == "log":
            new_pred = np.abs(new_pred) + 1e-8
            new_targ = np.abs(new_targ) + 1e-8

        lim_min = min(np.min(new_pred), np.min(new_targ))
        if lim_min < 0:
            if lim_min > -0.1:
                lim_min = -0.1
            lim_min *= inc_factor
        else:
            if lim_min < 0.1:
                lim_min = -0.1
            lim_min *= dec_factor
        lim_max = max(np.max(new_pred), np.max(new_targ))
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
        ax.set_xticks(ax.get_yticks())
        ax.set_yticks(ax.get_yticks())
        ax.set_aspect("equal")

        # ax.plot((lim_min, lim_max),
        #        (lim_min, lim_max),
        #        color='#000000',
        #        zorder=-1,
        #        linewidth=0.5)
        ax.axline((0, 0), (1, 1), color="#000000", zorder=-1, linewidth=0.5)

        hb = ax.hexbin(
            new_targ,
            new_pred,
            cmap=cmap,
            gridsize=60,
            bins=bins,
            mincnt=1,
            edgecolors=None,
            linewidths=(0.1,),
            xscale=scale,
            yscale=scale,
            extent=(lim_min, lim_max, lim_min, lim_max),
            norm=matplotlib.colors.LogNorm(),
        )

        cb = fig.colorbar(hb, shrink=0.822)
        cb.set_label("Count")

        if plot_helper_lines:

            if scale == "linear":
                x = np.linspace(lim_min, lim_max, 50)
                y_up = x + mae
                y_down = x - mae

            elif scale == "log":
                x = np.logspace(np.log10(lim_min), np.log10(lim_max), 50)

                # one order of magnitude
                y_up = np.maximum(x + 1e-2, x * 10)
                y_down = np.minimum(np.maximum(1e-8, x - 1e-2), x / 10)

                # one kcal/mol/Angs
                y_up = x + 1
                y_down = np.maximum(1e-8, x - 1)

            for y in [y_up, y_down]:
                ax.plot(x, y, color="#000000", zorder=2, linewidth=0.5, linestyle="--")

        ax.set_title(title, fontsize=8)
        ax.set_ylabel("Predicted %s [%s]" % (PROPERTIES[prop_key], UNITS[prop_key]), fontsize=8)
        ax.set_xlabel("Calculated %s [%s]" % (PROPERTIES[prop_key], UNITS[prop_key]), fontsize=8)

        ax.annotate(
            "Pearson's r: %.3f \nSpearman's r: %.3f \nMAE: %.3f %s " % (r, r_s, mae, UNITS[prop_key]),
            (0.05, 0.85),
            xycoords="axes fraction",
            fontsize=6,
        )

    return fig, ax, r, mae, hb


def flatten(t):
    return [item for sublist in t for item in sublist]


def plot_violin(
    targs,
    preds,
    target_index: int,
    prop_key: str,
    test_ids: List,
    structures: Structure,
    color1: str = "#679289",
    color2: str = "#F4C095",
    Z_range=[22, 30],
    legend=True,
    legend_loc="lower left",
    style='scifig'
):

    pred_dictionary = {}
    dictionary = {}

    for index in tqdm(range(len(test_ids))):

        id_ = test_ids[index]
        structure = structures[id_]
        elems = [Element(x.symbol).Z for x in structure.species]

        pred = preds[index]
        targ = targs[index]

        # get dictionary
        for i in range(len(elems)):

            elem = elems[i]
            y = targ[i][target_index]

            if elem in dictionary:
                array = dictionary[elem]
                array.append(y)
                dictionary[elem] = array

            else:
                dictionary[elem] = [y]

        # get pred_dictionary
        for i in range(len(elems)):

            elem = elems[i]
            y = pred[i][target_index]

            if elem in pred_dictionary:
                array = pred_dictionary[elem]
                array.append(y)
                pred_dictionary[elem] = array

            else:
                pred_dictionary[elem] = [y]

    df = pd.DataFrame(columns=("Z", "y", "hue"))
    Zs = []
    ys = []
    hues = []
    for Z in Z_range:

        ys.append(dictionary[Z])
        Zs.append([Z for i in range(len(dictionary[Z]))])
        hues.append(["targ" for i in range(len(dictionary[Z]))])

        ys.append(pred_dictionary[Z])
        Zs.append([Z for i in range(len(pred_dictionary[Z]))])
        hues.append(["pred" for i in range(len(pred_dictionary[Z]))])

    Zs = flatten(Zs)
    ys = flatten(ys)
    hues = flatten(hues)

    df["Z"] = Zs
    df[prop_key] = ys
    df["hue"] = hues

    my_pal = {"targ": color1, "pred": color2}
    with plt.style.context(style):
        f = plt.figure()
        ax = sns.violinplot(
            x="Z",
            y=prop_key,
            hue="hue",
            data=df,
            palette=my_pal,
            split=True,
            inner=None,
            linewidth=1,
        )

        # plt.ylim([-0.1,5.2])'

        plt.legend(loc=legend_loc, fontsize=6)
        ax.set_ylabel("%s [%s]" % (PROPERTIES[prop_key], UNITS[prop_key]), fontsize=8)
        ax.set_xlabel("")
        ax.tick_params(axis='x',labelsize=8)
        ax.tick_params(axis='y',labelsize=6)
        xticks = [Element.from_Z(int(text._text)).symbol for text in ax.get_xticklabels()]
        ax.set_xticklabels(xticks)
        if not legend:
            ax.get_legend().remove()

    return f, ax


"""
Orignal script from:
"https://github.com/arosen93/ptable_trends"
from bokeh.plotting import output_file, show
output_file("output.html")
show(figure)
"""


def ptable_plotter(
    filename: str,
    show: bool = True,
    output_filename: str = None,
    width: int = 1050,
    cmap: str = "plasma",
    alpha: float = 0.65,
    extended: bool = True,
    periods_remove: List[int] = None,
    groups_remove: List[int] = None,
    log_scale: bool = False,
    cbar_height: float = None,
    cbar_standoff: int = 12,
    cbar_fontsize: int = 14,
    blank_color: str = "#c4c4c4",
    under_value: float = None,
    under_color: str = "#140F0E",
    over_value: float = None,
    over_color: str = "#140F0E",
    special_elements: List[str] = None,
    special_color: str = "#6F3023",
) -> figure:

    """
    Plot a heatmap over the periodic table of elements.
    Parameters
    ----------
    filename : str
        Path to the .csv file containing the data to be plotted.
    show : str
        If True, the plot will be shown.
    output_filename : str
        If not None, the plot will be saved to the specified (.html) file.
    width : float
        Width of the plot.
    cmap : str
        plasma, inferno, viridis, magma, cividis, turbo
    alpha : float
        Alpha value (transparency).
    extended : bool
        If True, the lanthanoids and actinoids will be shown.
    periods_remove : List[int]
        Period numbers to be removed from the plot.
    groups_remove : List[int]
        Group numbers to be removed from the plot.
    log_scale : bool
        If True, the colorbar will be logarithmic.
    cbar_height : int
        Height of the colorbar.
    cbar_standoff : int
        Distance between the colorbar and the plot.
    cbar_fontsize : int
        Fontsize of the colorbar label.
    blank_color : str
        Hexadecimal color of the elements without data.
    under_value : float
        Values <= under_value will be colored with under_color.
    under_color : str
        Hexadecimal color to be used for the lower bound color.
    over_value : float
        Values >= over_value will be colored with over_color.
    under_color : str
        Hexadecial color to be used for the upper bound color.
    special_elements: List[str]
        List of elements to be colored with special_color.
    special_color: str
        Hexadecimal color to be used for the special elements.
    Returns
    -------
    figure
        Bokeh figure object.
    """

    options.mode.chained_assignment = None

    # Assign color palette based on input argument
    if cmap == "plasma":
        cmap = plasma
        bokeh_palette = "Plasma256"
    elif cmap == "inferno":
        cmap = inferno
        bokeh_palette = "Inferno256"
    elif cmap == "magma":
        cmap = magma
        bokeh_palette = "Magma256"
    elif cmap == "viridis":
        cmap = viridis
        bokeh_palette = "Viridis256"
    elif cmap == "cividis":
        cmap = cividis
        bokeh_palette = "Cividis256"
    elif cmap == "turbo":
        cmap = turbo
        bokeh_palette = "Turbo256"
    else:
        ValueError("Invalid color map.")

    # Define number of and groups
    period_label = ["1", "2", "3", "4", "5", "6", "7"]
    group_range = [str(x) for x in range(1, 19)]

    # Remove any groups or periods
    if groups_remove:
        for gr in groups_remove:
            gr = gr.strip()
            group_range.remove(str(gr))
    if periods_remove:
        for pr in periods_remove:
            pr = pr.strip()
            period_label.remove(str(pr))

    # Read in data from CSV file
    data_elements = []
    data_list = []
    for row in reader(open(filename)):
        data_elements.append(row[0])
        data_list.append(row[1])
    data = [float(i) for i in data_list]

    if len(data) != len(data_elements):
        raise ValueError("Unequal number of atomic elements and data points")

    period_label.append("blank")
    period_label.append("La")
    period_label.append("Ac")

    if extended:
        count = 0
        for i in range(56, 70):
            elements.period[i] = "La"
            elements.group[i] = str(count + 4)
            count += 1

        count = 0
        for i in range(88, 102):
            elements.period[i] = "Ac"
            elements.group[i] = str(count + 4)
            count += 1

    # Define matplotlib and bokeh color map
    if log_scale:
        for datum in data:
            if datum < 0:
                raise ValueError(
                    f"Entry for element {datum} is negative but log-scale is selected"
                )
        color_mapper = LogColorMapper(
            palette=bokeh_palette, low=min(data), high=max(data)
        )
        norm = LogNorm(vmin=min(data), vmax=max(data))
    else:
        color_mapper = LinearColorMapper(
            palette=bokeh_palette, low=min(data), high=max(data)
        )
        norm = Normalize(vmin=min(data), vmax=max(data))
    color_scale = ScalarMappable(norm=norm, cmap=cmap).to_rgba(data, alpha=None)

    # Set blank color
    color_list = [blank_color] * len(elements)

    # Compare elements in dataset with elements in periodic table
    for i, data_element in enumerate(data_elements):
        element_entry = elements.symbol[
            elements.symbol.str.lower() == data_element.lower()
        ]
        if element_entry.empty == False:
            element_index = element_entry.index[0]
        else:
            warnings.warn("Invalid chemical symbol: " + data_element)
        if color_list[element_index] != blank_color:
            warnings.warn("Multiple entries for element " + data_element)
        elif under_value is not None and data[i] <= under_value:
            color_list[element_index] = under_color
        elif over_value is not None and data[i] >= over_value:
            color_list[element_index] = over_color
        else:
            color_list[element_index] = to_hex(color_scale[i])

    if special_elements:
        for k, v in elements["symbol"].iteritems():
            if v in special_elements:
                color_list[k] = special_color

    # Define figure properties for visualizing data
    source = ColumnDataSource(
        data=dict(
            group=[str(x) for x in elements["group"]],
            period=[str(y) for y in elements["period"]],
            sym=elements["symbol"],
            atomic_number=elements["atomic number"],
            type_color=color_list,
        )
    )

    # Plot the periodic table
    p = figure(x_range=group_range, y_range=list(reversed(period_label)), tools="save")
    p.plot_width = width
    p.outline_line_color = None
    p.background_fill_color = None
    p.border_fill_color = None
    p.toolbar_location = "above"
    p.rect("group", "period", 0.9, 0.9, source=source, alpha=alpha, color="type_color")
    p.axis.visible = False
    text_props = {
        "source": source,
        "angle": 0,
        "color": "black",
        "text_align": "left",
        "text_baseline": "middle",
    }
    x = dodge("group", -0.4, range=p.x_range)
    y = dodge("period", 0.3, range=p.y_range)
    p.text(
        x=x,
        y="period",
        text="sym",
        text_font_style="bold",
        text_font_size="16pt",
        **text_props,
    )
    p.text(x=x, y=y, text="atomic_number", text_font_size="11pt", **text_props)

    color_bar = ColorBar(
        color_mapper=color_mapper,
        ticker=BasicTicker(desired_num_ticks=10),
        border_line_color=None,
        label_standoff=cbar_standoff,
        location=(0, 0),
        orientation="vertical",
        scale_alpha=alpha,
        major_label_text_font_size=f"{cbar_fontsize}pt",
    )

    if cbar_height is not None:
        color_bar.height = cbar_height

    p.add_layout(color_bar, "right")
    p.grid.grid_line_color = None

    if output_filename:
        output_file(output_filename)

    if show:
        show_(p)

    return p
