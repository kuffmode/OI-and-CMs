import seaborn as sns

CM = 1 / 2.54
TINY = (5 * CM, 3 * CM)
SMALL = (9 * CM, 5 * CM)
HALF_PAGE = (18 * CM, 7 * CM)
FULL_PAGE = (18 * CM, 15 * CM)

DEEP_BLUE = "#006685"
BLUE = "#3FA5C4"
WHITE = "#FFFFFF"
HALF_BLACK = "#232324"
ORANGE = "#E84653"
RED = "#BF003F"

PURPLE = "#A6587C"
PURPLER = "#591154"
PURPLEST = "#260126"
NIGHT_BLUE = "#394D73"
YELLOW = "#E6B213"

TEAL = "#44cfcf"
SLOW_GREEN = "#a1d4ca"
GRAY = "#b8b8b8"

black_to_gray_to_reds = [GRAY, GRAY, WHITE, ORANGE, RED]
from_white = [DEEP_BLUE, BLUE, WHITE, ORANGE, RED]
white_to_reds = [WHITE, ORANGE, RED]
white_to_blues = [WHITE, BLUE, DEEP_BLUE]
teal_to_red = [TEAL, SLOW_GREEN, WHITE, YELLOW, ORANGE, RED]

black_to_reds = [HALF_BLACK, ORANGE, RED]
black_to_blues = [HALF_BLACK, BLUE, DEEP_BLUE]

from_black = [DEEP_BLUE, BLUE, HALF_BLACK, ORANGE, RED]
purples = [PURPLE, WHITE, NIGHT_BLUE]

mono_black_gray_red = sns.blend_palette(black_to_gray_to_reds, as_cmap=True)
diverge_from_white = sns.blend_palette(from_white, as_cmap=True)
purples_diverge_from_white = sns.blend_palette(purples, as_cmap=True)

diverge_from_black = sns.blend_palette(from_black, as_cmap=True)

white_red_mono = sns.blend_palette(white_to_reds, as_cmap=True)
white_blue_mono = sns.blend_palette(white_to_blues, as_cmap=True)

black_red_mono = sns.blend_palette(black_to_reds, as_cmap=True)
black_blue_mono = sns.blend_palette(black_to_blues, as_cmap=True)

purple_red = sns.blend_palette([PURPLEST, PURPLE, RED])
teal_red = sns.blend_palette(teal_to_red, as_cmap=True)
monochrome = sns.blend_palette([HALF_BLACK, GRAY], as_cmap=True)


def set_visual_style():
    sns.set_theme(
        style="ticks",
        font_scale=0.75,
        rc={
            "font.family": "Atkinson Hyperlegible",
            "font.sans-serif": ["Atkinson-Hyperlegible"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "axes.labelpad": 2,
            "axes.linewidth": 0.5,
            "axes.titlepad": 4,
            "lines.linewidth": 1,
            "legend.fontsize": 8,
            "legend.title_fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.size": 2,
            "xtick.major.pad": 1,
            "xtick.major.width": 0.5,
            "ytick.major.size": 2,
            "ytick.major.pad": 1,
            "ytick.major.width": 0.5,
            "xtick.minor.size": 2,
            "xtick.minor.pad": 1,
            "xtick.minor.width": 0.5,
            "ytick.minor.size": 2,
            "ytick.minor.pad": 1,
            "ytick.minor.width": 0.5,
            "text.color": "#232324",
            "patch.edgecolor": "#232324",
            "patch.force_edgecolor": False,
            "hatch.color": "#232324",
            "axes.edgecolor": "#232324",
            "axes.labelcolor": "#232324",
            "xtick.color": "#232324",
            "ytick.color": "#232324",
        },
    )
