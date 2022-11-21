# main libraries
import numpy as np
import pandas as pd
import seaborn as sns

# visualization libraries
import matplotlib.pyplot as plt
import plotly.express as px

from math import ceil
import math
from pywaffle import Waffle

import plotnine
from plotnine import *

# inferential statistics libraries
import statsmodels.stats.api as sms
import statsmodels.api as sm
import scipy.stats.distributions as dist
import scipy.stats as stats

# modeling libraries
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE 
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline as ImPipeline

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold , cross_validate, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import plot_precision_recall_curve, plot_roc_curve
from sklearn.metrics import PrecisionRecallDisplay, confusion_matrix, ConfusionMatrixDisplay, r2_score, roc_curve, RocCurveDisplay
from sklearn.metrics import brier_score_loss, mean_squared_error, roc_auc_score, log_loss, accuracy_score, confusion_matrix, precision_recall_curve
from sklearn.metrics import precision_score, plot_confusion_matrix, f1_score, recall_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from optuna.integration import LightGBMPruningCallback


# DATA CLEANING, FEATURE ENGINEERING


def missing_data(data: pd.DataFrame) -> pd.DataFrame:
    """Takes pd.DataFrame as an input and counts the percentage of NaN values in each column."""
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum() / data.isnull().count() * 100).sort_values(
        ascending=False
    )
    return pd.concat([total, percent], axis=1, keys=["Total", "Percent"])


def copy_df(df: pd.DataFrame) -> pd.DataFrame:
    """Takes pd.DataFrame as an input and returns it's copy"""
    return df.copy()


def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Takes pd.DataFrame as an input and drops all columns that have more than 50 percent of NaN values.
    params: df: pd.DataFrame, which must be cleaned"""
    thresh = len(df) * 0.5
    df.dropna(axis=1, thresh=thresh, inplace=True)
    return df


def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Takes to Data Frames and list of columns.
    Drops those columns
    Returns new dataframe
    """
    df = df.drop(columns=columns)
    return df


def drop_all_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Takes to Data Frames and drops all rows with NaN values."""
    df = df.dropna(axis=0)
    return df


def drop_nan_rows_from_certain_cols(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Takes pd.DataFrame and list of columns, drops rows with NaN values from those certain columns"""
    df.dropna(axis=0, subset=columns, inplace=True)
    return df


def lower_case_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Takes as an input pd.DataFrame and lowercase the names of all columns"""
    df.columns = df.columns.str.lower()
    return df


def new_col_count_binary(
    df: pd.DataFrame, new_col_name: str, columns: list
) -> pd.DataFrame:
    """Takes as an input pd.DataFrame, creates new column of the proportions of "1" in listed other columns.
    params: df: pd.DataFrame: usable, which columns to count and where to insert new column;
            new_col_name: str, the name of the new columns;
            columns: list - of binary columns, in which to count "1" values and divide by the len(columns);
    return: pd.DataFrame"""
    df[new_col_name] = df[columns].sum(axis=1).divide(len(columns))
    return df


def days_to_years(df: pd.DataFrame, column: str, new_column_name: str) -> pd.DataFrame:
    """Takes pd.DataFrame certain column values in negative days and returns in years."""
    df[new_column_name] = df[column].divide(-365)
    return df


def lowercase_values(df: pd.DataFrame) -> pd.DataFrame:
    """Takes column of the pd.DataFrame and lowercases values."""
    df = df.applymap(lambda s: s.lower() if type(s) == str else s)
    return df


def replace_char_underscore(df: pd.DataFrame) -> pd.DataFrame:
    """Takes pd.DataFrame and replaces 'space' and some characters with underscore."""
    df = df.applymap(lambda s: s.replace(" / ", "_") if type(s) == str else s)
    df = df.applymap(lambda s: s.replace(" ", "_") if type(s) == str else s)
    return df


def replace_characters(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Takes pd.DataFrame certain columns and replaces 'spaces \ space' with underscore."""
    df[columns] = df[columns].replace(" \ ", "_", regex=True)
    return df


def replace_space_underscore(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Takes certain columns of given pd.DataFrame and replaces 'space' with 'underscore'."""
    df[columns] = df[columns].replace(" ", "_", regex=True)
    return df


def replace_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Takes pd.DataFrame column and replaces extreme values below 0.5 and above .95 quantile with the closest values."""

    lower = df[column].quantile(0.05)
    upper = df[column].quantile(0.95)

    df[column] = df[column].clip(lower=lower, upper=upper)
    return df


def count_categorical(df: pd.DataFrame, group_var: str, df_name: str) -> pd.DataFrame:
    """Computes counts and normalized counts for each observation
    of `group_var` of each unique category in every categorical variable
    
    Parameters
    --------
    df : dataframe 
        The dataframe to calculate the value counts for.
        
    group_var : string
        The variable by which to group the dataframe. For each unique
        value of this variable, the final dataframe will have one row
        
    df_name : string
        Variable added to the front of column names to keep track of columns

    
    Return
    --------
    categorical : dataframe
        A dataframe with counts and normalized counts of each unique category in every categorical variable
        with one row for every unique value of the `group_var`.
        
    """

    categorical = pd.get_dummies(df.select_dtypes("object"))
    categorical[group_var] = df[group_var]
    categorical = categorical.groupby(group_var).agg(["sum", "mean"])
    column_names = []

    for var in categorical.columns.levels[0]:
        for stat in ["count", "count_norm"]:
            column_names.append(f"{df_name}_{var}_{stat}")

    categorical.columns = column_names

    return categorical


def count_categorical_norm(df: pd.DataFrame, group_var: str, df_name: str) -> pd.DataFrame:
    """Computes counts and normalized counts for each observation
    of `group_var` of each unique category in every categorical variable
    
    Parameters
    --------
    df : dataframe 
        The dataframe to calculate the value counts for.
        
    group_var : string
        The variable by which to group the dataframe. For each unique
        value of this variable, the final dataframe will have one row
        
    df_name : string
        Variable added to the front of column names to keep track of columns

    
    Return
    --------
    categorical : dataframe
        A dataframe with normalized counts of each unique category in every categorical variable
        with one row for every unique value of the `group_var`.
        
    """

    categorical = pd.get_dummies(df.select_dtypes("object"))
    categorical[group_var] = df[group_var]
    categorical = categorical.groupby(group_var).agg("mean")

    column_names = []

    for var in categorical.columns:
        column_names.append(f"{df_name}_{var}_normalized")

    categorical.columns = column_names

    return categorical


def log_feature(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Takes as an input pd.DataFrame certain column and returns dataframe with new column of log_feature"""
    df["log_" + feature] = np.log(df[feature] + 0.0001)
    return df


def cyclic_data_to_sin_cos(df: pd.DataFrame, feature: str):
    df["cos_" + feature] = np.cos(2 * math.pi * df[feature] / df[feature].max())
    df["sin_" + feature] = np.sin(2 * math.pi * df[feature] / df[feature].max())
    return df


def weekdays_to_numbers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Takes column of the pd.DataFrame with categorical weekdays and returns numbers"""
    df = df.replace(
        {
            column: {
                "monday": 1,
                "tuesday": 2,
                "wednesday": 3,
                "thursday": 4,
                "friday": 5,
                "saturday": 6,
                "sunday": 7,
            }
        }
    )
    return df


def change_negative_to_positive(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df[column] = df[column].apply(lambda x: np.abs(x))
    return df


def change_dtype_category(df: pd.DataFrame) -> pd.DataFrame:
    """Takes pd.Dataframe and changes dtypes from 'object' to 'category'"""
    for column in df.columns:
        col_type = df[column].dtype
        if col_type == "object" or col_type.name == "category":
            df[column] = df[column].astype("category")
        else:
            continue
    return df


# VISUALIZATIONS


def plot_total_percentage_of_loans(df: pd.DataFrame, column: str, title: str) -> None:
    """Takes certain pd.DataFrame column and plots waffle chart with percentage of different values
    params: df: pd.DataFrame to use;
            columns: str - name of the main plotted column;
            title: str - title of the waffle chart"""

    fig = plt.figure(
        figsize=(10, 10),
        FigureClass=Waffle,
        rows=5,
        values=df[column],
        colors=["green", "blue"],
        title={"label": title, "loc": "left"},
        icons="child",
        icon_size=30,
        icon_legend=True,
        labels=["{0} ({1:.2f}%)".format(k, v) for k, v in zip(df.index, df[column])],
        legend={
            "loc": "lower left",
            "bbox_to_anchor": (0, -0.4),
            "ncol": len(df),
            "framealpha": 0,
        },
    )
    fig.gca().set_facecolor("#EEEEEE")
    fig.set_facecolor("#EEEEEE")


def plot_boxplot(df: pd.DataFrame, x: str, y: str, title: str) -> None:
    """Takes as an input name of the pd.DataFrame, certain columns to plot on axis and plots boxplot.

    :param: df: the name of the pd.DataFrame to use;
            x: str - name of the column to plot on X axis;
            y: str - name of the column to plot on Y axis;
            title: str - title of the whole chart;
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_yscale("linear")
    ax.xaxis.grid(True)
    ax.set(ylabel="")
    ax.set(xlabel="")
    sns.boxplot(x=x, y=y, data=df, palette="viridis")

    sns.despine(trim=True, left=True)
    ax.set_title(title)


def plot_kde(df: pd.DataFrame, feature1: str, feature2: str) -> None:
    """ Plots KDE plot of two numerical features.
        params: df: usable pd.DataFrame;
                feature1: str - name of the numerical column, which values to plot;
                feature2: str - name of the categorical binary column of "yes" and "no", which values to use as a "hue";
        """
    plt.figure(figsize=(15, 5))
    plt.title(f"KDE Plot: {feature1} vs. {feature2}", fontsize=30, fontweight="bold")
    ax = sns.kdeplot(
        df[df[feature2] == 1][feature1],
        color="green",
        label="had payment problem",
        lw=2,
        legend=True,
    )
    ax1 = sns.kdeplot(
        df[df[feature2] == 0][feature1],
        color="blue",
        label="no payment problem",
        lw=2,
        legend=True,
    )
    legend = ax.legend(loc="upper right")
    ax.yaxis.grid(True)
    sns.despine(right=True, left=True)
    plt.tight_layout()


def plot_countplot(
    df: pd.DataFrame, feature1: str, feature2: str, title: str, size: tuple
) -> None:
    """Takes as an input name of the pd.DataFrame and names of needed columns and plots a count plot.

    :param: df: the name of the pd.DataFrame to use;
            feature1: str - name of the columns to plot on X axis;
            feature2: str - name of the columns to plot as a 'hue' parameter (to do some cross counting);
            title: str -  final title (name) of the whole plot;
            size: tuple: size of the plot.
    """

    fig, ax = plt.subplots(figsize=size)

    sns.countplot(
        x=feature1,
        hue=feature2,
        data=df,
        order=df[feature1].value_counts().index,
        palette="viridis",
    )

    if feature2 != None:
        ax.legend(loc="upper right", title=feature2)
        ax.bar_label(ax.containers[1])

    ax.set_title(title)
    ax.set(ylabel="")
    ax.set(xlabel="")
    #     plt.xticks(rotation=45)
    ax.bar_label(ax.containers[0])
    sns.despine(trim=True, left=True)
    plt.tight_layout()


def make_crosstab_number(
    df: pd.DataFrame, feature1: str, feature2: str
) -> pd.DataFrame:
    """Takes as an input name of the pd.DataFrame and certain features to use, 
        outputs Pd.DataFrame with cross count of the values in these features.

        :param: df: the name of the pd.DataFrame to use;
                feature1: str - name of the first column which values to cross count;
                feature2: str - name of the second column which values to cross count;
        :return: pd.DataFrame with statistics of cross counted values from both used columns.
        """
    return pd.crosstab(df[feature1], df[feature2])


def make_crosstab_percent(
    df: pd.DataFrame, feature1: str, feature2: str
) -> pd.DataFrame:
    """Takes as an input name of the pd.DataFrame and certain features to use, 
        outputs Pd.DataFrame with cross count and turn into percent of the values in these features.

        :param: df: the name of the pd.DataFrame to use;
                feature1: str - name of the first column which values to cross count;
                feature2: str - name of the second column which values to cross count;
        :return: pd.DataFrame with percent of cross counted values from both used columns.
        """
    return pd.crosstab(df[feature1], df[feature2], normalize="index") * 100


def multiple_violinplots(df: pd.DataFrame, feature1: str) -> None:
    """Plots multiple violin plots of 1 categorical feature and all other - numerical features.

        You need to make a new df only with needed columns: 1 categorical, other- numerical.
        params: df: pd.DatFrame, name of the data frame to use.
                Feature1: str, categorical feature in which to compare numerical values of all other features.
        """
    plt.figure(figsize=(15, 10))
    for i in range(1, len(df.columns)):
        plt.subplot(int(len(df.columns) / 3) + 1, 3, i)
        ax = sns.violinplot(
            x=feature1, y=df.columns[i - 1], data=df, palette="viridis", dodge=True
        )
        ax.xaxis.grid(True)
        ax.set_title(f"{df.columns[i-1]}")
        ax.set(xlabel=feature1)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set(ylabel="")
        sns.despine(right=True, left=True)
    plt.show()


def plot_heatmap(df: pd.DataFrame, title: str) -> None:
    """Takes as an input pd.DataFrame and plot the heatmap with correlation coefficients between all features.

        :param: df: the name of the pd.DataFrame to use;
                title: str - title of the whole heatmap.
        """
    sns.set_theme(style="white")
    corr = df.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))

    f, ax = plt.subplots(figsize=(10, 8))

    cmap = sns.diverging_palette(150, 275, as_cmap=True)
    heatmap = sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        annot=True,
    )

    heatmap.set_title(
        title, fontdict={"fontsize": 16}, pad=12,
    )
    plt.xlabel("")
    plt.ylabel("")


def plot_countplot_vertical(
    df: pd.DataFrame, feature1: str, feature2: str, title: str, size: tuple
) -> None:
    """Takes as an input name of the pd.DataFrame and names of needed columns and plots a count plot.

    :param: df: the name of the pd.DataFrame to use;
            feature1: str - name of the columns to plot on X axis;
            feature2: str - name of the columns to plot as a 'hue' parameter (to do some cross counting);
            title: str -  final title (name) of the whole plot;
            size: tuple - size of the plot.
    """

    fig, ax = plt.subplots(figsize=size)

    sns.countplot(
        y=feature1,
        hue=feature2,
        data=df,
        order=df[feature1].value_counts().index,
        palette="viridis",
    )

    if feature2 != None:
        ax.legend(loc="lower right", title=feature2)
        ax.bar_label(ax.containers[1])

    ax.set_title(title)
    ax.set(ylabel="")
    # ax.set(xlabel=feature1)
    ax.bar_label(ax.containers[0])
    sns.despine(trim=True, left=True)
    plt.tight_layout()


def plot_barplot(df: pd.DataFrame, feature: str, title: str, size: tuple) -> None:
    """Takes as an input pd.DataFrame and it's column name and plots simple bar plot. 
    On x axis is plotted index of the given dataframe.
    params: df: pd.DataFrame to use;
            features: str - column to plot on y axis.
            title: str - name of the chart;
            size: tuple: size of the plot.
            """
    plt.figure(figsize=size)
    ax = sns.barplot(x=df.index, y=df[feature], palette="viridis")
    plt.title(title)
    ax.set(ylabel="")
    plt.xticks(rotation=45)
    ax.bar_label(ax.containers[0])
    sns.despine(trim=True, left=True)


def multiple_kde(df: pd.DataFrame, feature1: str) -> None:
    """Plots multiple kde plots of 1 categorical feature (binary) and all other - numerical features.

    You need to make a new df only with needed columns: 1 categorical (binary), other- numerical.
    params: df: pd.DatFrame, name of the data frame to use.
            feature1: str, categorical feature in which to compare numerical values of all other features.
    """
    for i in range(1, len(df.columns)):
        sns.set_theme(style="darkgrid")
        plt.figure(figsize=(20, 5))
        plt.subplot(int(len(df.columns) / 2) + 1, 2, i)
        plt.title(
            f"KDE Plot: {feature1} vs. {df.columns[i-1]}",
            fontsize=10,
            fontweight="bold",
        )
        ax = sns.kdeplot(
            df[df[feature1] == 1][df.columns[i - 1]],
            color="green",
            label="had payment problem",
            lw=2,
            legend=True,
        )
        ax1 = sns.kdeplot(
            df[df[feature1] == 0][df.columns[i - 1]],
            color="blue",
            label="no payment problem",
            lw=2,
            legend=True,
        )
        legend = ax.legend(loc="upper right")
        ax.set(xlabel="")
        ax.set(ylabel="")
        sns.despine(right=True, left=True)
        plt.tight_layout()
        plt.show()


def plot_kde(df: pd.DataFrame, feature1: str, feature2: str) -> None:
    """ Plots KDE plot of two numerical features.
        params: df: usable pd.DataFrame;
                feature1: str - name of the numerical column, which values to plot;
                feature2: str - name of the categorical binary column of "yes" and "no", which values to use as a "hue";
        """
    plt.figure(figsize=(15, 5))
    plt.title(f"KDE Plot: {feature1} vs. {feature2}", fontsize=30, fontweight="bold")
    ax = sns.kdeplot(
        df[df[feature2] == 1][feature1],
        color="green",
        label="had payment problem",
        lw=2,
        legend=True,
    )
    ax1 = sns.kdeplot(
        df[df[feature2] == 0][feature1],
        color="blue",
        label="no payment problem",
        lw=2,
        legend=True,
    )
    legend = ax.legend(loc="upper right")
    ax.yaxis.grid(True)
    sns.despine(right=True, left=True)
    plt.tight_layout()


def plot_box_stripplot(df: pd.DataFrame, x: str, y: str, title: str) -> None:
    """Takes as an input name of the pd.DataFrame, certain columns to plot on axis and plots boxplot+stripplot together.

    :param: df: the name of the pd.DataFrame to use;
            x: str - name of the column to plot on X axis;
            y: str - name of the column to plot on Y axis;
            title: str - title of the whole chart;
    """
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_yscale("linear")
    ax.xaxis.grid(True)
    ax.set(ylabel="")
    ax.set(xlabel="")
    sns.boxplot(x=x, y=y, data=df, palette="viridis")

    sns.stripplot(x=x, y=y, data=df, palette="GnBu", size=5, edgecolor="gray")

    sns.despine(trim=True, left=True)
    ax.set_title(title)


def plot_stacked_barchart_plotly(
    df: pd.DataFrame, x: str, y: list, title: str, legend_title: str
) -> None:
    """Takes as an input name of the pd.Dataframe, needed columns, titles and plots a stacked bar chart of percentage
        
        param: df: the name of the pd.DataFrame to use;
               x: str - name of the column to plot on X axis;
               y: str - list of names (str) of the columns with percentage to plot on Y axis;
               title: str - title of the whole chart;
               legend_title: str - to rename the legend.
        """
    fig = px.bar(
        df,
        x=x,
        y=y,
        text_auto=True,
        color_discrete_map={y[0]: "#b5de2b", y[1]: "#26828e"},
        title=title,
    )
    fig.update_yaxes(title="")
    fig.update_xaxes(title="")
    fig.update_layout(legend_title=legend_title)
    fig.show()


# INFERENTIAL STATISTICAL ANALYSIS


class DiffMeans:
    """Module DiffMeans proceeds the needed table and all calculations
    for inferential statistical analysis of two the difference in means of to populations.
    Attribute:
    - df
    - feature1: first subgroup;
    - feature2: second subgroup;

    Methods of this class:
    - make_table;
    - diff_of_means;
    - sample_size_needed;
    - t_statistics (p_value);
    - conf_interval_of_difference"""

    def __init__(self, df: pd.DataFrame, feature1: str, feature2: str) -> None:
        self._df = df
        self._feature1 = feature1
        self._feature2 = feature2

    def make_populations(self) -> pd.DataFrame:
        self._first_pop = pd.DataFrame(
            self._df[self._df[self._feature1] == 0][self._feature2]
        )
        self._second_pop = pd.DataFrame(
            self._df[self._df[self._feature1] == 1][self._feature2]
        )
        return self._first_pop, self._second_pop

    def make_table(self) -> pd.DataFrame:
        """Creates a table - pd.DataFrame that helps to calculate the difference of means, estimated std."""
        self._table = self._df.groupby(self._feature1)[self._feature2].agg(
            ["count", "mean", "std"]
        )
        return self._table

    def diff_of_means(self) -> float:
        """Calculates the difference of two means."""
        self._table.reset_index(inplace=True)
        self._diff = self._table["mean"][0] - self._table["mean"][1]
        return self._diff

    def sample_size_needed(self) -> None:
        """Calculates the required sample size to avoid p-hacking"""
        self._table.reset_index(inplace=True)
        est_std = np.sqrt((self._table["std"][0] ** 2 + self._table["std"][1] ** 2) / 2)
        effect_size = self._diff / est_std
        required_n = sms.NormalIndPower().solve_power(
            effect_size, power=0.8, alpha=0.05, ratio=1, alternative="larger"
        )
        required_n = ceil(required_n)
        print(f"Required sample size:{required_n}")

    def z_statistics(self, approach: str) -> None:
        """Calculate the test statistic.
        params: approach: str - pooled or unequal approach."""
        col1 = sms.DescrStatsW(self._first_pop[self._feature2])
        col2 = sms.DescrStatsW(self._second_pop[self._feature2])

        cm_obj = sms.CompareMeans(col1, col2)

        zstat, z_pval = cm_obj.ztest_ind(alternative="larger", usevar=approach, value=0)
        print(f"Z-statistic: {zstat.round(3)}, p-value: {z_pval.round(3)}")

    def conf_interval_of_difference(self, approach: str) -> None:
        """Calculates the confidence interval of the difference of two population means.
        params: approach: str: pooled or unequal approach."""
        self._table.reset_index(inplace=True)
        cm = sms.CompareMeans(
            sms.DescrStatsW(self._first_pop[self._feature2]),
            sms.DescrStatsW(self._second_pop[self._feature2]),
        )
        print(cm.tconfint_diff(usevar=approach))


if __name__ == "__main__":
    DiffMeans()


class DiffProportions:
    """
    Module Diff_2_proportions proceeds the needed table and all calculations
    for inferential statistical analysis of two proportions.

    Attribute:
    - df;
    - feature1: subgroups of people, proportion of which interests us (like gender);
    - feature2: feature by which we calculate the proportion and it's difference of the feature1 (like: stroke: yes);

    Methods of this class:
    - make_table;
    - total_proportion;
    - diff_of_proportions;
    - sample_size_needed;
    - std_error;
    - t_statistics;
    - p_value;
    - conf_interval_of_difference
    """

    def __init__(self, df: pd.DataFrame, feature1: str, feature2: str) -> None:
        self._df = df
        self._feature1 = feature1
        self._feature2 = feature2

    def make_table(self) -> pd.DataFrame:
        """Creates a table - pd.DataFrame that helps to calculate the standard error."""

        self._table = self._df.groupby(self._feature1)[self._feature2].agg(
            [lambda z: np.mean(z == 1), "size"]
        )
        self._table.columns = ["proportion", "total_count"]
        print(f"Table of {self._feature2} per each group of {self._feature1}")
        return self._table

    def total_proportion(self) -> float:
        """Calculates the total proportion of feature2 together in all groups of feature1."""

        self._total_proportion = (self._df[self._feature2] == 1).mean()
        print(f"Total proportion of {self._feature2} cases in the dataset:")
        return self._total_proportion

    def diff_of_proportions(self) -> float:
        """Calculates the difference in proportions"""

        self._diff = self._table.proportion.iloc[0] - self._table.proportion.iloc[1]
        print("Difference of two independent proportions:")
        return self._diff

    def sample_size_needed(self) -> None:
        """Calculates the required sample size to avoid p-hacking"""

        effect_size = sms.proportion_effectsize(
            self._table.proportion.iloc[0], self._table.proportion.iloc[1]
        )

        required_n = sms.NormalIndPower().solve_power(
            effect_size, power=0.8, alpha=0.05, ratio=1
        )
        required_n = ceil(required_n)
        print(f"Required sample size:{required_n}")

    def std_error(self):
        """Calculating standard error"""

        self._variance = self._total_proportion * (1 - self._total_proportion)
        self._standard_error = np.sqrt(
            self._variance
            * (
                1 / self._table.total_count.iloc[0]
                + 1 / self._table.total_count.iloc[1]
            )
        )
        return self._standard_error

    def t_statistics(self) -> float:
        """Calculate the test statistic"""

        hypothesized_estimate = 0
        self._test_stat = (self._diff - hypothesized_estimate) / self._standard_error
        print("Computed Test Statistic is:")
        return self._test_stat

    def p_value(self) -> float:
        """Calculate the  p-value, for 1 tail testing only"""

        self._pvalue = dist.norm.cdf(-np.abs(self._test_stat))
        print("Computed P-value is")
        return self._pvalue

    def conf_interval_of_difference(self):
        """Calculates the confidence interval of the difference of two proportions"""

        se_no = np.sqrt(
            self._table.proportion.iloc[0]
            * (1 - self._table.proportion.iloc[0])
            / self._table.total_count.iloc[0]
        )
        se_yes = np.sqrt(
            self._table.proportion.iloc[1]
            * (1 - self._table.proportion.iloc[1])
            / self._table.total_count.iloc[1]
        )

        se_diff = np.sqrt(se_no ** 2 + se_yes ** 2)

        self._lcb = self._diff - 2 * se_diff
        self._ucb = self._diff + 2 * se_diff
        print("CI in proportion of stroke cases among female and male:")
        return self._lcb, self._ucb


if __name__ == "__main__":
    DiffProportions()


# Modeling


def base_line(
    X: pd.DataFrame, y: pd.DataFrame, preprocessor: np.array, resample: SMOTE
) -> pd.DataFrame:
    """
        Takes as an input X (all usable predictors) and y (outcome, dependent variable) pd.DataFrames.
        The function performs cross validation with different already selected models.
        Returns metrics and results of the models in pd.DataFrame format.

        :param: X - pd.DataFrame of predictors(independent features);
                y - pd.DataFrame of the outcome;
                preprocessor: ColumnTransformer with all needed scalers, transformers;
                resample: resampler from SMOTE() with different parameters.
        """

    balanced_accuracy = []
    roc_auc = []
    accuracy = []
    recall = []
    precision = []
    f1_score = []
    fit_time = []
    classifiers = [
        "Logistic regression",
        "Decision Tree",
        "Random Forest",
        "SVC",
        "KNN",
        "XGB classifier",
        "LGBM classifier",
    ]

    models = [
        LogisticRegression(solver="saga", n_jobs=-1),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=100),
        SVC(),
        KNeighborsClassifier(),
        XGBClassifier(n_jobs=-1),
        LGBMClassifier(n_jobs=-1),
    ]

    for model in models:
        pipeline = ImPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("resample", resample),
                ("classifier", model),
            ]
        )
        result = cross_validate(
            pipeline,
            X,
            y,
            cv=3,
            scoring=(
                "balanced_accuracy",
                "accuracy",
                "f1_macro",
                "recall_macro",
                "precision_macro",
                "roc_auc",
            ),
        )
        fit_time.append(result["fit_time"].mean())
        balanced_accuracy.append(result["test_balanced_accuracy"].mean())
        accuracy.append(result["test_accuracy"].mean())
        recall.append(result["test_recall_macro"].mean())
        precision.append(result["test_precision_macro"].mean())
        f1_score.append(result["test_f1_macro"].mean())
        roc_auc.append(result["test_roc_auc"].mean())
        print(f"Done {model}")

    base_models = pd.DataFrame(
        {
            "Balanced accuracy": balanced_accuracy,
            "Accuracy": accuracy,
            "Recall": recall,
            "Precision": precision,
            "f1": f1_score,
            "Roc Auc": roc_auc,
            "Fit_time": fit_time,
        },
        index=classifiers,
    )
    base_models = base_models.style.background_gradient(cmap="Greens")
    return base_models


def plot_classifier_scores(
    model, X: pd.DataFrame, y: pd.DataFrame, predictions: np.array, target_labels: list
):
    """Plots the Confusion matrix and classification report from scikit-learn.
        
        :param: model - chosen model, modeled Pipeline from sklearn, on which data is trained.
                X - pd.DataFrame, X_train, X_validation, X_test data, which on to predict and plot the prediction 
                result.
                y - pd.DataFrame, the outcome, dependent variable: y_train. y_val, y_test, what to predict.
                predictions: y_hat, predictions from the model.
        """
    cmap = sns.dark_palette("seagreen", reverse=True, as_cmap=True)
    plot_confusion_matrix(
        model, X, y, normalize="true", cmap=cmap, display_labels=target_labels
    )
    plt.title("Confusion Matrix: ")
    plt.show()
    print(classification_report(y, predictions, target_names=target_labels))

    print()


def feature_names(
    module, numerical_features: list, binary_features: list, remainder_features: list
) -> list:
    """
    Takes trained model.
    Extracts and returns feature name from preprocessor.
    """
    categorical = list(
        module.named_steps["preprocessor"]
        .transformers_[1][1]
        .named_steps["encoder"]
        .get_feature_names()
    )

    cat_all = numerical_features + categorical + binary_features + remainder_features

    return cat_all


def logistic_regression_objective(
    trial, X: pd.DataFrame, y: pd.DataFrame, preprocessor: Pipeline, resampler: SMOTE
) -> float:
    """Logistic regression hyper parameter searcher.
        
        Takes as an input X (independent variables), y(outcome) pd.DataFrame and a Pipeline with 
        preprocessors, transformers and certain model, fits the given data and searches for the best 
        hyper parameters.

        :param: X: pd.DataFrame with features;
                y: pd.DataFrame with outcome (dependent variable);
                model: sklearn.Pipeline with all needed transformers, preprocessors and chosen main model, in this
                example - Logistic Regression;
                resampler - SMOTE.
    """
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    if penalty == "elasticnet":
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
    elif penalty == "l1":
        l1_ratio = 1.0
    else:
        l1_ratio = 0.0

    C = trial.suggest_loguniform("C", 1e-5, 100)
    class_weight = trial.suggest_categorical("class_weight", ["balanced", None])

    model = LogisticRegression(
        solver="saga",
        penalty=penalty,
        C=C,
        class_weight=class_weight,
        l1_ratio=l1_ratio,
        random_state=123,
    )

    pipeline = ImPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("resample", resampler),
            ("estimator", model),
        ]
    )

    score = cross_val_score(pipeline, X, y, cv=2, scoring="roc_auc", n_jobs=-1).mean()

    return score


def LGBM_objective(trial, X, y) -> dict:
    """
    Takes x and y and dataframes
    Performs model training.
    Returns best parameters for particular model.
    """

    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 0, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 50),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
    }

    model = LGBMClassifier(objective="binary", **param_grid,)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="binary_logloss",
        early_stopping_rounds=100,
        callbacks=[LightGBMPruningCallback(trial, "binary_logloss")],
    )

    preds = model.predict(X_valid)
    cv_scores[idx] = log_loss(y_valid, preds)

    return np.mean(cv_scores)


def get_calibration_curve_values(
    model: Pipeline, 
    X_train: pd.DataFrame, 
    y_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_test: pd.DataFrame) -> tuple:
    """Takes model with data as an input and returns calibration curve values"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    prob_pos = model.predict_proba(X_test)[:, 1]
    model_score = brier_score_loss(y_test, y_pred, pos_label=y.max())

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, prob_pos, n_bins=10
    )

    return (fraction_of_positives, mean_predicted_value, model_score)


def optimal_threshold_ROC_AUC_G_mean(
    y: np.array, y_hat: np.array
    ):
    """Takes target and predictions, plot ROC_AUC and gets the G_mean of TPR/FPR: optimal threshold."""
    # Create the ROC curve
    fpr, tpr, thresholds = roc_curve(y, y_hat)

    # Plot the ROC curve
    df_fpr_tpr = pd.DataFrame({"FPR": fpr, "TPR": tpr, "Threshold": thresholds})
    df_fpr_tpr.head()

    # Calculate the G-mean
    gmean = np.sqrt(tpr * (1 - fpr))

    # Find the optimal threshold
    index = np.argmax(gmean)
    thresholdOpt = round(thresholds[index], ndigits=4)
    gmeanOpt = round(gmean[index], ndigits=4)
    fprOpt = round(fpr[index], ndigits=4)
    tprOpt = round(tpr[index], ndigits=4)
    print(f"Best Threshold: {thresholdOpt} with G-Mean: {gmeanOpt}")
    print(f"FPR: {fprOpt}, TPR: {tprOpt}")

    # Create data viz
    plotnine.options.figure_size = (8, 4.8)
    (
        ggplot(data=df_fpr_tpr)
        + geom_point(aes(x="FPR", y="TPR"), size=0.4)
        +
        # Best threshold
        geom_point(aes(x=fprOpt, y=tprOpt), color="#981220", size=4)
        + geom_line(aes(x="FPR", y="TPR"))
        + geom_text(
            aes(x=fprOpt, y=tprOpt),
            label="Optimal threshold \n for class: {}".format(thresholdOpt),
            nudge_x=0.14,
            nudge_y=-0.10,
            size=10,
            fontstyle="italic",
        )
        + labs(title="ROC Curve")
        + xlab("False Positive Rate (FPR)")
        + ylab("True Positive Rate (TPR)")
        + theme_minimal()
    )


def calibration_plot_isotonic(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> None:
    """Takes certain model and data and plots calibration plots."""
    sns.set_style("ticks")
    fig = plt.figure(figsize=(16, 7))
    ax = fig.add_subplot()
    ax.plot([0, 1], [0, 1], "b--", label="Perfectly calibrated", color="#000000")
    ax.set_ylabel("Fraction of positives")
    ax.set_xlabel("Mean predicted value")
    ax.set_title("Calibration plot (reliability curve)")

    (
        fraction_of_positives,
        mean_predicted_value,
        model_score,
    ) = get_calibration_curve_values(model, X_train, y_train, X_test, y_test)
    ax.plot(
        mean_predicted_value,
        fraction_of_positives,
        "s-",
        label="%s (%1.3f)" % ("LGBM", model_score),
        color="blue",
    )

    calibrator_isotonic = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
    fp, mpv, score = get_calibration_curve_values(
        calibrator_isotonic, X_train, y_train, X_test, y_test
    )
    ax.plot(mpv, fp, "s-", label="%s (%1.3f)" % ("LGBM Isotonic", score), color="green")

    ax.legend(loc="lower right")
    sns.despine()
    plt.show()