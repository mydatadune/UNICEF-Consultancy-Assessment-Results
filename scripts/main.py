from datetime import date
import sys
import logging
from pathlib import Path
from dataclasses import dataclass, is_dataclass, fields
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# -------------------------------
# User Profile Configuration
# -------------------------------
@dataclass
class CountriesConfig:
    path: str

@dataclass
class PopulationConfig:
    path: str
    worksheet: str
    header_row: int

@dataclass
class GDRConfig:
    path: str

@dataclass
class ReportConfig:
    path: str
    version: str
    date: date
    prefix_text: str
    suffix_text: str

@dataclass
class FilesConfig:
    root_path: str
    data_path: str
    output_path: str
    countries: CountriesConfig
    population: PopulationConfig
    gdr: GDRConfig
    report: ReportConfig

@dataclass
class ProcessingConfig:
    population_year: int

@dataclass
class UserProfile:
    files: FilesConfig
    processing: ProcessingConfig

# -------------------------------
# Constants
# -------------------------------
USER_PROFILE_FILENAME = 'user_profile.yaml'

# Countries data constants
COUNTRIES_ISO3CODE_COLUMN = "ISO3Code"
COUNTRIES_OFFICIAL_NAME_COLUMN = "OfficialName"
COUNTRIES_STATUS_U5MR_COLUMN = "Status.U5MR"
COUNTRIES_STATUS_U5MR_VALUES_ONTRACK = {"On Track", "Achieved"}
COUNTRIES_STATUS_COLUMN = "Status"
COUNTRIES_STATUS_VALUE_ONTRACK = "On-track"
COUNTRIES_STATUS_VALUE_OFFTRACK = "Off-track"

# Population data constants
POPULATION_ISO3CODE_COLUMN = "ISO3 Alpha-code"
POPULATION_TYPE_COLUMN = "Type"
POPULATION_TYPE_VALUE_COUNTRY = "Country/Area"
POPULATION_YEAR_COLUMN = "Year"
POPULATION_BIRTHS_COLUMN = "Births (thousands)"

# GDR data constants
GDR_REF_AREA_COLUMN = "REF_AREA:Geographic area"
GDR_REF_AREA_PATTERN = r"^[A-Za-z]{3}:"
GDR_INDICATOR_COLUMN = "INDICATOR:Indicator"
GDR_TIME_PERIOD_COLUMN = "TIME_PERIOD:Time period"
GDR_OBS_VALUE_COLUMN = "OBS_VALUE:Observation Value"
GDR_ISO3CODE_COLUMN = "ISO3Code"
GDR_WEIGHTED_VALUE_COLUMN = "Weighted Value"
WEIGHTED_AVERAGE_COLUMN = "Weighted Average"

# -----------------------------------------------
# Recursive dict Loader into dataclass hierarchy
# -----------------------------------------------
def from_dict(dataclass_type, data):
    """
    Recursively converts a dictionary into an instance of the specified dataclass type.

    This function iterates through the fields of the target dataclass. If a field's
    value in the input dictionary is itself a dictionary, and the field's type
    is also a dataclass, it recursively calls itself to instantiate the nested
    dataclass. Otherwise, it assigns the value directly.

    Args:
        dataclass_type: The dataclass type (e.g., `MyDataclass`) to instantiate.
                        This should be a class, not an instance.
        data: The dictionary containing the data to populate the dataclass.
              Keys in the dictionary should match the field names of the dataclass.

    Returns:
        An instance of `dataclass_type` populated with the data from the dictionary.
        If `dataclass_type` is not a dataclass, the original `data` dictionary is returned.

    Notes:
        - This function assumes that `data` contains values that are compatible with
          the dataclass field types (e.g., dates are already `datetime.date` objects,
          numbers are `int` or `float`). It does not perform type coercion beyond
          what Python's default assignment or `PyYAML`'s loading might provide.
        - It handles direct nested dictionaries that map to nested dataclasses.
    """
    if not is_dataclass(dataclass_type):
        return data
    kwargs = {}
    for field in fields(dataclass_type):
        field_value = data.get(field.name)
        field_type = field.type
        if isinstance(field_value, dict):
            kwargs[field.name] = from_dict(field_type, field_value)
        else:
            kwargs[field.name] = field_value
    return dataclass_type(**kwargs)

# -------------------------------
# Load User Profile
# -------------------------------
def load_user_profile(path: str) -> UserProfile:
    """
    Loads user profile configuration from a YAML file into a UserProfile dataclass.

    This function reads the YAML content from the specified path, parses it into
    a dictionary, and then converts that dictionary into a structured
    `UserProfile` dataclass instance using the `from_dict` helper function.

    Args:
        path: The file path to the `user_profile.yaml` configuration file.

    Returns:
        A `UserProfile` dataclass instance populated with the configuration values.
    """
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)

    user_profile = from_dict(UserProfile, config_dict)
    return user_profile

# Load User Profile
# Get the directory of the current script
# Go up one level to the parent directory (project root)
script_path = Path(__file__).resolve().parent
project_path = script_path.parent
user_profile_filepath = project_path / USER_PROFILE_FILENAME
user_profile = load_user_profile(user_profile_filepath)

# -------------------------------
# Load Countries Data
# -------------------------------
def load_countries_data(filepath: Path) -> pd.DataFrame:
    """
    Loads the Countries Excel file, performs initial data type conversion,
    and derives a new Status column.

    Args:
        filepath: The Countries Excel file.

    Returns:
        A DataFrame containing the loaded data with the additional Status column.
    """
    try:
        df = pd.read_excel(filepath, dtype=str)
    except FileNotFoundError as e:
        logging.critical(f"Countries file not found: '{filepath}'")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        logging.critical(f"Countries file is empty or has no valid data: '{filepath}'")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"An unexpected error occurred while reading Counries file '{filepath}': {e}")
        sys.exit(1)

    # Add a boolean OnTrack column based on the status values.
    df[COUNTRIES_STATUS_COLUMN] = np.where(
        df[COUNTRIES_STATUS_U5MR_COLUMN].isin(COUNTRIES_STATUS_U5MR_VALUES_ONTRACK),
        COUNTRIES_STATUS_VALUE_ONTRACK,
        COUNTRIES_STATUS_VALUE_OFFTRACK,
    )
    logging.info(f"Countries row count is {len(df)}")
    return df

# -------------------------------
# Load Population Data
# -------------------------------
def load_population_data(filepath: Path) -> pd.DataFrame:
    """
    Loads the Population Excel file and drop non-necessary columns.

    Args:
        filepath: The Population Excel file.

    Returns:
        A DataFrame containing the loaded data.
    """
    try:
        df = pd.read_excel(
            filepath,
            sheet_name=user_profile.files.population.worksheet,
            header=user_profile.files.population.header_row - 1,
            usecols=[
                POPULATION_ISO3CODE_COLUMN,
                POPULATION_TYPE_COLUMN,
                POPULATION_YEAR_COLUMN,
                POPULATION_BIRTHS_COLUMN
            ],
        )
    except FileNotFoundError as e:
        logging.critical(f"Population file not found: '{filepath}'")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        logging.critical(f"Population file is empty or has no valid data: '{filepath}'")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"An unexpected error occurred while reading Population file '{filepath}': {e}")
        sys.exit(1)

    # Retain only country rows.
    df = df.loc[df[POPULATION_TYPE_COLUMN] == POPULATION_TYPE_VALUE_COUNTRY]
    df.drop(POPULATION_TYPE_COLUMN, axis=1, inplace=True)

    # Retain only required year.
    df = df.loc[df[POPULATION_YEAR_COLUMN] == user_profile.processing.population_year]
    df.drop(POPULATION_YEAR_COLUMN, axis=1, inplace=True)

    return df

# -------------------------------
# Load GDR Data
# -------------------------------
def load_gdr_data(filepath: Path) -> pd.DataFrame:
    """
    Loads the GDR CSV file, calculates ISO3 country name, and retains only necessary rows.

    Args:
        filepath: The GDR CSV file.

    Returns:
        A DataFrame containing the loaded data.
    """
    try:
        df = pd.read_csv(
            filepath,
            usecols=[
                GDR_REF_AREA_COLUMN,
                GDR_INDICATOR_COLUMN,
                GDR_TIME_PERIOD_COLUMN,
                GDR_OBS_VALUE_COLUMN,
            ],
        )
    except FileNotFoundError as e:
        logging.critical(f"GDR file not found: '{filepath}'")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        logging.critical(f"GDR file is empty or has no valid data: '{filepath}'")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"An unexpected error occurred while reading GDR file '{filepath}': {e}")
        sys.exit(1)


    # Add an ISO3Code column.
    # Step 1: Keep rows starting with exactly 3 characters + colon.
    df = df.loc[df[GDR_REF_AREA_COLUMN].str.match(GDR_REF_AREA_PATTERN)]

    # Step 2: Extract the part after the colon, strip whitespace.
    df[GDR_ISO3CODE_COLUMN] = df[GDR_REF_AREA_COLUMN].str[:3]

    # Retain row with latest year for each country/indicator.
    df = df.loc[df.groupby([GDR_ISO3CODE_COLUMN, GDR_INDICATOR_COLUMN])[GDR_TIME_PERIOD_COLUMN].idxmax()]

    logging.info(f"GDR row count is {len(df)}")
    logging.info(f"GDR unique country count is {df[GDR_ISO3CODE_COLUMN].nunique()}")

    return df

# -------------------------------
# Validate Data
# -------------------------------
def validate_data(
    countries_df: pd.DataFrame,
    population_df: pd.DataFrame,
    gdr_df: pd.DataFrame,
):
    """
    Validates loaded dataframes, logging any inconsistancies.

    Args:
        countries_df: The Countries dataframe.
        population_df: The Population dataframe.
        gdr_df: The GDR dataframe.
    """
    # Verify GDR countries exist in Country data.
    missing_countries_df = gdr_df[~gdr_df[GDR_ISO3CODE_COLUMN].isin(countries_df[COUNTRIES_ISO3CODE_COLUMN])]
    if len(missing_countries_df) > 0:
        missing_countries_ref_area = missing_countries_df[GDR_REF_AREA_COLUMN].drop_duplicates().tolist()
        logging.error(f"GDR countries not found in Countries data: {missing_countries_ref_area}")
    else:
        logging.info("All GDR countries found in Countries data")

    # Verify GDR countries exist in Population data.
    missing_countries_df = gdr_df[~gdr_df[GDR_ISO3CODE_COLUMN].isin(population_df[POPULATION_ISO3CODE_COLUMN])]
    if len(missing_countries_df) > 0:
        missing_countries_ref_area = missing_countries_df[GDR_REF_AREA_COLUMN].drop_duplicates().tolist()
        logging.error(f"GDR countries not found in Population data: {missing_countries_ref_area}")
    else:
        logging.info("All GDR countries found in Population data")

# -------------------------------
# Merge Data
# -------------------------------
def merge_data(
    countries_df: pd.DataFrame,
    population_df: pd.DataFrame,
    gdr_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge dataframes.
    
    Args:
        countries_df: The Countries dataframe.
        population_df: The Population dataframe.
        gdr_df: The GDR dataframe.

    Returns:
        A DataFrame containing the merged data.
    """
    df = pd.merge(gdr_df, countries_df, left_on=GDR_ISO3CODE_COLUMN, right_on=COUNTRIES_ISO3CODE_COLUMN, how="inner")
    if GDR_ISO3CODE_COLUMN != COUNTRIES_ISO3CODE_COLUMN:
        df.drop(COUNTRIES_ISO3CODE_COLUMN, axis=1, inplace=True)

    df = pd.merge(df, population_df, left_on=GDR_ISO3CODE_COLUMN, right_on=POPULATION_ISO3CODE_COLUMN, how="inner")
    if GDR_ISO3CODE_COLUMN != POPULATION_ISO3CODE_COLUMN:
        df.drop(POPULATION_ISO3CODE_COLUMN, axis=1, inplace=True)

    logging.info("GDR data merged with Countries and Population data")
    return df

# -------------------------------
# Calculate Weighted Averages
# -------------------------------
def calculate_weighted_averages(
    input_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate weighted averages.

    Args:
        input_df: The dataframe after prior merging.

    Returns:
        A DataFrame containing the weighted average data grouped by status and indicator.
    """
    # Calculate weighted coverage per row.
    input_df[GDR_WEIGHTED_VALUE_COLUMN] = input_df[GDR_OBS_VALUE_COLUMN] * input_df[POPULATION_BIRTHS_COLUMN]

    # Sum columns per OnTrack/Indicator.
    grouped_sums_df = input_df.groupby([COUNTRIES_STATUS_COLUMN, GDR_INDICATOR_COLUMN])[[GDR_WEIGHTED_VALUE_COLUMN, POPULATION_BIRTHS_COLUMN]].sum()

    # Calculate the weighted average from the grouped sums
    weighted_df = (grouped_sums_df[GDR_WEIGHTED_VALUE_COLUMN] / grouped_sums_df[POPULATION_BIRTHS_COLUMN]).reset_index()

    # Rename for column clarity.
    weighted_df.rename(columns={0: WEIGHTED_AVERAGE_COLUMN}, inplace=True)

    logging.info("Calculated weighted averages")
    return weighted_df

# -------------------------------
# Create Report
# -------------------------------
def create_report(
    input_df: pd.DataFrame,
    report_filepath: Path,
    report_version: str,
    report_date: str,
):
    """
    Create report.

    Args:
        input_df: The weighted average dataframe.
        report_filepath: The path of the output pdf.
        report_version: Version string for embedding into report.
        report_date: Date string for embedding into report.
    """
    # Grouped Bar Chart: Each Indicator's 2 Status values side-by-side

    # Mapping of long indicator names to short codes.
    indicator_map = {
        'MNCH_ANC4: Antenatal care 4+ visits - percentage of women (aged 15-49 years) attended at least four times during pregnancy by any provider': 'ANC4',
        'MNCH_SAB: Skilled birth attendant - percentage of deliveries attended by skilled health personnel': 'SAB',
    }
    input_df[GDR_INDICATOR_COLUMN] = input_df[GDR_INDICATOR_COLUMN].map(indicator_map)

    # Pivot the DataFrame with 'Indicator' as index and 'Status' as columns
    # This will make the Indicators the main groups on the X-axis,
    # and Status bars will be displayed side-by-side for each.
    pivot_df = input_df.pivot(index=GDR_INDICATOR_COLUMN, columns=COUNTRIES_STATUS_COLUMN, values=WEIGHTED_AVERAGE_COLUMN)

    # Create figure and gridspec layout.
    fig = plt.figure(figsize=(8.5, 11))  # Standard letter size
    gs = GridSpec(3, 1, height_ratios=[1, 4, 1])  # Top, middle, bottom

    # --- Top (prefix) ---
    prefix_text = user_profile.files.report.prefix_text.format(version=report_version, date=report_date)
    ax_prefix = fig.add_subplot(gs[0])
    ax_prefix.axis('off')  # Hide axes
    ax_prefix.text(0.5, 0.5, prefix_text, ha='center', va='center', fontsize=10, wrap=True)

    # --- Middle (chart) ---
    ax_chart = fig.add_subplot(gs[1])
    pivot_df.plot(kind='bar', ax=ax_chart)
    ax_chart.set_title('Weighted Averages by Indicator and Status')
    ax_chart.set_ylabel('Weighted Average')
    ax_chart.set_xlabel('Indicator')
    ax_chart.set_xticklabels(pivot_df.index, rotation=0)
    ax_chart.legend(title='Status')

    # --- Bottom (suffix) ---
    suffix_text = user_profile.files.report.suffix_text.format(version=report_version, date=report_date)
    ax_suffix = fig.add_subplot(gs[2])
    ax_suffix.axis('off')
    ax_suffix.text(0.01, 0.9, suffix_text, ha='left', va='top', fontsize=9)

    # Save the plot to a PDF file.
    fig.tight_layout()
    fig.savefig(report_filepath)

    logging.info("Created report")

# -------------------------------
# Main
# -------------------------------
def main():
    # Initialize logging.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Initialize file locations.
    root_path = Path(user_profile.files.root_path)
    data_path = root_path / user_profile.files.data_path
    output_path = root_path / user_profile.files.output_path

    countries_filepath = data_path / user_profile.files.countries.path
    countries_df = load_countries_data(countries_filepath)

    population_filepath = data_path / user_profile.files.population.path
    population_df = load_population_data(population_filepath)

    gdr_filepath = data_path / user_profile.files.gdr.path
    gdr_df = load_gdr_data(gdr_filepath)

    # Process dataframes.
    validate_data(countries_df, population_df, gdr_df)
    merged_df = merge_data(countries_df, population_df, gdr_df)
    weighted_df = calculate_weighted_averages(merged_df)

    # Create report.
    report_version = user_profile.files.report.version
    report_date = user_profile.files.report.date.strftime("%Y-%m-%d")
    report_path = user_profile.files.report.path.format(version=report_version, date=report_date)
    report_filepath = output_path / report_path
    create_report(weighted_df, report_filepath, report_version, report_date)

if __name__ == "__main__":
    main()
