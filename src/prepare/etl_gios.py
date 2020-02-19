import pandas as pd
import time
import os
import random
import re
from pathlib import Path
import requests
import urllib.request
from bs4 import BeautifulSoup

from prepare import get_files_for_name_pattern


def get_gios_pollution_data_files(base_url: str, path_to_save: str) -> None:
    """
    Downloads GIOS pollution data into a folder.
    :param base_url: base url of GIOS archive page
    :param path_to_save: parent folder to store downloaded files
    :return: None
    """

    # Get page content and parse it
    response = requests.get(base_url)

    if response.status_code == 200:

        soup = BeautifulSoup(response.text, "html.parser")

        # Use the method .find to locate <ul> of id
        ul = soup.find("ul", {"id": "archive_files"})

        # Locate resources
        resources = []
        lis = ul.find_all('li')
        for li in lis:
            file_name = li.find("p", {"class": "archive_file_name"}).getText()
            file_url = li.find("a")['href'].split('/')
            resources.append((file_name, file_url[3]+'/'+file_url[4]))

        # Create a target folder if not exists
        target_folder_location = f"{path_to_save}"
        Path(target_folder_location).mkdir(parents=True, exist_ok=True)

        for resource in resources:

            # Define paths
            source_file_location = f"{base_url}/{resource[1]}"
            target_file_location = os.path.join(target_folder_location, resource[0]+'.zip')

            # Download a file
            urllib.request.urlretrieve(source_file_location, target_file_location)

            # Wait randomly 1-5 seconds (simulate manual download)
            time.sleep(random.randrange(1, 6))

            print("ok:", response.status_code, source_file_location)

    else:
        print("error:", response.status_code)


def get_pollutant_measures_for_locations(data_file_path: str,
                                         ems_codes: list,
                                         measure: str,
                                         year: str) -> pd.DataFrame:
    """
    Takes measures of a specified pollutant for a given year and emission measurement stations,
    averages the measurement values (for not NaNs only) and builds a set of aggregations. Takes
    into account metadata differences before and after 2016.
    :param data_file_path: full GIOS data file path
    :param ems_codes: emission measurement stations codes in scope
    :param measure: pollutant name
    :param year: year when measures were taken, this allows adjustments regarding data files
    structure changes
    :return: data frame with aggregated values for the pollutants from areas defined by ems_codes
    """

    years_conf1 = ['2016', '2017', '2018', '2019']
    years_conf2 = ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009',
                   '2010', '2011', '2012', '2013', '2014', '2015']

    datetime_col_name = "Datetime"
    header = 0

    # Adjust to different formats of data files (number of row with header)
    if year in years_conf1:
        header = 1  # read 2nd row as header
    elif year in years_conf2:
        header = 0

    df = pd.read_excel(data_file_path, header=header)  # read 2nd row as header
    df.rename(columns={df.columns[0]: datetime_col_name}, inplace=True)

    # Get columns defined in ems_codes and datetime
    cols_in_scope = ems_codes
    cols_in_scope.append(df.columns[0])  # add time column
    df = df.loc[:, df.columns.isin(cols_in_scope)]  # handle not existing columns

    # Remove first X rows as they contain metadata
    if year in years_conf1:
        df = df.iloc[4:, :]
    elif year in years_conf2:
        df = df.iloc[2:, :]

    cols = df.columns[1:]

    if year in years_conf1:
        # Replace commas with dots (in all columns but the first one - detatime)
        df[cols] = df[cols].apply(lambda x: x.str.replace(',', '.'))

    if len(cols) > 0:
        # Change columns type
        df[df.columns[0]] = df[df.columns[0]].apply(pd.to_datetime)
        df[cols] = df[cols].apply(pd.to_numeric)

    # Set datetime index
    df = df.set_index(df.columns[0])

    # Calculate statistics for the measure
    cols = df.columns
    df_return = pd.DataFrame(index=df.index.copy())

    # If the measurements are available from multiple stations
    if len(cols) >= 1:
        df_return[measure + '_mean'] = df[cols].mean(axis=1, skipna=True)
        df_return[measure + '_median'] = df[cols].median(axis=1, skipna=True)
        df_return[measure + '_min'] = df[cols].min(axis=1, skipna=True)
        df_return[measure + '_max'] = df[cols].max(axis=1, skipna=True)
        df_return[measure + '_std'] = df[cols].std(axis=1, skipna=True)
        df_return[measure + '_sum'] = df[cols].sum(axis=1, skipna=True)
        df_return[measure + '_obs_num'] = df.apply(lambda x: x.count(),
                                                   axis=1)  # count not-null values in a row

    return df_return


def build_gios_analytical_view(ems_codes: list,
                               years: list,
                               sampling_freq: str,
                               root_folder: str) -> pd.DataFrame:
    """
    Builds GIOS analytical view from XLSX files, for specific year and averaging period
    :param ems_codes: emission measurement stations code list for geographical filtering
    :param years: a list of of strings specifying years of observations in scope
    :param sampling_freq: can be '1g' for hourly or '24g' for daily averaging pollutants measures
    :param root_folder: a folder from which the data files search start (works recursively with subfolders)
    :return: data frame with full analytical view for the pollution data
    """

    df_full = pd.DataFrame()

    for year in years:

        file_search_pattern = year + '_*_' + sampling_freq + '.xlsx'
        files = get_files_for_name_pattern(root_folder, file_search_pattern)

        df_for_year = pd.DataFrame()

        print(f"Year: {year} - df_full.shape {df_full.shape}")

        for file in files:
            # Take measurement from a file name
            measurement_name = file.split('_')[1]

            # Manual corrections to inconsistent names created by data supplier
            file_name = os.path.basename(file)

            # Unify headers, instead of PM2.5 we should have PM25
            if re.search('PM2.5', file_name):
                measurement_name = 'PM25'

            print(f"File: {file} - measurement_name: {measurement_name}")

            # Gather data for a measurement
            df_measure = get_pollutant_measures_for_locations(file,
                                                              ems_codes,
                                                              measurement_name,
                                                              year)

            # Merge data frames on datetime index (add more columns for the specified time range)
            df_for_year = pd.merge(df_for_year,
                                   df_measure,
                                   how='outer',
                                   left_index=True,
                                   right_index=True)

        # Append new rows with new range of datetimes
        df_full = df_full.append(df_for_year,
                                 ignore_index=False,
                                 verify_integrity=True,
                                 sort=True)  # keep the appended df index intact
        print(f"----------------------------------------\n")

    return df_full
