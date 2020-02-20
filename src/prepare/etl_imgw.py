import pandas as pd
import requests
import urllib.request
import time
from bs4 import BeautifulSoup
import os
from pathlib import Path
import random
import re
import glob


def get_imgw_yearly_weather_data_files(years: list, base_url: str, path_to_save: str) -> None:
    """
    Downloads IMGW weather data for specified years into a parent folder. Each year data will be saved in a separate sub-folder.
    :param years: list of years (as strings)
    :param base_url: root url containing weather data of certain type, like 'terminowe/synop'
    :param path_to_save: parent folder to store downloaded files (folders and files)
    :return: None
    """
    for year in years:
        url = f"{base_url}/{year}/"
        response = requests.get(url)

        if response.status_code == 200:

            # Parse returned HTML content
            soup = BeautifulSoup(response.text, "html.parser")
            a_tags = soup.findAll('a')[5:] # files start at 5th element of the list

            # Create a year folder if does not exist
            target_folder_location = f"{path_to_save}/{year}"
            Path(target_folder_location).mkdir(parents=True, exist_ok=True)

            for a_tag in a_tags:

                # Define paths
                filename = a_tag['href']
                source_file_location = f"{url}{filename}"
                target_file_location = os.path.join(target_folder_location, filename)

                # Download a file
                urllib.request.urlretrieve(source_file_location, target_file_location)

                # Wait randomly 1-5 seconds (simulate manual download)
                time.sleep(random.randrange(1, 6))

                print("ok:", response.status_code, source_file_location)
        else:
            print("error:", response.status_code)


def parse_imgw_metadata(file_input: str,
                        file_output: str,
                        input_encoding="cp1250",
                        output_encoding="utf-8") -> None:
    """
    Reads IMGW metadata weather data from a file, transforms it to a CSV format and saves as
    another file with specified encoding.
    :param file_input: full path to the metadata file
    :param file_output: full path where the transformed file is created
    :param input_encoding: file_input encoding
    :param output_encoding: file_output encoding
    :return: None
    """

    # Read file and remove multiple space between columns
    with open(file_input, 'r', encoding=input_encoding) as f:
        lines = f.readlines()
        new_lines = []
        for line in lines:
            line = re.sub(r"\n", "", line)  # remove new lines
            line = re.split('\s\s+', line)  # split elements by any number of white characters
            if len(line) == 2:  # empty lines are 1-element lists
                new_lines.append(line[0])

    output_str = ','.join(new_lines)

    # Save results to a new file
    with open(file_output, 'w', encoding=output_encoding) as f:
        f.write(output_str)


def build_imgw_analytical_view(source_dir: str, columns: str, file_search_pattern: str,
                               sms_codes: list) -> pd.DataFrame:
    """
    Builds IMGW analytical view for all files under source_dir, specific file_search_pattern,
    and synoptic measurement stations.
    :param source_dir: root search directory
    :param columns: location of CSV file with columns definition
    :param file_search_pattern: files matching the pattern to be processed
    :param sms_codes: list of synoptic measurement stations code
    :return:
    """
    # Find all files matching criteria in the folder
    files_to_process = glob.glob(source_dir + file_search_pattern)

    # Read columns file
    cols = pd.read_csv(columns, encoding='utf-8', sep=",")

    # Build analytical view
    df = pd.DataFrame()
    file_no = 1
    total_rows = 0

    for file in files_to_process:
        # Read a single file
        df_temp = pd.read_csv(file, names=cols.columns.tolist(), encoding='cp1250',
                              low_memory=False, header=None, index_col=None)

        # Fiter its rows to get measurements for sms_codes
        df_temp = df_temp.loc[df_temp['Kod stacji'].isin(sms_codes)]

        # Combine separate columns representing date/time into a single datetime column
        df_temp.rename(
            columns={'Rok': 'year', 'Miesiąc': 'month', 'Dzień': 'day', 'Godzina': 'hour'},
            inplace=True)
        datecols = ['year', 'month', 'day', 'hour']
        df_temp.index = pd.to_datetime(df_temp[datecols])
        df_temp.index.name = 'Datetime'

        # Remove not needed columns
        # df_temp = df_temp.drop(datecols, axis=1)

        df = df.append(df_temp)

        print(f"{file_no:0>4} -> Shape: {df_temp.shape}, file: {file}, total rows: {total_rows}")
        file_no += 1
        total_rows += df_temp.shape[1]

    return df
