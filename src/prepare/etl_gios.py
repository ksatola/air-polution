import time
import os
import random
from pathlib import Path
import requests
import urllib.request
from bs4 import BeautifulSoup


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
