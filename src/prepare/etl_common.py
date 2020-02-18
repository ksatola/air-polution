import os
import fnmatch
import zipfile


def extract_archived_data(source_dir: str, target_dir: str, file_search_pattern: str) -> None:
    """
    Extracts ZIP data files from a single folder and its sub-folders (recursively) to a target
    folder (with a flat structure).
    :param source_dir: root folder for the recursive search of files matching the file_search_pattern
    :param target_dir: folder to which the content of archived files is extracted
    :param file_search_pattern: wildcard filter determining files to be processed
    :return: None
    """

    # Walk recursively top-down the directory tree
    for dirpath, dirnames, files in os.walk(source_dir,
                                            topdown=True,
                                            followlinks=False):  # no symbolic links following
        print(f'Found directory: {dirpath}')
        for file_name in files:

            # Extract only files which meet the criteria
            if fnmatch.fnmatch(file_name, file_search_pattern):
                full_path = os.path.join(dirpath, file_name)

                # Extract archived content
                with zipfile.ZipFile(full_path, 'r') as zip_obj:
                    print(f'Extracting: {full_path}')
                    zip_obj.extractall(path=target_dir)


def get_files_for_name_pattern(folder: str,
                               file_search_pattern: str) -> list:
    """
    Walks recursively top-down the directory tree and finds all files which meet the filename
    pattern.
    :param folder: root folder for the recursive search of files matching the file_search_pattern
    :param file_search_pattern: wildcard filter determining files to be processed
    :return: list of files matching criteria with the full path
    """

    matching_files = []

    # Walk recursively top-down the directory tree
    for dirpath, dirnames, files in os.walk(folder, topdown=True,
                                            followlinks=False):  # no symbolic links following
        for file_name in files:

            # Extract only files which meet the criteria
            if fnmatch.fnmatch(file_name, file_search_pattern):
                full_path = os.path.join(dirpath, file_name)
                matching_files.append(full_path)

    return matching_files
