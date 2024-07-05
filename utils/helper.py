import ast
import logging
import os

import numpy as np


def setup_logging(log_dir, script_name, logging_level):
    """
    Set up a custom logger for a given script

    Args:
        log_dir (str): directory where logs should be saved
        script_name (str): script name
    """
    log_filename = os.path.join(
        log_dir, f'{os.path.splitext(os.path.basename(script_name))[0]}.log'
    )

    logging.basicConfig(
        level=logging_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_filename, mode='w'), logging.StreamHandler()],
    )


def convert_to_binary(input_string):
    """
    Convert string of available bands to a binary integer. The standard order of bands is 'ugriz'.
    2 -> present, 1 -> absent.

    Args:
        input_string (str): input string of available bands.

    Returns:
        int: number representing the availability of bands.
    """
    # Define the standard order of letters
    standard_letters = 'ugriz'
    # Initialize the result as an empty string
    result = ''
    # Iterate over each letter in the standard order
    for letter in standard_letters:
        # Append '2' if the letter is in the input string, '1' otherwise
        if letter in input_string:
            result += '2'
        else:
            result += '1'
    # Convert the binary string to a decimal integer
    return np.int32(result)


def split_tile_nums(df):
    """
    Split the tile column into two separate columns.

    Args:
        df (dataframe): A pandas DataFrame containing the metadata for each cutout.

    Returns:
        dataframe: Dataframe with the tile column split into two separate columns.
    """
    # Split the tuple into two separate columns
    if isinstance(df['tile'][0], str):
        df['tile'] = df['tile'].apply(ast.literal_eval)
    df['tile_num1'], df['tile_num2'] = zip(*df['tile'])
    # Drop tile column
    df.drop('tile', axis=1, inplace=True)
    return df


def tensor_compatible(df):
    """
    Convert the bands column to a binary integer and split the tile column into two separate columns.

    Args:
        df (dataframe): A pandas DataFrame containing the metadata for each cutout.

    Returns:
        dataframe: The torch compatible DataFrame.
    """
    # Convert band info to integer number, 2 -> present, 1 -> absent
    if isinstance(df['bands'][0], str):
        df['bands'] = df['bands'].apply(convert_to_binary)
    # Split tile numbers up to two different columns and delete the tile column
    df = split_tile_nums(df)
    return df


def shuffle_dataset(cutouts, catalog):
    """
    Shuffle the dataset by shuffling the cutouts and catalog in unison.

    Args:
        cutouts (ndarray): A 4D array of cutouts.
        catalog (DataFrame): A pandas DataFrame containing the metadata for each cutout.

    Returns:
        tuple: The shuffled cutouts and catalog.

    Raises:
        ValueError: If the number of cutouts does not match the number of catalog entries.
        RuntimeError: If an error occurs during shuffling.

    """
    if len(cutouts) != len(catalog):
        raise ValueError('The number of cutouts must match the number of catalog entries.')
    try:
        # Generate a random permutation
        idx = np.random.permutation(len(cutouts))
        cutouts[:] = cutouts[idx]
        # Shuffle catalog accordingly
        return cutouts, catalog.iloc[idx].reset_index(drop=True)
    except Exception as e:
        raise RuntimeError(f'Error during shuffling: {e}')
