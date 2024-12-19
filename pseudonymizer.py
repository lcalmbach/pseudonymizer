import streamlit as st
import json
import pandas as pd
import random
from faker import Faker
from pathlib import Path
from datetime import datetime, timedelta
import unicodedata
import re

fake = Faker("de_DE")
url_addresses = "https://data.bs.ch/api/explore/v2.1/catalog/datasets/100259/exports/csv?lang=de&timezone=Europe%2FBerlin&use_labels=false&delimiter=%3B&select=str_name,hausnr,hausnr_zus,plz,ort"
url_first_names = f"https://data.bs.ch/api/explore/v2.1/catalog/datasets/100129/exports/csv?lang=de&timezone=Europe%2FBerlin&use_labels=false&delimiter=%3B&select=vorname,geschlecht,anzahl&where=jahr={datetime.now().year-1}"
url_last_names = f"https://data.bs.ch/api/explore/v2.1/catalog/datasets/100127/exports/csv?lang=de&timezone=Europe%2FBerlin&use_labels=false&delimiter=%3B&select=nachname,anzahl&where=jahr={datetime.now().year-1}"
address_file = './data/100259.parquet'
first_name_file = './data/100129.parquet'
last_name_file = './data/100127.parquet'


def generate_json_template(file_name: str):
    """            address = 
    Generate a JSON template basemagicd on the columns of a DataFrame.
    Each column is given a default configuration for anonymization.

    :param df: pandas DataFrame
    :return: Dictionary template for JSON configuration
    """
    df = pd.read_excel(file_name)
    file_path = Path(file_name)
    file_base = file_path.stem  # example_document
    json_file_path = file_base + ".json"

    config = {}
    for column in df.columns:
        # Default configuration
        entry = {
            "pseudonymize": True,
            "faker_function": None,
            "faker_function_input_parameters": {},
        }

        with open(json_file_path, "w") as file:
            json.dump(config, file, indent=4)

    return config


class DataMasker:
    def __init__(self, file_path, config_path):
        self.file_path = file_path
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)
        self.data_in_df = pd.read_excel(file_path)
        self.data_out_df = self.data_in_df.copy()
        self.addresses = self.get_ogd_data(url= url_addresses, file=address_file)
        self.first_names = self.get_ogd_data(url=url_first_names, file=first_name_file)
        self.last_names = self.get_ogd_data(url=url_last_names, file=last_name_file)

    def get_ogd_data(self, file:str, url:str):
        """
        Get the addresses from a CSV file.

        :return: DataFrame with addresses
        """
        if Path(file).exists():
            df = pd.read_parquet(file)
        else:
            df = pd.read_csv(url, sep=";")
            if "vorname" in df.columns:
                expanded_df = pd.DataFrame(
                    df.apply(lambda row: [[row["vorname"], row["geschlecht"]]] * row["anzahl"], axis=1)
                    .explode().tolist(), columns=["vorname", "geschlecht"]
                )
                df = expanded_df
            elif "nachname" in df.columns:
                expanded_df = pd.DataFrame(
                    df.apply(lambda row: [row["nachname"]] * row["anzahl"], axis=1)
                    .explode().tolist(), columns=["nachname"]
                )
                df = expanded_df
            df.to_parquet(file)
        return df
        
    def generate_ahv_number(self):
        """
        Generates a random, valid Swiss AHV number.

        Returns:
            str: A 13-digit Swiss AHV number in the format '756.XXXX.XXXX.XX'.
        """
        # Country code for Switzerland
        country_code = "756"
        unique_identifier = f"{random.randint(0, 99999999):08d}"

        # Combine the first 11 digits
        base_number = f"{country_code}{unique_identifier}"

        # Calculate the checksum using Modulo 11
        def calculate_checksum(number):
            weights = [1, 3]  # Alternating weights
            total = 0
            for i, digit in enumerate(number):
                weight = weights[i % 2]
                total += int(digit) * weight
            remainder = total % 11
            checksum = (11 - remainder) if remainder != 0 else 0
            return checksum if checksum < 10 else 0  # Replace 10 with 0

        # Compute the checksum
        checksum = calculate_checksum(base_number)

        # Format the AHV number
        ahv_number = f"{country_code}.{unique_identifier[:4]}.{unique_identifier[4:]}.{checksum:02d}"
        return ahv_number

    def save_pseudonymized(self, file_path):
        """
        Save the pseudonymized DataFrame to a file.

        :param file_path: File path to save the pseudonymized DataFrame
        """
        self.df.to_excel(file_path, index=False)

    def delete_rows_with_missing_values(self):
        """
        Delete rows with missing values in the DataFrame.
        """
        for column, entry in self.config.items():
            if "not_null" in entry and entry["not_null"]:
                self.data_in_df = self.data_in_df[self.data_in_df[column].notnull()]
        self.data_out_df = self.data_in_df.copy()

    def pseudonymize(self):
        """
        Pseudonymize the columns in the DataFrame based on the configuration.

        :return: DataFrame with pseudonymized columns
        """
        st.markdown("### Input")
        st.write(self.data_in_df)
        self.delete_rows_with_missing_values()
        with st.expander("Columns", expanded=True):
            for column in self.data_in_df.columns:
                if column in self.config and self.config[column]["pseudonymize"]:
                    st.write(f"Pseudonymizing column: {column}")
                    self.pseudonymize_column(column)
        st.markdown("### Output")
        st.write(self.data_out_df)
        st.download_button(
            label="Download Pseudonymized Data",
            data=self.data_out_df.to_csv(index=False, sep=';'),
            file_name="pseudonymized_data.csv",
            mime="text/csv",
        )   

    def pseudonymize_column(self, column):
        """
        Pseudonymize a column in a DataFrame using a faker function.

        :param column: Column name to pseudonymize
        :return: DataFrame with pseudonymized column
        """
        entry = self.config[column]
        faker_function = entry["faker_function"]
        faker_parameters = entry.get("faker_function_input_parameters", {})
        if faker_function == "first_name":
            self.fake_first_names(column, faker_parameters)
        elif faker_function == "last_name":
            self.fake_last_names(column, faker_parameters)
        elif faker_function == "random_number":
            self.random_number(column, faker_parameters)
        elif faker_function == "date_add_random_days":
            self.date_add_random_days(column, faker_parameters)
        elif faker_function == "ahv_nr":
            self.ahv_nr(column, faker_parameters)
        elif faker_function == "random_address":
            self.random_address(column, faker_parameters)
        elif faker_function == "mobile":
            self.mobile(column, faker_parameters)
        elif faker_function == "email":
            self.email(column, faker_parameters)
        elif faker_function == "street":
            self.street(column, faker_parameters)
        elif faker_function == "house_number":
            self.house_number(column, faker_parameters)

    @staticmethod
    def normalize_name(name):
        """
        Normalize a name by replacing special characters with their ASCII equivalents
        and removing any other unwanted characters.

        :param name: The input name (str).
        :return: The normalized name (str).
        """
        # Replace accented characters with their ASCII equivalents
        normalized = (
            unicodedata.normalize("NFKD", name)
            .encode("ascii", "ignore")
            .decode("utf-8")
        )

        # Replace spaces or hyphens with dots
        normalized = re.sub(r"[ -]", ".", normalized)

        # Remove any remaining non-alphanumeric characters (excluding dots)
        normalized = re.sub(r"[^a-zA-Z0-9.]", "", normalized)

        return normalized.lower()

    def mobile(self, column, settings):
        """
        Add a random number of days to the dates in the specified column of the DataFrame.

        :param column: The column to add random days to.
        :param faker_parameters: A dictionary with configuration, including:
                            - "min_days": The minimum number of days to add.
                            - "max_days": The maximum number of days to add.
        """

        def fake_ch_mobile():
            providers = ["075", "076", "077", "078", "079", "079"]
            random_number = random.randint(1, 999)
            block1 = f"{random.randint(10, 99):03d}"
            random_number = random.randint(100, 9999)
            block2 = f"{random_number:04d}"
            return f"{random.choice(providers)}-{block1}-{block2}"

        unique_mobile_df = (
            self.data_in_df[[column]].drop_duplicates().reset_index(drop=True)
        )
        mobile_dict = {}

        for _, row in unique_mobile_df.iterrows():
            original_mobile = row[column]
            mobile_dict[original_mobile] = fake_ch_mobile()
        self.data_out_df[column] = self.data_in_df[column].map(
            lambda x: mobile_dict[x] if pd.notnull(x) else x
        )

    def email(self, column, settings):
        """
        Add a random number of days to the dates in the specified column of the DataFrame.

        :param column: The column to add random days to.
        :param faker_parameters: A dictionary with configuration, including:
                            - "min_days": The minimum number of days to add.
                            - "max_days": The maximum number of days to add.
        """

        def get_fake_email_from_name(row):
            """
            Generate a fake email address based on the specified name.

            :param name: The name to generate a fake email address for.
            :return: A fake email address based on the specified name.
            """
            # Normalize the name for consistency
            providers = (
                ["gmail.com"] * 3
                + ["yahoo.com"] * 2
                + ["gmx.com"] * 2
                + ["stud.edu.ch"] * 5
                + ["hotmail.com"]
                + ["outlook.com"]
            )
            provider = random.choice(providers)
            first_name = DataMasker.normalize_name(row[settings["first_name_col"]])
            last_name = DataMasker.normalize_name(row[settings["last_name_col"]])
            return f"{first_name}.{last_name}@{provider}"

        if "first_name_col" in settings:
            fname = settings["first_name_col"]
            lname = settings["last_name_col"]
            self.data_out_df[column] = self.data_out_df.apply(
                lambda row: (
                    get_fake_email_from_name(row)
                    if row[fname] and row[lname]
                    else row["email"]
                ),
                axis=1,
            )
        else:
            self.data_out_df[column] = self.data_in_df[column].map(
                lambda x: fake.email()
            )

    def get_fake_address(self, location_dict):
        """
        Get a fake address based on the specified location.

        :param ort: The location to generate a fake address for.
        :return: A fake address based on the specified location.
        """
        # limit choice either by ort or plz, if not available, use fakder.fake address
        if not(location_dict["location_value"] in self.addresses[location_dict["location_code_col"]].values):
            return fake.street_address()
        else:                                                        
            df = self.addresses[
                self.addresses[location_dict["location_code_col"]]
                == location_dict["location_value"]
            ]
            address = df.sample(1).iloc[0]
            street_housenr = f"{address['str_name']} {int(address['hausnr'])}"
            if pd.notnull(address["hausnr_zus"]) and address["hausnr_zus"] != "":
                street_housenr += f"{address['hausnr_zus']}"
            return street_housenr

    def get_fake_street(self, location_dict):
        """
        Get a fake address based on the specified location.

        :param ort: The location to generate a fake address for.
        :return: A fake address based on the specified location.
        """
        if not(location_dict["location_value"] in self.addresses[location_dict["location_code_col"]].values):
            return fake.street_name()
        else:
            df = self.addresses[
                self.addresses[location_dict["location_code_col"]]
                == location_dict["location_value"]
            ]
            streets = df.sample(1).iloc[0]
            return streets["str_name"]

    def street(self, column, settings):
        """
        replaces street names by random street names from the same postal code

        :param column: The column to add random days to.
        :param faker_parameters: A dictionary with configuration, including:
                            - "min_days": The minimum number of days to add.
                            - "max_days": The maximum number of days to add.
        """

        # Determine the range of days to add
        unique_street_fields = [column, settings["location_data_col"]]
        unique_street_fields = (
            self.data_in_df[unique_street_fields]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        # Initialize name dictionary for consistent mapping
        street_dict = {}

        # Generate aliases for unique names based on gender
        for _, row in unique_street_fields.iterrows():
            original_street = row[column]
            street_dict[original_street] = self.get_fake_street(
                {
                    "location_code_col": settings["location_code_col"],
                    "location_value": row[settings["location_data_col"]],
                }
            )
        self.data_out_df[column] = self.data_in_df[column].map(
            lambda x: street_dict[x] if pd.notnull(x) else x
        )

    def house_number(self, column, settings):
        """
        Gets a different random house number from the same street
        :param column: The column to add random days to.
        :param faker_parameters: A dictionary with configuration, including:
                            - "min_days": The minimum number of days to add.
                            - "max_days": The maximum number of days to add.
        """

        def get_random_housenumber(has_suffix):
            house_number = random.randint(1, 100)
            if has_suffix:
                suffix = random.choice(["a", "b", "c", "d", "e", "f"])
                return f"{house_number}{suffix}"
            return house_number

        # Generate aliases for unique names based on gender
        for index in self.data_out_df[self.data_out_df[column].notna()].index:
            with_suffix = random.random() < settings["frequency_suffix"]
            self.data_out_df.loc[index, column] = get_random_housenumber(with_suffix)

    def split_street_housenumber(self, address):
        """
        Split the street and house number from the combined string.

        :param street_housenumber: The combined street and house number.
        :return: The street and house number as separate strings.
        """
        address = address.strip()
    
        # Use a regular expression to split street and house number
        match = re.match(r"^(.*)\s+(\d+)$", address)
        
        if match:
            street = match.group(1).strip()  # The part before the last number
            house_number = match.group(2).strip()  # The numeric house number
            return (street, house_number)
        else:
            return (address, None)
    
    def change_house_number(self, address, location_dict):
        """
        Change the house number in the address to a random number from the same street. if there are no other house numbers 
        in the same street, pick a random address from the same location.

        :param address: The original address.
        :param location_dict: A dictionary with configuration, including:
                            - "location_code_col": The column containing the postal code.
                            - "location_value": The postal code of the address.
        :return: The address with a random house number from the same street.
        """
        # Split the street and house number
        street, house_number = self.split_street_housenumber(address)
        if house_number is None:
            return address
        else:
            df = self.addresses[(self.addresses['str_name'] == street) & (self.addresses[location_dict['location_code_col']] == location_dict['location_value'])]
            if len(df) > 1:
                random_address_in_street = df.sample(1).iloc[0]
                house_number = random_address_in_street['hausnr'] if random_address_in_street['hausnr_zus'] is None else random_address_in_street['hausnr'] + random_address_in_street['hausnr_zus']
                return f"{street} {house_number}"
            else:
                address = self.get_fake_address(location_dict)
                return address
                
    def blur_address(self, column, settings):
        """
        Add a random number of days to the dates in the specified column of the DataFrame.

        :param column: The column to add random days to.
        :param faker_parameters: A dictionary with configuration, including:
                            - "min_days": The minimum number of days to add.
                            - "max_days": The maximum number of days to add.
        """

        # Determine the range of days to add
        unique_address_fields = settings["unique_address_fields"]
        unique_address_df = (
            self.data_in_df[unique_address_fields]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        # Initialize name dictionary for consistent mapping
        address_dict = {}

        # Generate aliases for unique names based on gender
        for _, row in unique_address_df.iterrows():
            original_address = row[column]
            address_dict[original_address] = self.change_house_number(
                original_address,
                {
                    "location_code_col": settings["location_code_col"],
                    "location_value": row[settings["location_data_col"]],
                }
            )
        self.data_out_df[column] = self.data_in_df[column].map(
            lambda x: address_dict[x] if pd.notnull(x) else x
        )

    def random_address(self, column, settings):
        """
        Add a random number of days to the dates in the specified column of the DataFrame.

        :param column: The column to add random days to.
        :param faker_parameters: A dictionary with configuration, including:
                            - "min_days": The minimum number of days to add.
                            - "max_days": The maximum number of days to add.
        """

        # Determine the range of days to add
        unique_address_fields = settings["unique_address_fields"]
        unique_address_df = (
            self.data_in_df[unique_address_fields]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        # Initialize name dictionary for consistent mapping
        address_dict = {}

        # Generate aliases for unique names based on gender
        for _, row in unique_address_df.iterrows():
            original_address = row[column]
            address_dict[original_address] = self.get_fake_address(
                {
                    "location_code_col": settings["location_code_col"],
                    "location_value": row[settings["location_data_col"]],
                }
            )
        self.data_out_df[column] = self.data_in_df[column].map(
            lambda x: address_dict[x] if pd.notnull(x) else x
        )

    def ahv_nr(self, column, settings):
        """
        Add a random number of days to the dates in the specified column of the DataFrame.

        :param column: The column to add random days to.
        :param faker_parameters: A dictionary with configuration, including:
                            - "min_days": The minimum number of days to add.
                            - "max_days": The maximum number of days to add.
        """
        # Determine the range of days to add

        unique_ahvnr_df = (
            self.data_in_df[[column]].drop_duplicates().reset_index(drop=True)
        )
        # Initialize name dictionary for consistent mapping
        ahvnr_dict = {}

        # Generate aliases for unique names based on gender
        for _, row in unique_ahvnr_df.iterrows():
            original_ahvnr = row[column]
            ahvnr_dict[original_ahvnr] = self.generate_ahv_number()
        self.data_out_df[column] = self.data_in_df[column].map(
            lambda x: ahvnr_dict[x] if pd.notnull(x) else x
        )

    def date_add_random_days(self, column, settings):
        """
        Add a random number of days to the dates in the specified column of the DataFrame.

        :param column: The column to add random days to.
        :param faker_parameters: A dictionary with configuration, including:
                            - "min_days": The minimum number of days to add.
                            - "max_days": The maximum number of days to add.
        """
        # Determine the range of days to add
        min_days = settings["min_value"]
        max_days = settings["max_value"]

        # Generate a list of random days to add
        random_days = [
            random.randint(min_days, max_days) for _ in range(len(self.data_in_df))
        ]

        unique_dates_df = (
            self.data_in_df[[column]].drop_duplicates().reset_index(drop=True)
        )
        # Initialize name dictionary for consistent mapping
        date_dict = {}

        # Generate aliases for unique names based on gender
        for _, row in unique_dates_df.iterrows():
            original_date = row[column]
            if isinstance(original_date, str):
                original_date = datetime.strptime(original_date, "%Y-%m-%dT%H:%M:%S")
                random_days = random.randint(min_days, max_days)
                # Add the random days to the base date
                date_dict[original_date] = original_date + timedelta(days=random_days)
            else:
                date_dict[original_date] = original_date
        # Apply the generated aliases back to the DataFrame
        self.data_out_df[column] = self.data_in_df[column].map(date_dict)

    def random_number(self, column, faker_parameters):
        """
        Generate distinct random numbers for each row in the specified column of the DataFrame.

        :param column: The column to populate with random numbers.
        :param faker_parameters: A dictionary with configuration, including:
                            - "min_value": The minimum value for the random number.
                            - "max_value": The maximum value for the random number.
        """
        # Determine the range and ensure it's large enough for distinct numbers
        min_value = faker_parameters["min_value"]
        max_value = faker_parameters["max_value"]
        row_count = len(self.data_in_df)

        if (max_value - min_value + 1) < row_count:
            raise ValueError(
                "Range is too small to generate unique random numbers for all rows."
            )

        # Generate a list of unique random numbers
        unique_numbers = random.sample(range(min_value, max_value + 1), row_count)

        # Assign the unique numbers to the specified column
        self.data_out_df[column] = unique_numbers

    

    def fake_first_names(self, column, settings: dict):
        """
        Generate fake first names based on gender for a specified column in the DataFrame.

        :param column: The column containing the original names.
        :param settings: A dictionary with configuration, including:
                        - "gender_col": The column indicating gender.
                        - "female": The value representing female in the gender column.
        """
        def get_fake_first_name(row, settings: dict):
            if "source" in settings and settings["source"] == "bs":
                if settings["use_gender_col"]:
                    # map data source gender code to internal m/w gender code
                    st.write(row)
                    gender = "w" if str(row[settings['gender_col']]) == row[str(settings['female'])] else "m"
                    return self.first_names[
                        self.first_names["geschlecht"] == gender].sample(1).iloc[0]["vorname"]
                else:
                    return self.first_names.sample(1).iloc[0]["vorname"]
            else:
                if settings["use_gender_col"]:
                    return fake.first_name_female() if row[settings["gender_col"]] == "w" else fake.first_name_male()
                else:
                    return fake.first_name()
        
        if settings["use_gender_col"]:
            unique_names_df = (
                self.data_in_df[[column, settings["gender_col"]]]
                .drop_duplicates()
                .reset_index(drop=True)
            )
        else:
            unique_names_df = (
                self.data_in_df[[column]].drop_duplicates().reset_index(drop=True)
            )
        name_dict = {}

        for _, row in unique_names_df.iterrows():
            original_name = row[column]
            name_dict[original_name] = get_fake_first_name(row, settings)

        self.data_out_df[column] = self.data_in_df[column].map(name_dict)

    def fake_last_names(self, column, settings: dict):
        """
        Generate fake first names based on gender for a specified column in the DataFrame.

        :param column: The column containing the original names.
        :param settings: A dictionary with configuration, including:
                        - "gender_col": The column indicating gender.
                        - "female": The value representing female in the gender column.
        """
        def get_fake_last_name(row, settings):
            if "source" in settings and settings["source"] == "bs":
                return self.last_names.sample(1).iloc[0]["nachname"]
            else:
                return fake.last_name()

        # Extract unique name-gender combinations
        unique_names_df = (
            self.data_in_df[[column]].drop_duplicates().reset_index(drop=True)
        )
        # Initialize name dictionary for consistent mapping
        name_dict = {}

        # Generate aliases for unique names based on gender
        for _, row in unique_names_df.iterrows():
            original_name = row[column]
            name_dict[original_name] = get_fake_last_name(row, settings)


        # Apply the generated aliases back to the DataFrame
        self.data_out_df[column] = self.data_in_df[column].map(name_dict)

    def save_json_template(json_template, file_path):
        """
        Save the JSON template to a file.

        :param json_template: Dictionary template for JSON configuration
        :param file_path: File path to save the JSON template
        """
        with open(file_path, "w") as file:
            json.dump(json_template, file, indent=4)
