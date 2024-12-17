# Pseudonymizer App

A user-friendly **Streamlit app** that enables you to pseudonymize sensitive data in **CSV** or **XLSX** files based on a customizable configuration file.

## Installation

## Features
- Upload **CSV** or **XLS/XLSX** files containing sensitive data.
- Provide a configuration file to define pseudonymization rules for each column.
- Pseudonymize the data interactively with options for review.
- Download the pseudonymized file for further use.

---

## How It Works

1. **Upload the Data File**:
   - Choose a **CSV** or **XLS/XLSX** file containing the data you want to pseudonymize.

2. **Upload the Configuration File**:
   - Provide a JSON configuration file that specifies:
     - Columns to pseudonymize.
     - The pseudonymization rules for each column (e.g., fake names, email addresses).

3. **Pseudonymization Process**:
   - The app applies the rules defined in the configuration file to the data.
   - Review and verify the pseudonymized output in the app.

4. **Download the Result**:
   - Download the pseudonymized file for secure use or further processing.

---

## Configuration File Format

The configuration file is a JSON file where each column in the dataset that requires pseudonymization is defined. Below is an example:

```json
{
    "sutdent_id": {
        "pseudonymize": true,
        "not_null": true,
        "faker_function": "random_number",
        "faker_function_input_parameters": {"min_value": 200000, "max_value": 700000, "unique": true}
    },
    "student_name": {
        "pseudonymize": true,
        "faker_function": "last_name",
        "faker_function_input_parameters": {}
    },
    "student_gender": {
        "pseudonymize": false,
        "faker_function": null
    },
    "addresse": {
        "pseudonymize": true,
        "faker_function": "address",
        "faker_function_input_parameters": {"unique_address_fields": ["adress", "postal_code"], "location_code_col": "plz", "location_data_col": "postal_code"}
    },
    ...
}
```

## Pseudonymizer functions

The following functions have been impolemented and can be used as the keyword faker_function in the config file

| function | description | parameters |
| -------- | ----------- | ---------- |
| random_number | Replaces each occurrence of a value in this column by a random number between min and max, passed as parameters | {"min_value": 200000, "max_value": 700000, "unique": true} |
| random_address | Replaces each occurrence of an address (street street number) by a random address of the same location | {"unique_address_fields": ["addresse", "postal_code"], "location_code_col": "plz", "location_data_col": "postal_code"} |
| blur_address | Replaces each occurrence of an address (street street number) by a random address from the same street by switching the house number. This garantees that persons or objects stay geographically close| {"unique_address_fields": ["addresse", "postal_code"], "location_code_col": "plz", "location_data_col": "postal_code"} |
| first_name | Replaces each occurrence of a first name by a random first name using the library faker. if a column holds the gender value, you can have male and female names generated based on the gender value. | {"use_gender_col": true, "gender_col": "student_gender", "female":"2", "male":"1"} |
| last_name | Replaces each occurrence of a last name by a random last name using the library faker. If a column holds the gender value, you can have male and female names generated based on the gender value. | {"use_gender_col": true, "gender_col": "student_gender", "female":"2", "male":"1"} |

## Opendata
The addresses and street names are generated using a dataset from [data.bs](https://data.bs.ch/explore/dataset/100259) in order to generate familiar addresses. generating addresses requires a location field, which is either the location name or a postal code. If no postal code from Basel-Stadt Switzerland is provided, the faker.street() function is used to generate the street names.
