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
    "name": {
        "pseudonymize": true,
        "method": "faker",
        "faker_function": "name",
        "consistent": true
    },
    "email": {
        "pseudonymize": true,
        "method": "faker",
        "faker_function": "email",
        "consistent": true
    },
    "address": {
        "pseudonymize": false
    }
}
