import os
import re
from collections import namedtuple
from datetime import datetime
from enum import Enum
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype
from paramiko import RSAKey, SFTPClient, Transport

DEFAULT_SFTP_PORT = 22
FILENAME_DATETIME_FORMAT = "%Y%m%d-%H%M"
VALUE_DATETIME_FORMAT = "%Y-%m-%d %H:%M"
FILENAME_PREFIX = "qrt_academy"
GROUP_ID_PATTERN = r"[A-Z0-9_]{2,25}"
TIME_PATTERN = r"\d{8}-\d{4}"
FILENAME_VALIDATION_PATTERN = f"^{FILENAME_PREFIX}_{GROUP_ID_PATTERN}_{TIME_PATTERN}.csv$"
FILENAME_PATTERN = "{prefix}_{group_id}_{timestamp}.csv"

ColDef = namedtuple("ColDef", "name type constraint")

TARGETS_SCHEMA = [
    ColDef("id_specific", str, 25),
    ColDef("extra_key", str, 20),
    ColDef("value_ts", datetime, None),
    ColDef("strategy", str, 100),
    ColDef("internal_code", str, 64),
    ColDef("ric", str, 20),
    ColDef("ticker", str, 25),
    ColDef("target_notional", float, None),
    ColDef("currency", str, 3),
    ColDef("target_contracts", int, None),
    ColDef("ref_price", float, 25),
    ColDef("advisor_name", str, 25),
]
REQUIRED_COLS = [
    "internal_code",
    "target_notional",
    "currency",
]

class Region(Enum):
    AMER = "AMER"
    EMEA = "EMEA"

def prepare_targets_file(targets: pd.DataFrame, group_id: str, region: Region, output_dir: Path) -> Path:
    if targets.empty:
        raise ValueError("Empty targets dataframe")

    missing_cols = set(REQUIRED_COLS).difference(targets.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns {missing_cols}")

    if group_id is None or re.fullmatch(GROUP_ID_PATTERN, group_id) is None:
        raise ValueError(f"Group Id should be maximum 25 characters and contain only capital letters and numbers: {group_id}")

    if region is None:
        raise ValueError("Region must be one of 'AMER' or 'EMEA'")

    if isinstance(region, str):
        region = Region(region.upper())

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    output_dir = output_dir.expanduser()

    if not (output_dir.exists() and output_dir.is_dir()):
        raise ValueError(f"Invalid output directory {output_dir}")

    targets = targets.copy()

    # autofill columns required by QRT
    targets["id_specific"] = group_id
    targets["strategy"] = f"{group_id}_{region.value}"
    targets["advisor_name"] = group_id
    timestamp = pd.Timestamp("now")
    targets["value_ts"] = timestamp
    targets["extra_key"] = targets["id_specific"] + "_" + targets["internal_code"]
    column_order = [c.name for c in TARGETS_SCHEMA]
    targets = targets.reindex(columns=column_order)

    targets["ric"] = targets.ric.fillna(targets.internal_code)
    targets["ticker"] = targets.ticker.fillna(targets.internal_code)
    targets["target_contracts"] = targets.target_contracts.fillna(0).round(0).astype(int)
    targets["ref_price"] = targets.ref_price.fillna(0.0)

    filename = FILENAME_PATTERN.format(
        prefix=FILENAME_PREFIX,
        group_id=group_id,
        timestamp=timestamp.strftime(FILENAME_DATETIME_FORMAT),
    )
    filepath = output_dir / filename
    targets.to_csv(filepath, date_format=VALUE_DATETIME_FORMAT, index=False)

    return filepath

def validate_targets_file(targets_csv_path: Path) -> list[str]:
    # (Official QRT Validation Logic)
    if targets_csv_path is None:
        raise ValueError("Targets csv path cannot be None")
    elif isinstance(targets_csv_path, str):
        targets_csv_path = Path(targets_csv_path)

    targets_csv_path = targets_csv_path.expanduser()
    if not targets_csv_path.exists():
        raise ValueError("Invalid targets file path")

    targets = pd.read_csv(targets_csv_path)
    errors = []

    if targets.empty:
        raise ValueError("Empty targets csv")

    if not re.match(FILENAME_VALIDATION_PATTERN, targets_csv_path.name):
        errors.append("Target filename should match the format: 'qrt_academy_XX_20241002-1120.csv'")

    schema_cols = [c.name for c in TARGETS_SCHEMA]
    missing_cols = set(schema_cols).difference(targets.columns)
    if missing_cols:
        errors.append(f"Missing required columns {missing_cols}")

    extra_cols = targets.columns.difference(schema_cols)
    if len(extra_cols):
        errors.append(f"Unsupported columns: {extra_cols}")

    if targets.columns.to_list() != schema_cols:
        errors.append(f"Column order should match: {schema_cols}")

    if not targets.extra_key.is_unique:
        errors.append("Values in the column extra_key must be unique")

    for idx, row in targets.iterrows():
        for col in TARGETS_SCHEMA:
            if err := _check_value(row, col):
                errors.append(f"(row: {idx+1}, column: {col.name}): {err}")

    print(f"Found {len(errors)} error(s) while validating {targets_csv_path}")
    return errors

def _check_value(row: pd.Series, col: ColDef) -> str | None:
    err = None
    value = row.get(col.name)
    if pd.isna(value):
        err = "null value"
    elif col.type is str:
        if not isinstance(value, str):
            err = "expecting a string value"
        elif value == "":
            err = "empty string"
        elif len(value) > col.constraint:
            err = f"value should be under {col.constraint} char"
    elif col.type is float:
        if not is_numeric_dtype(type(value)):
            err = "expecting a decimal value"
    elif col.type is int:
        if not isinstance(value, int):
            err = "expecting an integer value"
    elif col.type is datetime:
        try:
            datetime.strptime(value, VALUE_DATETIME_FORMAT)
        except ValueError:
            err = f"date format does not match '{VALUE_DATETIME_FORMAT}'"
    return err

def upload_targets_file(targets_csv_path: Path, region: Region, sftp_username: str, private_key_path: Path, sftp_host: str, sftp_port: str = DEFAULT_SFTP_PORT):
    if targets_csv_path is None: raise ValueError("Targets csv path cannot be None")
    elif isinstance(targets_csv_path, str): targets_csv_path = Path(targets_csv_path)
    targets_csv_path = targets_csv_path.expanduser()

    if private_key_path is None: raise ValueError("Private key path cannot be None")
    elif isinstance(private_key_path, str): private_key_path = Path(private_key_path)
    private_key_path = private_key_path.expanduser()

    validation_errors = validate_targets_file(targets_csv_path)
    if validation_errors:
        raise ExceptionGroup(f"{len(validation_errors)} error(s) found while validating the targets csv", [ValueError(err) for err in validation_errors])

    remote_file_path = Path("incoming") / region.value.lower() / targets_csv_path.name

    try:
        print(f"Reading private key from: {private_key_path}")
        private_key = RSAKey.from_private_key_file(private_key_path)

        print(f"Connecting to {sftp_host}:{sftp_port}")
        with Transport((sftp_host, sftp_port)) as transport:
            print(f"Logging in as {sftp_username}")
            transport.connect(username=sftp_username, pkey=private_key)

            with SFTPClient.from_transport(transport) as sftp:
                print(f"Uploading {targets_csv_path} to {remote_file_path.as_posix()}")
                sftp.put(targets_csv_path, remote_file_path.as_posix(), confirm=False)

    except Exception as e:
        raise RuntimeError(f"Error while uploading the file {targets_csv_path} to the SFTP account: {sftp_host}:{sftp_port}/{remote_file_path.as_posix()}") from e

    print(f"SUCCESS: File '{targets_csv_path}' officially uploaded to {region.value}.")

# =====================================================================
# GITHUB ACTIONS AUTOMATION BLOCK
# =====================================================================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TARGETS_DIR = Path(BASE_DIR) / "targets"
    raw_target_file = TARGETS_DIR / "targets.csv"
    
    if not raw_target_file.exists():
        print("ERROR: Raw targets.csv not found! run_strategy.py likely failed.")
        exit(1)
        
    # 1. Load the raw 3-column output from your strategy
    print("Loading raw strategy targets...")
    raw_targets = pd.read_csv(raw_target_file)
    
    # Verify Group ID based on SSH key naming
    GROUP_ID = "IND03" 
    
    # 2. Re-format the file to meet QRT's exact 12-column strict schema
    print(f"Formatting targets using official QRT schema for Group {GROUP_ID}...")
    formatted_file_path = prepare_targets_file(
        targets=raw_targets,
        group_id=GROUP_ID,
        region=Region.AMER,
        output_dir=TARGETS_DIR
    )
    
    # 3. Retrieve secure GitHub credentials
    USERNAME = os.environ.get("SFTP_USERNAME")
    KEY_PATH = Path("private_key.pem")
    
    if not USERNAME:
        print("ERROR: SFTP_USERNAME environment variable is missing.")
        exit(1)
        
    # 4. Validate & Upload
    print("Initiating Official QRT Upload Sequence...")
    upload_targets_file(
        targets_csv_path=formatted_file_path,
        region=Region.AMER,
        sftp_username=USERNAME,
        private_key_path=KEY_PATH,
        sftp_host="sftp.qrt.cloud"
    )
