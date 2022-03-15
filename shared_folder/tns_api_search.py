#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Avgust 2021

Developed and tested on:

- Linux 20.04 LTS
- Windows 10
- Python 3.8 (Spyder 4)

@author: Nikola Knezevic ASTRO DATA
"""

import os
import requests
import json
from collections import OrderedDict
import time

from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

# ----------------------------------------------------------------------------------

TNS = "www.wis-tns.org"
url_tns_api = "https://" + TNS + "/api/get"

TNS_BOT_ID = "YOUR_BOT_ID_HERE"
TNS_BOT_NAME = "YOUR_BOT_NAME_HERE"
TNS_API_KEY = "YOUR_BOT_API_KEY_HERE"

# list that represents json file for search obj
search_obj = [
    ("ra", ""),
    ("dec", ""),
    ("radius", ""),
    ("units", ""),
    ("objname", ""),
    ("objname_exact_match", 0),
    ("internal_name", ""),
    ("internal_name_exact_match", 0),
    ("objid", ""),
    ("public_timestamp", ""),
]

# list that represents json file for get obj
get_obj = [
    ("objname", ""),
    ("objid", ""),
    ("photometry", "0"),
    ("spectra", "1"),
]

# file url (for downloading file from TNS)
file_tns_url = "Here put url of a file you want to download from TNS."

# current working directory
cwd = os.getcwd()
# directory for downloaded files
download_dir = os.path.join(cwd, "downloaded_files")

# external http errors
ext_http_errors = [403, 500, 503]
err_msg = [
    "Forbidden",
    "Internal Server Error: Something is broken",
    "Service Unavailable",
]

# ----------------------------------------------------------------------------------


def set_bot_tns_marker():
    tns_marker = (
        'tns_marker{"tns_id": "'
        + str(TNS_BOT_ID)
        + '", "type": "bot", "name": "'
        + TNS_BOT_NAME
        + '"}'
    )
    return tns_marker


def format_to_json(source):
    parsed = json.loads(source, object_pairs_hook=OrderedDict)
    result = json.dumps(parsed, indent=4)
    return result


def is_string_json(string):
    try:
        json_object = json.loads(string)
    except Exception:
        return False
    return json_object


def print_status_code(response):
    json_string = is_string_json(response.text)
    if not json_string:
        print(
            "status code ---> [ "
            + str(json_string["id_code"])
            + " - '"
            + str(json_string["id_message"])
            + "' ]\n"
        )
    else:
        status_code = response.status_code
        if status_code == 200:
            status_msg = "OK"
        elif status_code in ext_http_errors:
            status_msg = err_msg[ext_http_errors.index(status_code)]
        else:
            status_msg = "Undocumented error"
        print(
            "status code ---> [ "
            + str(status_code)
            + " - '"
            + status_msg
            + "' ]\n"
        )


def search():
    search_url = url_tns_api + "/search"
    tns_marker = set_bot_tns_marker()
    headers = {"User-Agent": tns_marker}
    json_file = OrderedDict(search_obj)
    search_data = {"api_key": TNS_API_KEY, "data": json.dumps(json_file)}
    response = requests.post(search_url, headers=headers, data=search_data)
    return response


def get():
    get_url = url_tns_api + "/object"
    tns_marker = set_bot_tns_marker()
    headers = {"User-Agent": tns_marker}
    json_file = OrderedDict(get_obj)
    get_data = {"api_key": TNS_API_KEY, "data": json.dumps(json_file)}
    response = requests.post(get_url, headers=headers, data=get_data)
    return response


def get_file():
    filename = os.path.basename(file_tns_url)
    tns_marker = set_bot_tns_marker()
    headers = {"User-Agent": tns_marker}
    api_data = {"api_key": TNS_API_KEY}
    print("Downloading file '" + filename + "' from the TNS...\n")
    response = requests.post(
        file_tns_url, headers=headers, data=api_data, stream=True
    )
    print_status_code(response)
    path = os.path.join(download_dir, filename)
    if response.status_code == 200:
        with open(path, "wb") as f:
            for chunk in response:
                f.write(chunk)
        print("File was successfully downloaded.\n")
    else:
        print("File was not downloaded.\n")


def print_response(response, json_file, counter):
    response_code = (
        str(response.status_code)
        if not json_file
        else str(json_file["id_code"])
    )
    stats = (
        "Test #"
        + str(counter)
        + "| return code: "
        + response_code
        + " | Total Rate-Limit: "
        + str(response.headers.get("x-rate-limit-limit"))
        + " | Remaining: "
        + str(response.headers.get("x-rate-limit-remaining"))
        + " | Reset: "
        + str(response.headers.get("x-rate-limit-reset"))
    )
    if response.headers.get("x-cone-rate-limit-limit") is not None:
        stats += (
            " || Cone Rate-Limit: "
            + str(response.headers.get("x-cone-rate-limit-limit"))
            + " | Cone Remaining: "
            + str(response.headers.get("x-cone-rate-limit-remaining"))
            + " | Cone Reset: "
            + str(response.headers.get("x-cone-rate-limit-reset"))
        )
    print(stats)


def get_reset_time(response):
    # If any of the '...-remaining' values is zero, return the reset time
    for name in response.headers:
        value = response.headers.get(name)
        if name.endswith("-remaining") and value == "0":
            return int(
                response.headers.get(name.replace("remaining", "reset"))
            )
    return None


def rate_limit_handling():
    counter = 0
    while True:
        counter = counter + 1
        response = search()
        json_file = is_string_json(response.text)
        print_response(response, json_file, counter)
        # Checking if rate-limit reached (...-remaining = 0)
        reset = get_reset_time(response)
        # A general verification if not some error
        if response.status_code == 200:
            if reset is not None:
                # Sleeping for reset + 1 sec
                print("Sleep for " + str(60) + " sec")
                time.sleep(60)
                # Can continue to submit requests...
                print("Continue to submit requests...")
                for i in range(3):
                    counter = counter + 1
                    response = search()
                    json_file = is_string_json(response.text)
                    print_response(response, json_file, counter)
                print("etc...\n")
                break
        else:
            print_status_code(response)
            break


# ----------------------------------------------------------------------------------

TNS_BOT_ID = 125088
TNS_BOT_NAME = "supernova_dtd_bot"
TNS_API_KEY = "2f6450eadd2253bc3be8c179707923d1e28f83e1"

data = fits.open("manga_firefly-v2_4_3-GLOBALPROP.fits")


ra = data[1].data["OBJRA"]
dec = data[1].data["OBJDEC"]

results = []

counter = 0
for r, d in zip(ra, dec):
    while True:
        c = SkyCoord(ra=r * u.degree, dec=d * u.degree, frame="icrs")
        r_hex, d_hex = c.to_string("hmsdms").split(" ")
        r_hex = r_hex.replace("h", ":")
        r_hex = r_hex.replace("m", ":")
        r_hex = r_hex.replace("s", "")
        d_hex = d_hex.replace("d", ":")
        d_hex = d_hex.replace("m", ":")
        d_hex = d_hex.replace("s", "")
        # EXAMPLE 1 (search obj)
        search_obj = [
            ("ra", r_hex),
            ("dec", d_hex),
            ("radius", "32"),
            ("units", "arcsec"),
            ("objname", ""),
            ("objname_exact_match", 0),
            ("internal_name", ""),
            ("internal_name_exact_match", 0),
            ("objid", ""),
            ("public_timestamp", ""),
        ]
        print(search_obj)
        counter = counter + 1
        response = search()
        json_file = is_string_json(response.text)
        print_response(response, json_file, counter)
        reset = get_reset_time(response)
        # A general verification if not some error
        if response.status_code == 200:
            if reset is not None:
                # Sleeping for reset + 1 sec
                print("Sleep for " + str(reset + 1) + " sec")
                time.sleep(reset + 1)
                # Can continue to submit requests...
                print("Continue to submit requests...")
                for i in range(3):
                    counter = counter + 1
                    response = search()
                    json_file = is_string_json(response.text)
                    print_response(response, json_file, counter)
            json_data = json.loads(response.text)
            results.append(json_data)
            if json_data["data"]["reply"] != []:
                print(json_data["data"]["reply"])
            break
        elif response.status_code == 429:
            if reset is not None:
                # Sleeping for reset + 1 sec
                print("Sleep for " + str(reset + 1) + " sec")
                time.sleep(reset + 1)
        else:
            print_status_code(response)

np.save("manga_firefly-v2_4_3_tns_matched", results)

get_obj_results = []
for json_data in results:
    counter = 0
    if json_data["data"]["reply"] != []:
        while True:
            get_obj = [
                ("objname", json_data["data"]["reply"][0]["objname"]),
            ]
            get_obj_response = get()
            reset = get_reset_time(get_obj_response)
            # A general verification if not some error
            if get_obj_response.status_code == 200:
                if reset is not None:
                    # Sleeping for reset + 1 sec
                    print("Sleep for " + str(reset + 1) + " sec")
                    time.sleep(reset + 1)
                    # Can continue to submit requests...
                    print("Continue to submit requests...")
                    for i in range(3):
                        counter = counter + 1
                        get_obj_response = get()
                        json_file = is_string_json(get_obj_response.text)
                        print_response(get_obj_response, json_file, counter)
                get_obj_json_data = json.loads(get_obj_response.text)
                print(get_obj_json_data)
                get_obj_results.append(get_obj_json_data)
                break
            elif get_obj_response.status_code == 429:
                if reset is not None:
                    # Sleeping for reset + 1 sec
                    print("Sleep for " + str(reset + 1) + " sec")
                    time.sleep(reset + 1)
            else:
                print_status_code(get_obj_response)


np.save("manga_firefly-v2_4_3_tns_matched_full_details", get_obj_results)


# # EXAMPLE 2 (get obj)
# get_obj = [
#     ("objname", "2017A"),
#     ("objid", ""),
#     ("photometry", "1"),
#     ("spectra", "1"),
# ]
# response = get()
# json_data = format_to_json(response.text)
# print(json_data)


# # EXAMPLE 3 (get file from TNS)
# file_tns_url = (
#     "https://" + TNS + "/system/files/uploaded/"
#     "Padova-Asiago/tns_2017A_2457777.69_Ekar_AFOSC_Padova-Asiago.txt"
# )
# get_file()


# # EXAMPLE 4 (test rate-limit search)
# search_obj = [
#     ("ra", ""),
#     ("dec", ""),
#     ("radius", ""),
#     ("units", ""),
#     ("objname", "2021rak"),
#     ("objname_exact_match", 0),
#     ("internal_name", ""),
#     ("internal_name_exact_match", 0),
#     ("objid", ""),
#     ("public_timestamp", ""),
# ]
# rate_limit_handling()


# # EXAMPLE 5 (test rate-limit cone search)
# search_obj = [
#     ("ra", "15:57:28"),
#     ("dec", "+30:03:39"),
#     ("radius", "5"),
#     ("units", "arcsec"),
#     ("objname", ""),
#     ("objname_exact_match", 0),
#     ("internal_name", ""),
#     ("internal_name_exact_match", 0),
#     ("objid", ""),
#     ("public_timestamp", ""),
# ]
# rate_limit_handling()


# ----------------------------------------------------------------------------------
