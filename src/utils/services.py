import json
import requests
from utils import config
import time
import os
os.environ['NO_PROXY'] = '127.0.0.1,localhost'


def post(data, target_path, times=1):
    """
    Used to insert or update objects
    :param data: Data Type must be dictionary type
    :param target_path: Path of service, define at config TARGET_PATH
    :param times: Retries numbers
    :return:
    """
    url = "{}/{}".format(config.API_URL, target_path)
    # headers = Auth().generate_header_token()

    for i in range(times):
        try:
            response = requests.post(url, data=json.dumps(data),
                                     # headers=headers
                                     )

            response_content = json.loads(response.content, encoding="utf-8")

            if response_content.get('code') != 200:
                time.sleep(3)
                continue
            break
        except Exception as e:
            raise Exception("PATH {}. No connection could be made because "
                            "the target machine actively refused it. {}. Data {}".format(url, e, data))

    if response_content.get('code') != 200:
        raise Exception("ERROR : {}, URL PATH : {}, DATA : {}".format(response_content, url, data))

    return response_content


def get(target_path, params={}, times=1):
    """

    :param target_path: str
    :param params: dict
    :param times: int
    :return:
    """
    url = "{}/{}".format(config.API_URL, target_path)
    # headers = Auth().generate_header_token()
    for i in range(times):
        try:
            if params is not None:
                response = requests.get(url, params=params,
                                        # headers=headers
                                        )
            else:
                response = requests.get(url,
                                        # headers=headers
                                        )

            response_content = json.loads(response.content, encoding="utf-8")

            if response_content.get('code') == 200:
                return response_content.get("result")
            else:
                time.sleep(3)
                continue
        except Exception as e:
            raise Exception("PATH {}. No connection could be made because "
                            "the target machine actively refused it. {}".format(url, e))

    if response_content.get('code') != 200:
        raise Exception("ERROR : {}, URL PATH : {}".format(response_content, url))
