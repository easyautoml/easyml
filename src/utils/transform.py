import pandas as pd
import pickle
import os
import xlsxwriter
import numpy as np
import string


FILE_BASE_DIR = "static/file_upload/"
EXPERIMENTS_BASE_DIR = "static/experiments/"
PREDICT_BASE_DIR = "static/predict/"
EDA_FILE_BASE_DIR = "main/templates/file/eda/"


def get_file_url(file_id):
    return "{}{}".format(FILE_BASE_DIR, file_id)


def get_experiments_url(experiment_id):
    return "{}{}".format(EXPERIMENTS_BASE_DIR, experiment_id)


def get_experiments_dataset_url(experiment_id, file_name):
    root = get_experiments_url(experiment_id)
    return "{}{}{}".format(root, "/utils/data/", file_name)


def get_predict_url(predict_id):
    return "{}{}.xlsx".format(PREDICT_BASE_DIR, predict_id)


def get_file_eda_url(file_id):
    return "{}{}.html".format(EDA_FILE_BASE_DIR, file_id)


class Input:
    def __init__(self, file_url):
        self.file_url = file_url

    def from_csv(self):
        try:

            df_data = pd.read_csv(self.file_url)
            return df_data
        except Exception as e:
            mes = "Can't load file from {}. ERROR : {}".format(self.file_url, e)
            raise Exception(mes)

    def from_pickle(self):
        try:
            with open(self.file_url, 'rb') as f:
                pickle_file = pickle.load(f)
                return pickle_file
        except Exception as e:
            mes = "Can't load file from {}. ERROR : {}".format(dir, e)
            raise Exception(mes)


class Output:
    def __init__(self, file_url):
        self.file_url = file_url

    def to_pickle(self, df_data):
        try:
            with open(self.file_url, 'wb') as f:
                pickle.dump(df_data, f)
        except Exception as e:
            raise Exception(e)


class Excel:
    def __init__(self, file_name):
        self.file_name = file_name

        self.workbook = xlsxwriter.Workbook(file_name)

        self.worksheets = {}

        # Default format
        self.common_format = {
            'header_lv0': self.workbook.add_format({
                'bold': 1,
                'border': 0,
                'align': 'left',
                'valign': 'vcenter',
                # 'fg_color': '#f09a59'
            }),
            'header_lv1': self.workbook.add_format({'bold': True, 'border': 1, 'align': 'center'}),
            'content': {
                'int': self.workbook.add_format({'num_format': '#,##0', 'border': 1}),
                'float': self.workbook.add_format({'num_format': '#,##0.00', 'border': 1}),
                'string': self.workbook.add_format({'border': 1})
            }
        }

    def create_worksheet(self, worksheet_name):

        worksheet = self.workbook.add_worksheet(worksheet_name)

        worksheet.set_default_row(25)

        init_ceil = {
            'row': 0,
            'col': 0
        }

        return {
            'worksheet': worksheet,
            'init_ceil': init_ceil
        }

    def add_data(self, worksheet_name, pd_data, header_lv0=None, is_fill_color_scale=False, columns_order=None):
        """
        Add data into worksheet
        """
        # Step 1. Get or create worksheet name
        cur_worksheet = self.worksheets.get(worksheet_name, None)

        if cur_worksheet is None:
            cur_worksheet = self.create_worksheet(worksheet_name)

        # Add header level 0
        width, height = len(pd_data.columns) - 1, len(pd_data)

        if header_lv0 is not None:
            if width > 1:
                header_lv0_ceil_range = self._get_ceil_range(from_ceil=cur_worksheet.get('init_ceil'),
                                                             to_ceil={'row': cur_worksheet.get('init_ceil').get('row'),
                                                                      'col': cur_worksheet.get('init_ceil').get(
                                                                          'col') + width
                                                                      })
                cur_worksheet.get('worksheet').merge_range(header_lv0_ceil_range, header_lv0,
                                                           self.common_format.get('header_lv0'))
            else:
                cur_worksheet.get('worksheet').write_string(cur_worksheet.get('init_ceil').get('row'),
                                                            cur_worksheet.get('init_ceil').get('col'), str(header_lv0),
                                                            self.common_format.get('header_lv0'))

        # Add header level 1
        _is_header = 1 if header_lv0 is not None else 0

        raw_data_col = pd_data.columns.tolist()
        if columns_order is not None:
            # Remove columns which not include in raw data

            diff_columns_order_vs_raw_data_col = set(columns_order).difference(set(raw_data_col))
            for col in diff_columns_order_vs_raw_data_col:
                columns_order.remove(col)

            # Add left column in raw data into columns order
            diff_raw_data_col_vs_columns_order = set(raw_data_col).difference(set(columns_order))
            columns_order += list(diff_raw_data_col_vs_columns_order)
        else:
            columns_order = pd_data.columns.tolist()

        for i, header in enumerate(columns_order):
            cur_worksheet.get('worksheet').write_string(cur_worksheet.get('init_ceil').get('row') + _is_header,
                                                        cur_worksheet.get('init_ceil').get('col') + i,
                                                        str(header), self.common_format.get('header_lv1'))

        # Re-order of pandas data
        pd_data = pd_data[columns_order]

        # Add content
        for i_row, values in enumerate(pd_data.values):
            for j_col, value in enumerate(values):
                if self._is_number(value):
                    _format = self.common_format.get('content').get('int') if self._is_int(value) else \
                                self.common_format.get('content').get('float')
                    try:
                        cur_worksheet.get('worksheet').write_number(
                            cur_worksheet.get('init_ceil').get('row') + 1 + _is_header + i_row,
                            cur_worksheet.get('init_ceil').get('col') + j_col, value, _format)
                    except Exception as e:
                        print("Value : {}. Error : {}".format(value, e))
                        continue
                else:
                    try:
                        x = cur_worksheet.get('init_ceil').get('row') + 1 + _is_header + i_row
                        y = cur_worksheet.get('init_ceil').get('col') + j_col
                        cur_worksheet.get('worksheet').write_string(x, y, str(value),
                                                                    self.common_format.get('content').get('string'))
                    except Exception as e:
                        x = cur_worksheet.get('init_ceil').get('row') + 1 + _is_header + i_row
                        y = cur_worksheet.get('init_ceil').get('col') + j_col

        self._fit_columns_width(worksheet=cur_worksheet, pd_data=pd_data)

        # Auto Color Scale. Fill all table
        if is_fill_color_scale:
            row, col = cur_worksheet.get('init_ceil').get('row'), cur_worksheet.get('init_ceil').get('col')
            full_ceil_range = self._get_ceil_range(
                from_ceil={'row': row + 1 + _is_header, 'col': col},
                to_ceil={'row': row + 1 + _is_header + height, 'col': col + width}
                )
            cur_worksheet.get('worksheet').conditional_format(full_ceil_range, {'type': '3_color_scale'})

        cur_worksheet.get('init_ceil')['col'] += (width + 1)

        self.worksheets[worksheet_name] = cur_worksheet

    def _get_ceil_range(self, from_ceil, to_ceil):
        from_add = self._address(from_ceil)
        to_add = self._address(to_ceil)
        return "{}:{}".format(from_add, to_add)

    @staticmethod
    def _address(ceil):
        """
        Convert location x,y to excel address. Ex (0,0) -> A1
        """
        _col, _row = ceil.get('col') + 1, ceil.get('row') + 1
        sub_column_name = ""
        if _col > len(string.ascii_uppercase):
            stt = _col % len(string.ascii_uppercase)
            sub_column_name = string.ascii_uppercase[stt - 1]
            column_name = string.ascii_uppercase[_col // len(string.ascii_uppercase) - 1]
        else:
            column_name = string.ascii_uppercase[_col - 1]

        return "{}{}{}".format(column_name, sub_column_name, _row)

    @staticmethod
    def _fit_columns_width(worksheet, pd_data):
        col_widths = [max([len(str(s)) for s in pd_data[col].values] + [len(col)]) for col in pd_data.columns]

        for i, width in enumerate(col_widths):
            col = worksheet.get('init_ceil').get('col')
            worksheet.get('worksheet').set_column(col + i, col + i + 1, width + 3)

    @staticmethod
    def _is_number(value):
        return (type(value) is not str) and (str(value).replace(".", "").replace("-", "").isnumeric())

    @staticmethod
    def _is_int(value):
        return type(value) in [int, np.int8, np.int16, np.int32, np.int64]

    def save(self):
        if not os.path.exists(os.path.dirname(self.file_name)):
            os.makedirs(os.path.dirname(self.file_name))
        self.workbook.close()
