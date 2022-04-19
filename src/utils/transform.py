import pandas as pd
import pickle
import os
import xlsxwriter
import numpy as np
import string
from sklearn.neighbors import KernelDensity
from numerize.numerize import numerize
import math
from pathlib import Path



FILE_BASE_DIR = "static/file_upload/"
EXPERIMENTS_BASE_DIR = "static/experiments/"
PREDICT_BASE_DIR = "static/predict/"
EVALUATION_BASE_DIR = "static/evaluation/"
EDA_FILE_BASE_DIR = "main/templates/file/eda/"


def get_file_url(file_id):
    #return "{}{}".format(FILE_BASE_DIR, file_id)
    path = Path(FILE_BASE_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path.joinpath(str(file_id))


def get_experiments_url(experiment_id):
    #return "{}{}".format(EXPERIMENTS_BASE_DIR, experiment_id)
    path = Path(EXPERIMENTS_BASE_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path.joinpath(str(experiment_id))


def get_evaluation_url(evaluation_id):
    #return "{}{}.xlsx".format(EVALUATION_BASE_DIR, evaluation_id)
    path = Path(EVALUATION_BASE_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path.joinpath(str(evaluation_id))


def get_experiments_dataset_url(experiment_id, file_name):
    root = get_experiments_url(experiment_id)
    #return "{}{}{}".format(root, "/utils/data/", file_name)
    path = Path(root, "utils/data/")
    path.mkdir(parents=True, exist_ok=True)
    return path.joinpath(str(file_name))


def get_predict_url(predict_id):
    #return "{}{}.xlsx".format(PREDICT_BASE_DIR, predict_id)
    path = Path(PREDICT_BASE_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path.joinpath(f"{predict_id}.xlsx") "{}{}.xlsx".format(PREDICT_BASE_DIR, predict_id)


def get_file_eda_url(file_id):
    #return "{}{}.html".format(EDA_FILE_BASE_DIR, file_id)
    path = Path(EDA_FILE_BASE_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path.joinpath(f"{file_id}.html")


def distribution_density(x, density_num):
    """
    Calculation probability density of x
    :param x: must be numpy array
    :param density_num:
    :return:
    """

    x_d = np.array([i / density_num for i in range(0, density_num)])

    kde = KernelDensity(bandwidth=0.05, kernel='gaussian')

    kde.fit(x[:, None])

    log_prob = kde.score_samples(x_d[:, None])

    prob = [np.exp(x) for x in log_prob]

    return prob


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


class Histogram:
    def __init__(self, pd_input, group_col, bin_num=10):
        self.bin_num = bin_num
        self.pd_data = pd_input.copy()
        self.pd_data.reset_index(inplace=True)

        self.decimal_num = 3
        self.group_col = group_col

        # TODO : After implement define category at upload data step. Remove astype float function
        try:
            self.pd_data[group_col] = self.pd_data[group_col].astype(float)
        except:
            pass

        self.series = self.pd_data[group_col]

        _is_category = self.series.dtype.name in ['category', 'object', 'datetime64[ns]']

        if _is_category:
            self.category_grouping()
        else:
            # Check data of series is integer or float
            self.series_is_integer = self.check_integer()

            # Outlier calculation
            self.outlier = self.outlier_cal()

            # Calculation range_interval
            self.range_interval = self.range_interval_cal()

            # Check group name resizeable
            self.group_name_resizeable = self.check_group_name_resizeable()

            self.numeric_grouping()

    def category_grouping(self):
        # 1. Create Group Name and Group Order
        self.pd_data['group_name'] = self.series

        self.pd_data['group_name'] = self.series.apply(lambda x: str(x).strip().lower())
        self.pd_data['group_order'] = self.series.apply(lambda x: self._parse_float(x))
        self.pd_data['is_outlier'] = False

    def numeric_grouping(self):

        if self.range_interval.get('type') == 1:
            self.pd_data[['group_name', 'group_order', 'is_outlier']] = self.pd_data[self.group_col].apply(
                self.group_name_allocate_type1).to_list()

        if self.range_interval.get('type') == 2:
            self.pd_data[['group_name', 'group_order', 'is_outlier']] = self.pd_data[self.group_col].apply(
                self.group_name_allocate_type2).to_list()

        if self.range_interval.get('type') == 3:
            self.pd_data[['group_name', 'group_order', 'is_outlier']] = self.pd_data[self.group_col].apply(
                self.group_name_allocate_type3).to_list()

    def outlier_cal(self):
        """
        Calculation IQR to define outlier.
        returns:
            If lower outlier exist, return lower_outlier values. If not, return None
        """
        min_series, max_series = min(self.series), max(self.series)

        q1 = np.percentile(self.series, 25, interpolation='midpoint')
        q3 = np.percentile(self.series, 75, interpolation='midpoint')
        iqr = q3 - q1
        lower_outlier, upper_outlier = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        if self.series_is_integer:
            # Round number. Lower : 0.9 -> 0, -1.9 -> 1
            lower_outlier = int(math.floor(lower_outlier))

            # Round number. Upper : 0.1 -> 1, -0.9 -> 0
            upper_outlier = int(math.ceil(upper_outlier))

        exist_lower_outlier = min_series < lower_outlier
        exist_upper_outlier = max_series > upper_outlier

        return {
            'IQR': iqr,
            'Q1': q1, 'Q3': q3, 'min_series': min_series, 'max_series': max_series,
            'lower_outlier': lower_outlier if exist_lower_outlier else None,
            'upper_outlier': upper_outlier if exist_upper_outlier else None
        }

    def range_interval_cal(self):
        """
        Create interval from data. Example : [1,4,2,3,5,9...100] -> [1, 10, 20, 30, .. 100] with BIN = 10
        :return:

        1. Interval will have three type.
            - Type 1. Interval for numeric, but IQR == 0. lower == upper
                Ex : Series = [-2, -1, 1, 1,1,1,1,1,1, 2,3]. range_interval_value : [lower_interval, q1, upper_interval]

            - Type 2. Interval for numeric, but series's unique number is not enough to split N bin. [1,2,3,4]
                Ex : Series  = [1,1,1,2,3,4,4]. Bin = 5. Unique Number (Series) = 4 < Bin

            - Type 3. Interval for numeric, series's unique number is enough to split N bin. [1,2,3,4,5,6....N]
                Ex : Series = [1,2,3,4,5,6,7,8,9,10]. Bin = 5. Unique Number (Series) = 10 > 5.

        """
        min_series, max_series = min(self.series), max(self.series)

        # -------------- TYPE 1 --------------
        if self.outlier.get('IQR') == 0:
            range_interval = [min_series, self.outlier.get('Q1'), max_series]
            return {
                'type': 1,
                'value': range_interval
            }

        # Define start point, end point
        start_point = min_series if self.outlier.get('lower_outlier') is None else self.outlier.get('lower_outlier')
        end_point = max_series if self.outlier.get('upper_outlier') is None else self.outlier.get('upper_outlier')

        series_unique = self.series[(self.series >= start_point) & (self.series <= end_point)]
        series_unique_sorted = np.sort(series_unique.unique()).tolist()

        # -------------- TYPE 2 --------------
        if len(series_unique_sorted) < self.bin_num * 2:
            range_interval = series_unique_sorted

            if self.outlier.get('lower_outlier') is not None:
                range_interval.insert(0, round(min_series, self.decimal_num))
            if self.outlier.get('upper_outlier') is not None:
                range_interval.append(round(max_series, self.decimal_num))
            return {
                'type': 2,
                'value': range_interval
            }

        # -------------- TYPE 3 --------------
        interval_width = (end_point - start_point) / self.bin_num
        # Convert type of interval, start point and end point to Int
        if self.series_is_integer:
            interval_width = int(math.ceil(interval_width))
            start_point, end_point = int(start_point), int(end_point)

        range_interval, i = [], 0
        while True:
            _iter_val = start_point + (i * interval_width)

            if _iter_val >= end_point:
                break
            else:
                range_interval.append(_iter_val)
                i += 1

        range_interval.append(end_point)

        # Add outlier for interval range
        if self.outlier.get('lower_outlier') is not None:
            range_interval.insert(0, min_series)
        if self.outlier.get('upper_outlier') is not None:
            range_interval.append(max_series)

        return {
            'type': 3,
            'value': range_interval
        }

    def group_name_allocate_type1(self, x):
        """  -------------- TYPE 1 --------------
        Allocate group name for X. With interval range type 1. Range_Interval = [a, b, c]
        - a : None or Number. If not none, this is minimum value of range and all value from [a, b) is outlier
        - b : Q1 values
        - c : None or Number. If not none, this is maximum values of range and all value from (b, c] is outlier
        :param x:
        :return: Group Name, Group Order, Is outlier
        """
        _range_interval = self.range_interval.get('value').copy()

        if self.outlier.get('lower_outlier') is not None:
            if _range_interval[0] <= x < _range_interval[1]:
                _from, _to = round(_range_interval[0], self.decimal_num), round(_range_interval[1], self.decimal_num)
                group_name, group_order = self.create_group_name(_from, _to, "[", ")")
                is_outlier = True
                return group_name, 0, is_outlier

        if x == _range_interval[1]:
            _val = x
            if self.group_name_resizeable:
                _val = numerize(x)

            group_name = '{}'.format(_val)
            is_outlier = False
            return group_name, 1, is_outlier

        if self.outlier.get('upper_outlier') is not None:
            if _range_interval[1] < x <= _range_interval[2]:
                _from, _to = round(_range_interval[1], self.decimal_num), round(_range_interval[2], self.decimal_num)
                group_name, group_order = self.create_group_name(_from, _to, "(", "]")
                is_outlier = True
                return group_name, 2, is_outlier

        return 'None', self.outlier.get('min_series') - 1, False

    def group_name_allocate_type2(self, x):
        """ -------------- TYPE 2 --------------
        Allocate group name for X. With interval range type 2. Range_Interval = [a, b, c, d, e, f]
        :param x:
        :return: group_name: str, group_order:str, is_outlier: boolean
        """
        _range_interval = self.range_interval.get('value').copy()

        if self.outlier.get('lower_outlier') is not None:
            _from, _to = _range_interval.pop(0), _range_interval[0]

            if _from <= x < _to:
                group_name, group_order = self.create_group_name(_from, _to, "[", ")")
                is_outlier = True
                return group_name, group_order, is_outlier

        if self.outlier.get('upper_outlier') is not None:
            _to, _from = _range_interval.pop(-1), _range_interval.pop(-1)

            if _from <= x <= _to:
                group_name, group_order = self.create_group_name(_from, _to, "[", "]")
                is_outlier = True
                return group_name, group_order, is_outlier

        if x in _range_interval:
            group_name, group_order = str(x), x
            if self.group_name_resizeable:
                group_name = numerize(x)
            is_outlier = False
            return group_name, group_order, is_outlier

        return 'None', self.outlier.get('min_series') - 1, False

    def group_name_allocate_type3(self, x):
        """ -------------- TYPE 3 --------------
        Allocate group name for X. With interval range type 3. Range_Interval = [a, b, c, d, e, f]
        :param x:
        :return: group_name: str, group_order:str, is_outlier: boolean
        """
        _range_interval = self.range_interval.get('value').copy()

        # Outlier Define
        is_outlier = False
        _lower_outlier, _upper_outlier = self.outlier.get('lower_outlier'), self.outlier.get('upper_outlier')
        if _lower_outlier is not None:
            is_outlier = x < _lower_outlier

        if _upper_outlier is not None:
            is_outlier = x >= _upper_outlier or is_outlier

        len_range = len(_range_interval)
        for i in range(len_range - 2):
            _from, _to = _range_interval[i], _range_interval[i + 1]

            if _from <= x < _to:
                group_name, group_order = self.create_group_name(_from, _to, "[", ")")
                return group_name, group_order, is_outlier

        # Last element
        _from, _to = _range_interval[-2], _range_interval[-1]
        if _from <= x <= _to:
            group_name, group_order = self.create_group_name(_from, _to, "[", "]")
            return group_name, group_order, is_outlier

        return 'None', self.outlier.get('min_series') - 1, False

    def create_group_name(self, from_val, to_val, from_sym, to_sym):
        _from_name, _to_name = round(from_val, self.decimal_num), round(to_val, self.decimal_num)
        if self.group_name_resizeable:
            _from_name, _to_name = numerize(from_val), numerize(to_val)

        group_name = '{}{}, {}{}'.format(from_sym, _from_name, _to_name, to_sym)
        group_order = from_val
        return group_name, group_order

    def check_category(self):
        """
        Series will determined as category if data type is category, object, datetime or unique number lester than bin
        :return: Bool
        """
        return self.series.dtype.name in ['category', 'object', 'datetime64[ns]']

    def check_integer(self):
        is_integer = self.series.dtype in [np.int, np.int8, np.int16, np.int32, np.int64]

        if is_integer:
            self.series = self.series.astype(int)
            return True

        # Check case data type is float 0.0 1.0 2.0 but it is integer
        if self.series.dtype in [np.float, np.float16, np.float32, np.float64]:
            _check_all_integer = True

            for row in self.series:
                _check_all_integer = row.is_integer()
                if not _check_all_integer:
                    self.series = self.series.astype(float)
                    return False
        return True

    def check_group_name_resizeable(self):
        """
        Group name will change from (1000, 1200] -> (1K, 2K]
        Bad case : group name will be (1000.1, 1000.2) -> (1K, 1K).
        Check if Bad case happen
        """
        resizeable = True

        _range_interval = self.range_interval.get('value')

        len_range = len(_range_interval)
        for i in range(len_range - 1):
            _from, _to = _range_interval[i], _range_interval[i + 1]

            try:
                _str_from, _str_to = numerize(_from), numerize(_to)
            except Exception as e:
                print("ERROR :", e)
                return False

            if _str_from == _str_to:
                resizeable = False
        return resizeable

    def plot(self):
        self.pd_data.boxplot(column=[self.group_col])

    @staticmethod
    def _parse_float(value):
        try:
            return float(value)
        except Exception:
            return 0
