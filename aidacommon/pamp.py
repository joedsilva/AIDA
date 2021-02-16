import pandas as pd

from aidacommon.dborm import CMP, DATE, Q


def convert_type(func):
    def inner(data, sc):
        # when condition type is data
        if isinstance(sc._col2_, DATE):
            data = pd.to_datetime(data[sc._col1_], format='%Y-%m-%d', errors='coerce')
            sc = Q(sc._col1_, sc._col2_.__date__, sc._operator_)
        else:
            data = data[sc._col1_]
        return func(data, sc)
    return inner


@convert_type
def map_eq(data, sc):
    return data == sc._col2_


@convert_type
def map_ne(data, sc):
    return data != sc._col2_


@convert_type
def map_gt(data, sc):
    return data > sc._col2_


@convert_type
def map_gte(data, sc):
    return data >= sc._col2_


@convert_type
def map_lt(data, sc):
    return data < sc._col2_


@convert_type
def map_lte(data, sc):
    return data <= sc._col2_


def map_or(data, sc):
    col1 = sc._col1_
    col2 = sc._col2_
    return PCMP_MAP[col1._operator_](data, col1) | PCMP_MAP[col2._operator_](data, col2)


def map_and(data, sc):
    col1 = sc._col1_
    col2 = sc._col2_
    return PCMP_MAP[col1._operator_](data, col1) & PCMP_MAP[col2._operator_](data, col2)


def map_not(data, sc):
    col1 = sc._col1_
    col2 = sc._col2_
    return ~ PCMP_MAP[col1._operator_](data, col1)


@convert_type
def map_in(data, sc):
    data2 = sc._col2.__data__
    return data[sc._col1_].isin()


@convert_type
def map_notin(data, sc):
    data2 = sc._col2.__data__
    return data[sc._col1_].isnotin()


@convert_type
def map_like(data, sc):
    pass

PCMP_MAP = {
    # binary numeric/str value comparison
    CMP.EQ: map_eq,
    CMP.EQUAL: map_eq,
    CMP.NOTEQUAL: map_ne,
    CMP.NE: map_ne,
    CMP.GREATERTHAN: map_gt,
    CMP.GT: map_gt,
    CMP.GREATERTHANOREQUAL: map_gte,
    CMP.GTE: map_gte,
    CMP.LESSTHAN: map_lt,
    CMP.LT: map_lt,
    CMP.LESSTHANOREQUAL: map_lte,
    CMP.LTE: map_lte,

    # logical operator
    CMP.OR: map_or,
    CMP.NOT: map_not,

    #subquery
    CMP.IN: map_in,
}