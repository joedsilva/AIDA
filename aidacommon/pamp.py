import logging
import re

import pandas as pd

from aidacommon.dborm import CMP, DATE, Q, F, C


def convert_type(func):
    def inner(data, sc):
        # when condition type is date
        if isinstance(sc._col2_, DATE):
            ndata = pd.to_datetime(data[sc._col1_], format='%Y-%m-%d', errors='coerce')
            sc = Q(sc._col1_, sc._col2_.__date__, sc._operator_)
        elif isinstance(sc._col2_, C):
            ndata = data[sc._col1_]
            sc = Q(sc._col1_, sc._col2_._val_, sc._operator_)
        else:
            ndata = data[sc._col1_]
        logging.info(
            f'PAMP after converting: data={ndata.head(10)}, sc.col1={sc._col1_}, <{ndata[0]}: {type(ndata[0])}, '
            f'col2={sc._col2_}, type: {type(sc._col2_)}')
        return func(ndata, sc)

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
    logging.info(f'MAP_IN: data: {data.head(10)}, col1 = {sc._col1_}, col2 = {sc._col2_} ')
    return data.isin(sc._col2_)


@convert_type
def map_notin(data, sc):
    return data.isnotin(sc._col2_)


MATCH_PATTERNS = ['^%([^%]+)$', '^([^%]+)%$', '^%([^%]+)%$']


@convert_type
def map_like(data, sc):
    s = sc._col2_
    for idx, p in enumerate(MATCH_PATTERNS):
        rex = re.compile(p)
        match = rex.match(sc._col2_)
        if match:
            logging.info(f'MATCH! pattern = {p}, group = {match.group(1)}, id={idx}')
            s = match.group(1)
            if idx == 0:
                return data.str.endswith(s)
            elif idx == 1:
                logging.info(data.str.startsswith(s).head(10))
                return data.str.startswith(s)
            else:
                return data.str.contains(s)
    return data == s


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
    CMP.AND: map_and,
    CMP.NOT: map_not,

    # subquery
    CMP.IN: map_in,

    CMP.LIKE: map_like,
}


def fop2pandas(data1, data2, op):
    if op == F.OP.ADD:
        return data1 + data2
    if op == F.OP.SUBTRACT:
        return data1 - data2
    if op == F.OP.MULTIPLY:
        return data1 * data2
    if op == F.OP.DIVIDE:
        return data1 / data2
    if op == F.OP.NEGATIVE:
        return -data1
    if op == F.OP.YEAR:
        return data1.dt.year
    if op == F.OP.MONTH:
        return data1.dt.month
    if op == F.OP.DAY:
        return data1.dt.day


def f2pandas(data, f):
    cols = [f._col1_, f._col2_]
    for col in cols:
        if isinstance(col, F):
            col = f2pandas(data, col)
        elif isinstance(col, str):
            col = data[col]
    return fop2pandas(cols[0], cols[1], f.OP)
