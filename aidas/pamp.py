import logging
import re

import pandas as pd
import numpy as np

from aidacommon.dborm import CMP, DATE, Q, F, C, EXTRACT, JOIN, SUBSTRING, CASE

def convert_type(func):
    def inner(data, sc):
        # when condition type is date
        from aidas.dborm import DataFrame
        if isinstance(sc._col2_, DATE):
            ndata = pd.to_datetime(data[sc._col1_], format='%Y-%m-%d', errors='coerce')
            sc = Q(sc._col1_, sc._col2_.__date__, sc._operator_)
        elif isinstance(sc._col2_, C):
            ndata = data[sc._col1_]
            sc = Q(sc._col1_, sc._col2_._val_, sc._operator_)
        elif isinstance(sc._col2_, DataFrame):
            ndata = data[sc._col1_]
            sc = Q(sc._col1_, sc._col2_.execute_pandas(), sc._operator_)
        else:
            ndata = data[sc._col1_]
            ndata.astype(type(sc._col2_))
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


@convert_type
def map_gtall(data, sc):
    return data > max(sc._col2_)


@convert_type
def map_gtany(data, sc):
    return data > min(sc._col2_)


@convert_type
def map_gteall(data, sc):
    return data >= max(sc._col2_)


@convert_type
def map_gteany(data, sc):
    return data >= min(sc._col2_)

@convert_type
def map_ltall(data, sc):
    return data < min(sc._col2_)


@convert_type
def map_ltany(data, sc):
    return data > max(sc._col2_)


@convert_type
def map_lteall(data, sc):
    return data <= min(sc._col2_)


@convert_type
def map_lteany(data, sc):
    return data <= max(sc._col2_)


def map_or(data, sc):
    col1 = sc._col1_
    col2 = sc._col2_
    return select2pandas(data, col1) | select2pandas(data, col2)


def map_and(data, sc):
    col1 = sc._col1_
    col2 = sc._col2_
    return select2pandas(data, col1) & select2pandas(data, col2)


def map_not(data, sc):
    col1 = sc._col1_
    col2 = sc._col2_
    return ~ select2pandas(data, col1)


@convert_type
def map_in(data, sc):
    ls = sc._col2_
    logging.info(f'MAP_IN: data: {data.head(10)}, col1 = {sc._col1_}, col2 = {sc._col2_} ')
    return data.isin(ls)


@convert_type
def map_notin(data, sc):
    ls = sc._col2_
    logging.info(f'MAP_NOTIN: data: {data.head(10)}, col1 = {sc._col1_}, col2 = {sc._col2_} ')
    return ~data.isin(ls)


MATCH_PATTERNS = ['^%([^%]+)$', '^([^%]+)%$', '^%([^%]+)%$']


@convert_type
def map_like(data, sc):
    s = sc._col2_
    for idx, p in enumerate(MATCH_PATTERNS):
        rex = re.compile(p)
        match = rex.match(sc._col2_)
        if match:
            #logging.info(f'MATCH! pattern = {p}, group = {match.group(1)}, id={idx}')
            s = match.group(1)
            if idx == 0:
                return data.str.endswith(s)
            elif idx == 1:
                #logging.info(data.str.startsswith(s).head(10))
                return data.str.startswith(s)
            else:
                return data.str.contains(s)
    return data == s

def map_notlike(data, sc):
    return ~ map_like(data, sc)

def map_null(data, sc):
    return data.isna()

def map_notnull(data, sc):
    return data.notna()


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
    CMP.GTALL: map_gtall,
    CMP.GTEALL: map_gteall,
    CMP.GTANY: map_gtany,
    CMP.GTEANY: map_gteany,
    CMP.LTALL: map_ltall,
    CMP.LTEALL: map_lteall,
    CMP.LTANY: map_ltany,
    CMP.LTEANY: map_lteany,

    # logical operator
    CMP.OR: map_or,
    CMP.AND: map_and,
    CMP.NOT: map_not,

    # subquery
    CMP.IN: map_in,
    CMP.NOTIN: map_notin,
    CMP.EXISTS: None,
    CMP.NOTEXISTS: None,

    CMP.LIKE: map_like,
    CMP.NOTLIKE: map_notlike,

    CMP.NULL: map_null,
    CMP.NL: map_null,
    CMP.NOTNULL: map_notnull,
    CMP.NNL: map_notnull
}


def select2pandas(data, sc):
    filter_func = PCMP_MAP.get(sc._operator_, map_eq)
    return filter_func(data , sc)


PJOIN_MAP = {
    JOIN.INNER_JOIN: 'inner',
    JOIN.CROSS_JOIN: 'cross',
    JOIN.LEFT_OUTER_JOIN: 'left',
    JOIN.RIGHT_OUTER_JOIN: 'right',
    JOIN.FULL_OUTER_JOIN: 'outer'
}

PAGG_MAP = {
    'COUNT': ('size', np.size),
    'MAX': ('amax', np.max),
    'MIN': ('amin', np.min),
    'SUM': ('sum', np.sum),
    'AVG': ('average', np.average)
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
        ndata = pd.to_datetime(data1, format='%Y-%m-%d', errors='coerce')
        return ndata.dt.year.astype('int32')
    if op == F.OP.MONTH:
        ndata = pd.to_datetime(data1, format='%Y-%m-%d', errors='coerce')
        return ndata.dt.month.astype('int32')
    if op == F.OP.DAY:
        ndata = pd.to_datetime(data1, format='%Y-%m-%d', errors='coerce')
        return ndata.dt.day.astype('int32')
    return data1


def f2pandas(data, f):
    """

    @param data: Entire data stored in transform or tabularData
    @param f: F class
    @return: Output column
    """
    cols = [f._col1_, f._col2_]
    logging.info(f"f2pandas col1: {cols[0]}, col2: {cols[1]}")

    if isinstance(f, SUBSTRING):
        col = f._col1_
        start = max(0, f._fromidx_- 1) # to have the same behavior as SQL
        length = f._len_
        # clamp the start index to 0 if negative, and clamp the stop index to at least start index
        # to avoid negative number
        #logging.info(f'In side substring: {col}-{start}-{length}')
        return data[col].str[start:max(start + length, start)]
    elif isinstance(f, CASE):
        # initialize the output column with the default value
        defval = f._deflt_._val_ if hasattr(f._deflt_, '_val_') else f._deflt_
        output = pd.DataFrame(defval, index=data.index, columns=['output'])
        for (case, val) in f._cases_:
            #logging.info(f'F Case |||||| case: {case}: {type(case)}, val: {val}: {type(val)}, op={case._operator_}')
            # update the output column based on Q conditions
            cond = PCMP_MAP[case._operator_](data, case)
            output[cond] = val
            #logging.info(f"after case: cond = {cond.head(10)}, output={output.head(10)}")
        return output
    for i in range(len(cols)):
        if isinstance(cols[i], F):
            #logging.info(f"IS F")
            cols[i] = f2pandas(data, cols[i])
        elif isinstance(cols[i], str):
            cols[i] = data[cols[i]]
    logging.info(f"f2pandas after: col1: {cols[0]}, col2: {cols[1]}")
    return fop2pandas(cols[0], cols[1], f._operator_)
