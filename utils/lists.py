from typing import List


def list_equals(lst1: List, lst2: List) -> bool:
    return len(list(set(lst1).difference(set(lst2)))) == 0 and len(list(set(lst2).difference(set(lst1)))) == 0


def list_contains(lst: List, elements: List) -> bool:
    return all([elem in lst for elem in elements])


def list_to_str(sep: str, lst: List) -> str:
    return sep.join(lst)


def list_drop_na(lst: List) -> List:
    return [x for x in lst if len(str(x)) > 0]


def list_drop_len(lst: List, length: int) -> List:
    return [x for x in lst if len(str(x)) > length]
