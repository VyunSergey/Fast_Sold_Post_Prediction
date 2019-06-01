import re
from typing import List

from utils.lists import list_to_str, list_drop_len


def str_to_list(sep: str, string: str) -> List:
    return str(string).split(sep)


def regexp_replace(pattern: str, replace: str, string: str) -> str:
    return re.sub(pattern, replace, string)


def str_drop_spaces(string: str) -> str:
    return regexp_replace('  ', ' ', regexp_replace('\s', '', string)).strip()


def str_drop_digits(string: str) -> str:
    return regexp_replace('  ', ' ', regexp_replace('\d', '', string)).strip()


def str_drop_words(string: str) -> str:
    pattern = '[a-zA-Zа-яА-ЯёЁ_]'
    return regexp_replace('  ', ' ', regexp_replace(pattern, '', string)).strip()


def str_drop_spec(string: str) -> str:
    pattern = '[!@#$%^&*\(\)_+-<>\?\:"\{\}\[\];'',.\/\\\|]'
    return regexp_replace('  ', ' ', regexp_replace(pattern, '', string)).strip()


def str_drop_len(sep: str, string: str, length: int) -> str:
    return list_to_str(sep, list_drop_len(str_to_list(sep, string), length))


def str_deduplicate(sep: str, string: str) -> str:
    return list_to_str(sep, list(set(str_to_list(sep, string))))


def str_drop_trash(sep: str, string: str, trash: List) -> str:
    return list_to_str(sep, [x for x in str_to_list(sep, string) if not(x in trash)])
