import re


def strip_html_tags_regex(html_string):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', html_string)

