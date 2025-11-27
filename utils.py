import re
import requests
import urllib.parse

def strip_html_tags_regex(html_string):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', html_string)

def get_article_data(article_id: int):
    """
    Fetch full article data from VNExpress GW API using the given article_id.
    """
    base_url = "https://gw.vnexpress.net/ar/get_full"
    
    # Define query parameters
    params = {
        "article_id": article_id,
        "data_select": urllib.parse.quote(
            "article_id,article_type,title,share_url,thumbnail_url,publish_time,lead,privacy,original_cate,article_category",
            safe=""
        ),
        # "exclude_id": urllib.parse.quote(
        #     "4662602,4662536,4662665,4662634,4662635,4656425,4653241,4662807,4662809,4662473",
        #     safe=""
        # ),
        # "thumb_size": urllib.parse.quote("680x408,500x300,300x180", safe=""),
        # "thumb_quality": 100,
        # "thumb_dpr": urllib.parse.quote("1,2", safe=""),
        # "thumb_fit": "crop"
    }

    # Compose final URL
    url = f"{base_url}?article_id={article_id}&data_select={params['data_select']}"

    # Send request
    response = requests.get(url, timeout=10)
    
    if response.status_code == 200:
        try:
            data = response.json()
            return data
        except ValueError:
            print("Error: Response is not valid JSON.")
            return None
    else:
        print(f"Error: HTTP {response.status_code}")
        return None
    
def build_input_data(article_id: int):
    data = get_article_data(article_id)
    if data is None:
        return None
    title = data["data"]["title"]
    lead = strip_html_tags_regex(data["data"]["lead"])
    content = strip_html_tags_regex(data["data"]["content"])
    content_stripped = ". ".join(content.split(". ")[:8])
    return title + "\n\n" + lead + "\n\n" + content_stripped