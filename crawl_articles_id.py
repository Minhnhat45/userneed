import requests
from bs4 import BeautifulSoup
from lxml import html
URL = "https://vnexpress.net/tam-su/hen-ho"

def crawl_article_links(url: str):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Cách 1: Dựa trên các attribute giống ví dụ của bạn
    # <a data-medium="Item-1" data-thumb="1" ...>
    anchors = soup.select('a[data-medium^="Item-"][data-thumb]')

    links = []
    for a in anchors:
        href = a.get("href")
        if href:
            links.append(href)

    # Loại bỏ trùng lặp, giữ nguyên thứ tự
    unique_links = list(dict.fromkeys(links))
    return unique_links


def crawl_article_links_xpath(url: str):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()

    tree = html.fromstring(resp.text)

    # Nếu muốn chính xác đúng section[6]/div/div[1]/div[1]/article/h2/a:
    # links = tree.xpath('/html/body/section[6]/div/div[1]/div[1]/article/h2/a/@href')

    # Thực tế dùng tương đối "dễ thở" hơn:
    #          /html/body/section[2]/div[1]/div/div[1]/article/h3/a
    #          /html/body/section[6]/div/div/div/div/article[1]/h3/a
    #          /html/body/section[6]/div/div/div/div/article[2]/h3/a
    #          /html/body/section[6]/div/div/div/div/article[3]/h3/a
    #                   //section[2]/div[1]//div[1]/article/h3/a
    #          /html/body/section[6]/div/div[1]/div[1]/article/h2/a
    # links = tree.xpath('//section[6]/div/div/div/div///h3/a/@href')
    xpath_expr = "/html/body/section[6]/div/div/div/div/article[3]/h3/a/@href"
    links = tree.xpath(xpath_expr)

    # Hoặc đơn giản hơn nữa nếu cấu trúc thay đổi:
    # links = tree.xpath('//article/h2/a/@href')

    # Loại trùng
    unique_links = list(dict.fromkeys(links))
    return unique_links

if __name__ == "__main__":
    links = crawl_article_links_xpath(URL)
    print(links)
    out_id = []
    for link in links:
        out_id.append(int(link.split(".")[1].split("-")[-1]))
    
    print(out_id[:10])
