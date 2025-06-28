#%%
from typing import List, Dict
import json
import requests
from requests.exceptions import ChunkedEncodingError
from time import sleep

headers = {
    'authority': 'old.nasdaq.com',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'sec-fetch-site': 'cross-site',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-user': '?1',
    'sec-fetch-dest': 'document',
    'referer': 'https://handsome-tristan.org',
    'accept-language': 'en-US,en;q=0.9',
    'cookie': 'AKA_A2=A; NSC_W.TJUFEFGFOEFS.OBTEBR.443=ffffffffc3a0f70e45525d5f4f58455e445a4a42378b',
}
csv_url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25&offset=0&download=true"
data: List[Dict] = None


def p2f(x:str) -> float:
    if x.strip().endswith("%"):
        return float(x.strip('%'))/100
    else:
        return float(x.strip())


def request_json():
    try:
        response = requests.get(csv_url, headers=headers)
    except ChunkedEncodingError as e:
        print(e)
        sleep(60)
        response = request_json()
    return response


def get_tickers(mktcap_min_million=None, mktcap_max_million=None,
                price_min=None, price_max=None,
                change_pct_min=None, change_pct_max=None,
                volume_min=None, should_reload_data: bool = True) -> List[str]:
    global data
    if mktcap_min_million:
        mktcap_min_million = mktcap_min_million * 1000000
    if mktcap_max_million:
        mktcap_max_million = mktcap_max_million * 1000000
    if should_reload_data:
        response = request_json()
        data_raw = json.loads(response.text)
        data = data_raw["data"]["rows"]
    symbols: List[str] = []
    for symbol_data in data:
        if mktcap_min_million:
            if symbol_data["marketCap"] =="" or float(symbol_data["marketCap"]) < mktcap_min_million:
                continue
        if mktcap_max_million:
            if symbol_data["marketCap"] =="" or float(symbol_data["marketCap"]) > mktcap_max_million:
                continue
        if price_min:
            if symbol_data["lastsale"] =="" or float(symbol_data["lastsale"].replace("$", "")) < price_min:
                continue
        if price_max:
            if symbol_data["lastsale"] =="" or float(symbol_data["lastsale"].replace("$", "")) > price_max:
                continue
        if change_pct_min:
            if symbol_data["pctchange"] =="" or p2f(symbol_data["pctchange"]) < change_pct_min:
                continue
        if change_pct_max:
            if symbol_data["pctchange"] =="" or p2f(symbol_data["pctchange"]) > change_pct_max:
                continue
        if volume_min:
            if symbol_data["volume"] =="" or int(symbol_data["volume"]) < volume_min:
                continue
        symbols.append(symbol_data["symbol"])
    symbols = list(set(symbols))
    return symbols


#%%
if __name__ == "__main__":
    symbols = get_tickers(should_reload_data=True)
    print(len(symbols))

# %%
