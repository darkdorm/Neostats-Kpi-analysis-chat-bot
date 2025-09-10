# utils/websearch.py
import requests
from config.config import CONFIG

def search_web(query: str, num: int = 3):
    """
    Perform a Google Custom Search query if keys are configured.
    Returns list of {title, snippet, link}.
    """
    api_key = CONFIG.get("CSE_API_KEY")
    cx = CONFIG.get("CSE_CX")

    if not api_key or not cx:
        return [{
            "title": "Web search not configured",
            "snippet": "Set CSE_API_KEY and CSE_CX in your .env or config/config.py.",
            "link": ""
        }]

    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": api_key, "cx": cx, "q": query, "num": num}
    try:
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        items = []
        for it in data.get("items", [])[:num]:
            items.append({
                "title": it.get("title"),
                "snippet": it.get("snippet"),
                "link": it.get("link")
            })
        return items
    except Exception as e:
        return [{
            "title": "Search error",
            "snippet": str(e),
            "link": ""
        }]
