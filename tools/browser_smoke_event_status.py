import os

from playwright.sync_api import sync_playwright


BASE_URL = os.environ.get("AGRIVISION_UI_BASE_URL", "http://127.0.0.1:5000")


with sync_playwright() as playwright:
    browser = playwright.chromium.launch(
        headless=True,
        args=["--disable-dev-shm-usage", "--disable-gpu", "--no-sandbox"],
    )
    page = browser.new_page()
    page.route("**/api/offline_events/sync_status", lambda route: route.abort())
    page.goto(f"{BASE_URL}/", wait_until="domcontentloaded")
    page.wait_for_selector("#eventSyncInfo")
    page.wait_for_timeout(500)
    status_text = page.locator("#eventSyncInfo").inner_text()
    assert status_text == "事件同步状态暂不可用", status_text
    assert page.locator("#statusBadge").count() == 1
    print("AGRIVISION_BROWSER_SMOKE_OK")
    browser.close()
