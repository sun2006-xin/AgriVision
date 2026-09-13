import os

from playwright.sync_api import sync_playwright


BASE_URL = os.environ.get("AGRIVISION_UI_BASE_URL", "http://127.0.0.1:5000")


with sync_playwright() as playwright:
    browser = playwright.chromium.launch(
        headless=True,
        args=["--disable-dev-shm-usage", "--disable-gpu", "--no-sandbox"],
    )
    page = browser.new_page()
    page_errors = []
    console_errors = []
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    def record_console_error(msg):
        location = msg.location or {}
        if msg.type != "error":
            return
        if "/api/offline_events/sync_status" in location.get("url", "") or "status of 503" in msg.text:
            return
        if msg.text.startswith("Failed to load resource:"):
            return
        if msg.type == "error":
            console_errors.append(msg.text)

    page.on("console", record_console_error)
    page.route(
        "**/api/offline_events/sync_status",
        lambda route: route.fulfill(status=503, content_type="application/json", body="{}"),
    )
    page.route(
        "**/api/dual_status**",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body='{"level":"正常","level_code":0,"disease_count":0,"white_count":0,"disease_ratio":0,"white_ratio":0,"green_ratio":1,"yolo_loaded":false,"yolo_enabled":false}',
        ),
    )
    page.route("**/favicon.ico", lambda route: route.fulfill(status=204, body=""))
    page.goto(f"{BASE_URL}/", wait_until="domcontentloaded")
    page.wait_for_selector("#eventSyncInfo")
    page.wait_for_timeout(500)
    status_text = page.locator("#eventSyncInfo").inner_text()
    assert status_text == "事件同步状态暂不可用", status_text
    assert page.locator("#statusBadge").count() == 1
    assert not page_errors
    assert not console_errors
    print("AGRIVISION_BROWSER_SMOKE_OK")
    browser.close()
